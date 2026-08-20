#!/usr/bin/env python3
"""
pipeline.py — KARAOKECREATOR: base PIENA (con cori, niente buchi) + voce PULITA,
in automatico (nessun intervento manuale per brano).

PROBLEMA risolto: col BS-Roformer la base restava "depressa" dove c'e' la voce
(buchi spettrali della rimozione-voce totale). Col modello KARAOKE la base resta
piena. Il coretto (anche self-overdub, stessa voce) si isola con UVR-MDX-NET Inst
girato SULLA VOCE (split armonico, metodo verificato dall'utente in UVR) e si
RIMETTE nella base. Una 2a passata Inst sul lead pulisce il residuo di coretto
dalla voce d'allineamento.

STADI:
  1) KARAOKE ensemble (aufr33/viperx + gabox_v2) sul MIX:
       (Instrumental) = BASE PIENA (niente buchi)       -> _base.wav   (temp)
       (Vocals)       = VOCE completa (lead + coretto)   -> _full.wav   (temp)
  2) Inst sulla VOCE (_full):
       (Vocals)       = lead 1a passata                  -> _lead1.wav  (temp)
       (Instrumental) = CORETTO isolato                  -> _coro.wav   (temp)
     Su brani con cori "veri" (cantanti diversi) il karaoke li ha gia' messi in
     _base e la VOCE e' solo-lead -> qui _coro esce ~vuoto e non si aggiunge nulla
     (auto-bilanciato). Su self-overdub il coretto era rimasto nella VOCE -> qui
     si estrae e si rimette nella base.
  3) Inst sul lead (_lead1), 2a passata -> LEAD PULITO -> lead_riferimento.wav
  REMIX (ffmpeg, somma no-normalize):
       base_piu_cori.wav = _base.wav + _coro.wav

Nomi stem finali INVARIATI (lead_riferimento.wav -> original_vocals.mp3,
base_piu_cori.wav -> original_instrumental.mp3). Modelli sovrascrivibili via env.
Reversibile via hotpatch.

Uso: python pipeline.py "brano.flac"
"""
import subprocess
import sys
import shutil
import os
from pathlib import Path

if len(sys.argv) < 2:
    print("Uso: python pipeline.py \"NOMEFILE\"")
    sys.exit(1)

INPUT = sys.argv[1]
if not Path(INPUT).exists():
    print(f"File non trovato: {INPUT}")
    sys.exit(1)

stem_name = Path(INPUT).stem
OUTDIR = Path(f"stems_{stem_name}")
OUTDIR.mkdir(exist_ok=True)

# STADIO 1: modello KARAOKE (base piena, cori nella base per i cori "veri").
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
KARAOKE_EXTRA = os.environ.get("KARAOKE_EXTRA", "mel_band_roformer_karaoke_gabox_v2.ckpt")
# STADIO 2/3: UVR-MDX-NET Inst sulla VOCE = split lead/coretto (metodo utente).
# Default Inst_HQ_3 (PRE-scaricato nell'immagine). Per l'ESATTO Inst_1 usato in UVR:
# env LEADBACK_MODEL=UVR-MDX-NET-Inst_1.onnx (va reso disponibile ad audio-separator).
LEADBACK_MODEL = os.environ.get("LEADBACK_MODEL", "UVR-MDX-NET-Inst_HQ_3.onnx")


def _wavs():
    return set(Path(".").glob("*.wav"))


def _find(new_files, needle):
    for f in sorted(new_files):
        if needle in f.name:
            return f
    return None


def _grab(new_files, needle, dst, descr):
    f = _find(new_files, needle)
    if f:
        shutil.copy(f, dst)
        print(f"  -> {dst.name} ({descr})")
        return True
    print(f"  !! stem '{needle}' non trovato per {dst.name}")
    return False


def _cleanup(new_files):
    for f in new_files:
        try:
            f.unlink()
        except Exception:
            pass


def _sep(args, descr):
    print(f"\n=== {descr} ===")
    subprocess.run(["audio-separator", *args, "--output_format", "WAV"], check=True)


def _sep_safe(args, descr):
    """Come _sep ma NON solleva: ritorna False se la separazione fallisce."""
    try:
        _sep(args, descr)
        return True
    except Exception as _e:
        print(f"  !! separazione fallita ({descr}): {_e} -> degrado")
        return False


def _rms_db(path):
    """RMS medio (dB) via ffmpeg volumedetect. None se non misurabile."""
    try:
        r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect",
                            "-f", "null", "-"], capture_output=True, text=True)
        for line in (r.stderr or "").splitlines():
            if "mean_volume:" in line:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
    except Exception:
        pass
    return None


_base = OUTDIR / "_base.wav"
_full = OUTDIR / "_full.wav"
_lead1 = OUTDIR / "_lead1.wav"
_coro = OUTDIR / "_coro.wav"
_lead = OUTDIR / "lead_riferimento.wav"
_out_base = OUTDIR / "base_piu_cori.wav"

# === STADIO 1: KARAOKE ensemble sul mix -> base piena + voce completa ===
_before = _wavs()
args = [INPUT, "-m", KARAOKE_MODEL]
if KARAOKE_EXTRA:
    args += ["--extra_models", KARAOKE_EXTRA]
if not _sep_safe(args, f"[1/3] KARAOKE ensemble ({KARAOKE_MODEL} + {KARAOKE_EXTRA})"):
    # fallback: solo modello principale
    _sep([INPUT, "-m", KARAOKE_MODEL], f"[1/3-fallback] KARAOKE ({KARAOKE_MODEL})")
_new = _wavs() - _before
_grab(_new, "(Instrumental)", _base, "BASE PIENA (karaoke), temp")
_grab(_new, "(Vocals)", _full, "VOCE completa (lead+coretto), temp")
_cleanup(_new)

# === STADIO 2: Inst sulla VOCE -> lead (1a) + CORETTO ===
if _full.exists():
    _before = _wavs()
    if _sep_safe([str(_full), "-m", LEADBACK_MODEL], f"[2/3] Inst ({LEADBACK_MODEL}) sulla voce -> lead + CORETTO"):
        _new = _wavs() - _before
        _grab(_new, "(Vocals)", _lead1, "lead 1a passata, temp")
        _grab(_new, "(Instrumental)", _coro, "CORETTO isolato -> va nella base")
        _cleanup(_new)
else:
    print("ATTENZIONE: voce completa mancante -> niente stadio 2")

# === STADIO 3: Inst sul lead (2a passata) -> LEAD PULITO ===
if _lead1.exists():
    _before = _wavs()
    if _sep_safe([str(_lead1), "-m", LEADBACK_MODEL], "[3/3] Inst sulla voce (2a passata) -> LEAD PULITO"):
        _new = _wavs() - _before
        if not _grab(_new, "(Vocals)", _lead, "LEAD PULITO, per allineamento"):
            shutil.copy(_lead1, _lead)
        _cleanup(_new)
    else:
        shutil.copy(_lead1, _lead)
elif _full.exists():
    shutil.copy(_full, _lead)   # fallback estremo

# === REMIX: base_piu_cori = base piena + coretto ===
# Se il coretto e' ~silenzioso (cori gia' nella base / self-overdub assente) non
# aggiunge nulla di udibile: auto-bilanciato.
_coro_ok = _coro.exists() and ((_rms_db(_coro) or -99) > -60)
if _base.exists() and _coro_ok:
    print("\n=== REMIX: base piena + coretto -> base_piu_cori.wav ===")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(_base), "-i", str(_coro),
        "-filter_complex", "amix=inputs=2:duration=longest:normalize=0",
        str(_out_base),
    ], check=True)
    print("  -> base_piu_cori.wav (base + coretto)")
elif _base.exists():
    shutil.copy(_base, _out_base)
    print("  -> base_piu_cori.wav (solo base: coretto assente/silenzioso)")

# pulizia temporanei
for _t in (_base, _full, _lead1, _coro):
    try:
        _t.unlink()
    except Exception:
        pass

if not _lead.exists() or not _out_base.exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: {[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

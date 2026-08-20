#!/usr/bin/env python3
"""
pipeline.py — KARAOKECREATOR: karaoke CON coretto + voce d'allineamento PULITA.

Metodo (indicato dall'utente, verificato coi suoi file UVR _lead_only/_backing_only):
UVR-MDX-NET Inst 1 girato SULLA VOCE tratta il lead come "vocals" e il coretto
(armonico) come "instrumental" -> in UNA passata separa lead/backing. Serve pero':
  - UNA passata Inst1 sulla voce  -> tira fuori il CORETTO (per la base karaoke);
  - una SECONDA passata Inst1 sul lead -> toglie il coretto RESIDUO dalla voce
    (voce PULITA per l'allineamento). Sempre lo STESSO modello.

STADI:
  1) VOCALE (BS-Roformer) sul MIX:
       (Vocals)       = cantato completo (lead+coretto) -> _full_vocals.wav (temp)
       (Instrumental) = BASE PURA (nessuna voce)        -> _base_pura.wav   (temp)
  2) Inst1 su _full_vocals:
       (Vocals)       = lead (1a passata)  -> _lead1.wav (temp)
       (Instrumental) = CORETTO            -> _coretto.wav (temp, va nella base)
  3) Inst1 su _lead1 (2a passata sulla voce):
       (Vocals)       = LEAD PULITO        -> lead_riferimento.wav (allineamento)
  REMIX (ffmpeg, somma no-normalize):
       base_piu_cori.wav = _base_pura.wav + _coretto.wav  (karaoke CON coretto)

Nomi stem finali INVARIATI (lead_riferimento.wav -> original_vocals.mp3,
base_piu_cori.wav -> original_instrumental.mp3): il server KC non cambia nulla.
Modelli sovrascrivibili via env. Reversibile via hotpatch.

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

VOICE_MODEL = os.environ.get("VOICE_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
# UVR-MDX-NET Inst (scelto dall'utente = "Inst 1"): sulla voce fa lo split lead/backing.
# Default = Inst_HQ_3 (PRE-scaricato nell'immagine, stessa famiglia "Inst"). Per usare
# ESATTAMENTE Inst_1 (non nell'immagine): env KARAOKE_MODEL=UVR-MDX-NET-Inst_1.onnx.
LEADBACK_MODEL = os.environ.get("KARAOKE_MODEL", "UVR-MDX-NET-Inst_HQ_3.onnx")


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
    """Come _sep ma NON solleva: ritorna False se la separazione fallisce (es. modello
    non disponibile) -> la pipeline degrada invece di far fallire tutto il job."""
    try:
        _sep(args, descr)
        return True
    except Exception as _e:
        print(f"  !! separazione fallita ({descr}): {_e} -> degrado")
        return False


_full = OUTDIR / "_full_vocals.wav"
_base = OUTDIR / "_base_pura.wav"
_lead1 = OUTDIR / "_lead1.wav"
_coro = OUTDIR / "_coretto.wav"
_lead = OUTDIR / "lead_riferimento.wav"
_out_base = OUTDIR / "base_piu_cori.wav"

# === STADIO 1: VOCALE sul MIX -> cantato completo + base pura ===
_before = _wavs()
_sep([INPUT, "-m", VOICE_MODEL], f"[1/3] VOCALE ({VOICE_MODEL}) sul mix -> cantato + base pura")
_new = _wavs() - _before
_grab(_new, "(Vocals)", _full, "cantato completo (lead+coretto), temp")
_grab(_new, "(Instrumental)", _base, "BASE PURA (nessuna voce), temp")
_cleanup(_new)

# === STADIO 2: Inst1 sulla voce -> lead (1a) + CORETTO ===
if _full.exists():
    _before = _wavs()
    if _sep_safe([str(_full), "-m", LEADBACK_MODEL], f"[2/3] Inst ({LEADBACK_MODEL}) sulla voce -> lead + CORETTO"):
        _new = _wavs() - _before
        _grab(_new, "(Vocals)", _lead1, "lead 1a passata, temp")
        _grab(_new, "(Instrumental)", _coro, "CORETTO isolato -> va nella base")
        _cleanup(_new)
else:
    print("ATTENZIONE: cantato completo mancante -> niente stadio 2")

# === STADIO 3: Inst sul lead (2a passata) -> LEAD PULITO per l'allineamento ===
if _lead1.exists():
    _before = _wavs()
    if _sep_safe([str(_lead1), "-m", LEADBACK_MODEL], f"[3/3] Inst sulla voce (2a passata) -> LEAD PULITO"):
        _new = _wavs() - _before
        if not _grab(_new, "(Vocals)", _lead, "LEAD PULITO, per allineamento"):
            shutil.copy(_lead1, _lead)   # fallback: tieni la 1a passata
        _cleanup(_new)
    else:
        shutil.copy(_lead1, _lead)       # 2a passata fallita: tieni la 1a
elif _full.exists():
    shutil.copy(_full, _lead)            # fallback estremo: cantato completo

# === REMIX: base_piu_cori = base pura + coretto ===
if _base.exists() and _coro.exists():
    print("\n=== REMIX: base pura + coretto -> base_piu_cori.wav ===")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(_base), "-i", str(_coro),
        "-filter_complex", "amix=inputs=2:duration=longest:normalize=0",
        str(_out_base),
    ], check=True)
    print("  -> base_piu_cori.wav (base + coretto)")
elif _base.exists():
    shutil.copy(_base, _out_base)
    print("  -> base_piu_cori.wav (solo base pura: coretto assente)")

# pulizia temporanei
for _t in (_full, _base, _lead1, _coro):
    try:
        _t.unlink()
    except Exception:
        pass

if not _lead.exists() or not _out_base.exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: {[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

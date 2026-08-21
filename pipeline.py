#!/usr/bin/env python3
"""
pipeline.py — KARAOKECREATOR: metodo mvsep per lead/cori (due Roformer in cascata).

Ricerca (mvsep "Lead/Back Vocals" leaderboard + audio-separator docs): i MIGLIORI
per separare i cori NON sono gli MDX-Net (Inst_1/HQ_3, modelli STRUMENTALI generici
che sul self-overdub non spaccano) ma i ROFORMER addestrati apposta per lead/back
(Mel-RoFormer Karaoke/Duet SDR 7.1). Workflow mvsep: BS-Roformer estrae le voci,
poi MelBand Roformer Karaoke (gabox V2) isola lead dai cori. I Roformer girano
NATIVI nel worker (niente torchvision/onnx2torch -> niente 'torchvision::nms').

STADI:
  1) KARAOKE ensemble (aufr33/viperx + gabox_v2) sul MIX:
       (Instrumental) = BASE PIENA (niente buchi)  -> _base.wav   (temp)
  2) BS-Roformer sul MIX:
       (Vocals)       = VOCI complete (lead+coretto, con corpo) -> _fullvox.wav (temp)
  3) KARAOKE Roformer (gabox V2) su _fullvox  [metodo mvsep lead/back]:
       (Vocals)       = LEAD               -> lead_riferimento.wav (allineamento)
       (Instrumental) = CORETTO (backing)  -> _coretto.wav (va nella base)
  REMIX (ffmpeg, somma no-normalize):
       base_piu_cori.wav = _base.wav + _coretto.wav

Auto-bilanciato: se _coretto e' ~silenzioso (nessun coro / gia' nella base) non
aggiunge nulla. Nomi stem finali INVARIATI. Modelli via env. Reversibile via hotpatch.

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

# STADIO 1: base piena (ensemble karaoke).
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
KARAOKE_EXTRA = os.environ.get("KARAOKE_EXTRA", "mel_band_roformer_karaoke_gabox_v2.ckpt")
# STADIO 2: estrazione voci complete (BS-Roformer, ottimo vocal separator).
VOICE_MODEL = os.environ.get("VOICE_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
# STADIO 3: split lead/coretto = KARAOKE Roformer (metodo mvsep). Roformer = NATIVO.
LEADBACK_MODEL = os.environ.get("LEADBACK_MODEL", "mel_band_roformer_karaoke_gabox_v2.ckpt")


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
    try:
        _sep(args, descr)
        return True
    except Exception as _e:
        print(f"  !! separazione fallita ({descr}): {_e} -> degrado")
        return False


def _rms_db(path):
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
_fullvox = OUTDIR / "_fullvox.wav"
_coro = OUTDIR / "_coretto.wav"
_lead = OUTDIR / "lead_riferimento.wav"
_out_base = OUTDIR / "base_piu_cori.wav"

# === STADIO 1: KARAOKE ensemble sul mix -> BASE PIENA ===
_before = _wavs()
args = [INPUT, "-m", KARAOKE_MODEL]
if KARAOKE_EXTRA:
    args += ["--extra_models", KARAOKE_EXTRA]
if not _sep_safe(args, f"[1/3] KARAOKE ensemble ({KARAOKE_MODEL} + {KARAOKE_EXTRA}) -> base piena"):
    _sep([INPUT, "-m", KARAOKE_MODEL], f"[1/3-fallback] KARAOKE ({KARAOKE_MODEL})")
_new = _wavs() - _before
_grab(_new, "(Instrumental)", _base, "BASE PIENA (karaoke), temp")
_cleanup(_new)

# === STADIO 2: BS-Roformer sul mix -> VOCI complete (lead+coretto) ===
_before = _wavs()
if _sep_safe([INPUT, "-m", VOICE_MODEL], f"[2/3] BS-Roformer ({VOICE_MODEL}) -> voci complete"):
    _new = _wavs() - _before
    _grab(_new, "(Vocals)", _fullvox, "VOCI complete (lead+coretto), temp")
    _cleanup(_new)

# === STADIO 3: KARAOKE Roformer sulle voci -> LEAD + CORETTO (metodo mvsep) ===
if _fullvox.exists():
    _before = _wavs()
    if _sep_safe([str(_fullvox), "-m", LEADBACK_MODEL], f"[3/3] KARAOKE Roformer ({LEADBACK_MODEL}) su voci -> lead + CORETTO"):
        _new = _wavs() - _before
        if not _grab(_new, "(Vocals)", _lead, "LEAD PULITO, per allineamento"):
            shutil.copy(_fullvox, _lead)
        _grab(_new, "(Instrumental)", _coro, "CORETTO (backing) -> va nella base")
        _cleanup(_new)
    else:
        shutil.copy(_fullvox, _lead)
else:
    print("ATTENZIONE: voci complete mancanti -> niente split lead/coretto")

# === REMIX: base + coretto (se udibile) ===
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

if not _lead.exists() and _base.exists():
    # fallback estremo: se manca del tutto la voce, non lasciare senza lead
    pass

for _t in (_base, _fullvox, _coro):
    try:
        _t.unlink()
    except Exception:
        pass

if not _lead.exists() or not _out_base.exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: {[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

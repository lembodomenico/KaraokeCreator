#!/usr/bin/env python3
"""
pipeline.py — KARAOKECREATOR: 2 STADI. Karaoke CON coretto, voce d'allineamento PULITA.

PROBLEMA (2026-08-20): su auto-sovraincisione (il "coretto" del finale = la STESSA voce
del solista, es. Cremonini) il modello karaoke sul mix scambia il coretto per lead e lo
TOGLIE dalla base -> il karaoke resta senza coretto. Servono i cori NELLA base e la voce
PULITA per allineare. Si fa a 2 stadi (come mvsep "extract vocals first"):

  STADIO 1 - VOCALE (BS-Roformer) sul MIX:
     (Vocals)       = TUTTO il cantato (lead+coretto) -> _full_vocals.wav (temp)
     (Instrumental) = BASE PURA (nessuna voce)        -> _base_pura.wav   (temp)
  STADIO 2 - KARAOKE (aufr33/viperx) sul CANTATO _full_vocals.wav:
     (Vocals)       = LEAD pulito  -> lead_riferimento.wav (allineamento, cori FUORI)
     (Instrumental) = CORETTO      -> _coretto.wav (temp)
  REMIX (ffmpeg, somma senza normalizzazione = niente sottrazioni instabili):
     base_piu_cori.wav = _base_pura.wav + _coretto.wav  (karaoke CON coretto)

Nomi stem finali INVARIATI (lead_riferimento.wav -> original_vocals.mp3,
base_piu_cori.wav -> original_instrumental.mp3): il server KC non cambia nulla.

Costo: 2 separazioni + 1 remix. Modelli gia' pre-scaricati nell'immagine.
Reversibile: rimettere la pipeline precedente via hotpatch.

Uso:
  python pipeline.py "Ligabue - Almeno credo.flac"
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
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")


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


_full = OUTDIR / "_full_vocals.wav"
_base = OUTDIR / "_base_pura.wav"
_coro = OUTDIR / "_coretto.wav"

# === STADIO 1: VOCALE sul MIX -> cantato completo + base pura ===
_before = _wavs()
_sep([INPUT, "-m", VOICE_MODEL], f"[1/2] VOCALE ({VOICE_MODEL}) sul mix -> cantato + base pura")
_new = _wavs() - _before
_grab(_new, "(Vocals)", _full, "cantato completo (lead+coretto), temp")
_grab(_new, "(Instrumental)", _base, "BASE PURA (nessuna voce), temp")
_cleanup(_new)

# === STADIO 2: KARAOKE sul CANTATO -> lead pulito + coretto ===
if _full.exists():
    _before = _wavs()
    _sep([str(_full), "-m", KARAOKE_MODEL], "[2/2] KARAOKE sul cantato -> lead pulito + coretto")
    _new = _wavs() - _before
    _grab(_new, "(Vocals)", OUTDIR / "lead_riferimento.wav", "LEAD pulito, per allineamento")
    _grab(_new, "(Instrumental)", _coro, "CORETTO isolato, temp")
    _cleanup(_new)
else:
    print("ATTENZIONE: cantato completo mancante -> niente stadio 2")

# === REMIX: base_piu_cori = base pura + coretto (somma, no normalize) ===
_out_base = OUTDIR / "base_piu_cori.wav"
if _base.exists() and _coro.exists():
    print("\n=== REMIX: base pura + coretto -> base_piu_cori.wav ===")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(_base), "-i", str(_coro),
        "-filter_complex", "amix=inputs=2:duration=longest:normalize=0",
        str(_out_base),
    ], check=True)
    print("  -> base_piu_cori.wav (base + coretto)")
elif _base.exists():
    # fallback: se il coretto non c'e', almeno la base pura
    shutil.copy(_base, _out_base)
    print("  -> base_piu_cori.wav (solo base pura: coretto assente)")

# pulizia temporanei
for _t in (_full, _base, _coro):
    try:
        _t.unlink()
    except Exception:
        pass

if not (OUTDIR / "lead_riferimento.wav").exists() or not _out_base.exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: {[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

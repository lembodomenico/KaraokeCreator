#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR (snella): SOLO voce+strumenti.

Produce in 'stems_<nome>/':
  - base_piu_cori.wav    (ensemble Roformer karaoke: base + cori, lead rimossa)
  - lead_riferimento.wav (la voce solista, per la trascrizione/sincronia)

NIENTE Demucs 6-stem (drums/bass/guitar/piano/other): quelli servono al
Separatore Strumenti, non a KaraokeCreator. Toglierli dimezza il tempo per job
e permette un'immagine molto piu' leggera (cold start piu' corto).

Uso:
  python pipeline.py "Ligabue - Almeno credo.flac"
"""

import subprocess
import sys
import shutil
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

print("\n=== Ensemble Roformer karaoke: base+cori e lead (GPU) ===")
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "--extra_models", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--ensemble_algorithm", "avg_wave",
    "--output_format", "WAV",
], check=True)

for f in Path(".").glob("*custom_ensemble*"):
    if "(Instrumental)" in f.name:
        shutil.copy(f, OUTDIR / "base_piu_cori.wav")
        print("  -> base_piu_cori.wav")
    elif "(Vocals)" in f.name:
        shutil.copy(f, OUTDIR / "lead_riferimento.wav")
        print("  -> lead_riferimento.wav")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

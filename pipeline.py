#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR (snella): SOLO voce+strumenti, UNA passata.

Una sola separazione Roformer karaoke (gabox_v2), come fa MVSep: NIENTE ensemble
a 2 modelli. Toglie la seconda passata (--extra_models) -> dimezza il tempo per
job.

Produce in 'stems_<nome>/':
  - base_piu_cori.wav    (base + cori, lead rimossa)  [da (Instrumental)]
  - lead_riferimento.wav (la voce solista, per la trascrizione/sincronia) [da (Vocals)]

NIENTE Demucs 6-stem (drums/bass/guitar/piano/other): quelli servono al
Separatore Strumenti, non a KaraokeCreator.

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

print("\n=== Roformer karaoke: base+cori e lead (GPU, modello singolo) ===")
# UNA SOLA separazione (gabox_v2). Niente --extra_models, niente
# --ensemble_algorithm: e' la seconda passata dell'ensemble a raddoppiare
# il tempo. Con un solo modello si dimezza, come MVSep.
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--output_format", "WAV",
], check=True)

# IMPORTANTE: con un modello singolo i file NON contengono piu' "custom_ensemble"
# nel nome (quello compariva solo con l'ensemble). audio-separator nomina i file
# col nome del modello. Quindi matcho solo su (Instrumental)/(Vocals).
found_inst = False
found_voc = False
for f in sorted(Path(".").glob("*.wav")):
    if "(Instrumental)" in f.name and not found_inst:
        shutil.copy(f, OUTDIR / "base_piu_cori.wav")
        print("  -> base_piu_cori.wav")
        found_inst = True
    elif "(Vocals)" in f.name and not found_voc:
        shutil.copy(f, OUTDIR / "lead_riferimento.wav")
        print("  -> lead_riferimento.wav")
        found_voc = True

if not found_inst or not found_voc:
    print(f"ATTENZIONE: stem non trovati (instrumental={found_inst}, vocals={found_voc}). "
          f"File .wav presenti: {[p.name for p in Path('.').glob('*.wav')]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR a 2 STEP (come il PC/karaoke_cli).

DUE separazioni sullo STESSO input, ognuna serve a una cosa diversa:

  STEP A  UVR-MDX-NET-Inst_HQ_3  ->  (Vocals) = LEAD + CORI  (TUTTE le voci)
          Usata SOLO per la TRASCRIZIONE. Cosi' Whisper/LAM sentono anche i CORI
          dei ritornelli: prima si trascriveva sulla sola voce solista e nei
          tratti cantati dai cori la traccia era quasi muta -> Whisper deragliava
          (riempiva il vuoto con ripetizioni allucinate). Con lead+cori il canto
          c'e' sempre e la trascrizione e' completa.

  STEP B  mel_band_roformer_karaoke_gabox_v2  ->  (Instrumental) = BASE + CORI
          Usata per il KARAOKE: il modello karaoke toglie SOLO il solista, i CORI
          restano nella base -> il karaoke non suona "nudo".

Produce in 'stems_<nome>/':
  - lead_riferimento.wav (LEAD+CORI, va alla trascrizione)
  - base_piu_cori.wav    (BASE+CORI, va al karaoke)

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


def _wavs():
    return set(Path(".").glob("*.wav"))


# === STEP A: voce COMPLETA (lead + cori) -> per la TRASCRIZIONE ===
print("\n=== STEP A: MDX Inst_HQ_3 -> voce completa (lead+cori) per la trascrizione ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", "UVR-MDX-NET-Inst_HQ_3.onnx",
    "--output_format", "WAV",
], check=True)
_new_a = _wavs() - _before
for f in sorted(_new_a):
    if "(Vocals)" in f.name:
        shutil.copy(f, OUTDIR / "lead_riferimento.wav")
        print("  -> lead_riferimento.wav (lead+cori)")
        break
# gli altri output dello STEP A (Instrumental base pura) non servono: pulizia
for f in _new_a:
    try:
        f.unlink()
    except Exception:
        pass

# === STEP B: base + cori (solista rimosso) -> per il KARAOKE ===
# ENSEMBLE aufr33/viperx + gabox_v2 (ripristinato: era la base "buona" originale,
# commit 3364d84). Due modelli mediati = base piu' pulita del solo gabox_v2.
print("\n=== STEP B: roformer karaoke ENSEMBLE (aufr33/viperx + gabox_v2) -> base+cori ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "--extra_models", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--output_format", "WAV",
], check=True)
_new_b = _wavs() - _before
for f in sorted(_new_b):
    if "(Instrumental)" in f.name:
        shutil.copy(f, OUTDIR / "base_piu_cori.wav")
        print("  -> base_piu_cori.wav (base+cori)")
        break
for f in _new_b:
    try:
        f.unlink()
    except Exception:
        pass

if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

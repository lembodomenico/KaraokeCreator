#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR: DUE separazioni con scopi diversi.

PERCHE' DUE (richiesta utente 2026-08-01 "UNISCI I CORI CON LA VOCE"):
la voce per l'ALLINEAMENTO deve contenere TUTTO il cantato (lead + CORI), altrimenti
dove cantano solo i cori lo stem e' MUTO -> il LAM/Whisper non ha voce da agganciare
-> l'allineamento DERAGLIA (verificato su Angelina Mango: buco a 10-13s = cori, poi
deraglia). Il modello "karaoke" mette il lead in (Vocals) e i cori nella base, quindi
NON va bene per allineare. Serve un modello VOCALE STANDARD: (Vocals) = lead + cori.

  1) Roformer KARAOKE ENSEMBLE (aufr33/viperx + gabox_v2)  -> per il PRODOTTO
       (Instrumental) = BASE + CORI  -> base_piu_cori.wav   (il karaoke che ascolta l'utente)
     [il suo (Vocals)=lead pulito NON serve: si scarta]

  2) Roformer VOCALE STANDARD (BS-Roformer 1297)           -> per l'ALLINEAMENTO
       (Vocals) = LEAD + CORI (tutto il cantato, NESSUN buco) -> lead_riferimento.wav
       (trascrizione DomAI + LAM si allineano su questo)
     [il suo (Instrumental)=base pura si scarta: la base del prodotto e' quella karaoke]

NB: il NOME dello stem voce resta `lead_riferimento.wav` (come prima) cosi' il server
KC non cambia nulla nel mapping (-> original_vocals.mp3): cambia solo il CONTENUTO, ora
voce+cori. La BASE del prodotto (base_piu_cori) e' INVARIATA (stesso karaoke ensemble).

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

# Modello vocale standard per la voce di allineamento (lead+cori). Sovrascrivibile
# via env se serve cambiarlo senza toccare il codice.
import os
VOICE_MODEL = os.environ.get(
    "VOICE_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
)


def _wavs():
    return set(Path(".").glob("*.wav"))


def _take(new_files, needle, dst_name, descr):
    """Copia in OUTDIR il primo wav di `new_files` il cui nome contiene `needle`."""
    for f in sorted(new_files):
        if needle in f.name:
            shutil.copy(f, OUTDIR / dst_name)
            print(f"  -> {dst_name} ({descr})")
            return True
    return False


def _cleanup(new_files):
    for f in new_files:
        try:
            f.unlink()
        except Exception:
            pass


# === 1) KARAOKE ENSEMBLE -> base+cori (PRODOTTO), invariato ===
print("\n=== [1/2] Roformer KARAOKE ensemble -> base+cori (Instrumental) ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "--extra_models", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--output_format", "WAV",
], check=True)
_new = _wavs() - _before
_take(_new, "(Instrumental)", "base_piu_cori.wav", "base+cori, per il karaoke")
_cleanup(_new)

# === 2) VOCALE STANDARD -> lead+cori (ALLINEAMENTO), niente buchi ===
print(f"\n=== [2/2] Roformer VOCALE standard ({VOICE_MODEL}) -> voce+cori (Vocals) ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", VOICE_MODEL,
    "--output_format", "WAV",
], check=True)
_new = _wavs() - _before
# NOME invariato (lead_riferimento.wav) ma contenuto = lead+cori
_take(_new, "(Vocals)", "lead_riferimento.wav", "LEAD+CORI, per trascrizione/allineamento")
_cleanup(_new)

if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

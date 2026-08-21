#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR: UNA separazione, DUE stem con scopi diversi.

Architettura (richiesta utente 2026-08-21):
  - VOCE PULITA (solo lead)   -> ALLINEAMENTO (trascrizione DomAI + LAM)
  - STRUMENTI + CORI (no lead) -> KARAOKE (la base che ascolta l'utente)
  - i due insieme (base+cori+lead) -> VOCE GUIDA (la monta il server KC dai 2 stem)

Il modello Roformer KARAOKE ENSEMBLE (aufr33/viperx + gabox_v2) produce ENTRAMBI
gli stem in UNA passata:
  (Vocals)       = LEAD PULITO   -> lead_riferimento.wav  (per allineamento)
  (Instrumental) = BASE + CORI   -> base_piu_cori.wav      (per il karaoke)

NB sul finale "coretto": dove Cremonini si sovraincide (stesso timbro), quel canto
resta col lead (in Vocals). E' un limite fisico dell'audio, non della pipeline.

NB nomi stem INVARIATI (lead_riferimento.wav / base_piu_cori.wav) cosi' il server KC
non cambia il mapping (-> original_vocals.mp3 / original_instrumental.mp3).

Modelli sovrascrivibili via env (KARAOKE_MODEL / KARAOKE_EXTRA) senza toccare il codice.

Uso:
  python pipeline.py "CESARE CREMONINI - ORA CHE NON HO PIU TE.flac"
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

# Roformer KARAOKE ensemble: (Vocals)=lead pulito, (Instrumental)=base+cori.
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
KARAOKE_EXTRA = os.environ.get("KARAOKE_EXTRA", "mel_band_roformer_karaoke_gabox_v2.ckpt")


def _wavs():
    return set(Path(".").glob("*.wav"))


def _take(new_files, needle, dst_name, descr):
    """Copia in OUTDIR il primo wav di `new_files` il cui nome contiene `needle`."""
    for f in sorted(new_files):
        if needle in f.name:
            shutil.copy(f, OUTDIR / dst_name)
            print(f"  -> {dst_name} ({descr})")
            return True
    print(f"  !! stem '{needle}' non trovato per {dst_name}")
    return False


def _cleanup(new_files):
    for f in new_files:
        try:
            f.unlink()
        except Exception:
            pass


# === UNA separazione: KARAOKE ensemble -> lead pulito + base+cori ===
print(f"\n=== Roformer KARAOKE ensemble ({KARAOKE_MODEL} + {KARAOKE_EXTRA}) ===")
_before = _wavs()
args = ["audio-separator", INPUT, "-m", KARAOKE_MODEL]
if KARAOKE_EXTRA:
    args += ["--extra_models", KARAOKE_EXTRA]
args += ["--output_format", "WAV"]
subprocess.run(args, check=True)
_new = _wavs() - _before

# (Vocals) = LEAD PULITO -> allineamento
_take(_new, "(Vocals)", "lead_riferimento.wav", "LEAD PULITO, per allineamento")
# (Instrumental) = BASE + CORI -> karaoke
_take(_new, "(Instrumental)", "base_piu_cori.wav", "BASE + CORI, per il karaoke")
_cleanup(_new)

if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

#!/usr/bin/env python3
"""
pipeline.py — KARAOKECREATOR: base karaoke PIENA (con cori) + voce d'allineamento PULITA.

Motivo del ritorno al modello KARAOKE (deciso dall'utente dopo il test "Donna...":
il 3-stadi con BS-Roformer lasciava (1) la base DEPRESSA dove c'e' la voce — buchi
spettrali della rimozione-voce totale — e (2) un residuo di coretto nella voce).

Il modello KARAOKE (ensemble aufr33/viperx + gabox_v2) fa gia' lo split giusto in
UNA sola separazione:
   (Instrumental) = BASE + CORI, piena (nessun buco)  -> base_piu_cori.wav
   (Vocals)       = SOLO LEAD (i cori vanno nella base) -> lead_riferimento.wav

Trade-off accettato: dove cantano SOLO i cori la voce e' muta (i cori restano
nell'audio karaoke, il loro testo si toglie in editing / AUTOSEG).

Nomi stem finali INVARIATI (il server KC non cambia nulla). Modelli sovrascrivibili
via env. Reversibile via hotpatch.

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

# Ensemble KARAOKE: principale aufr33/viperx (nitido) + extra gabox_v2 (fuso).
# Entrambi PRE-scaricati nell'immagine. Sovrascrivibili via env.
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
KARAOKE_EXTRA = os.environ.get("KARAOKE_EXTRA", "mel_band_roformer_karaoke_gabox_v2.ckpt")


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


_base = OUTDIR / "base_piu_cori.wav"
_lead = OUTDIR / "lead_riferimento.wav"

# === SEPARAZIONE KARAOKE (ensemble) sul mix ===
_before = _wavs()
args = [INPUT, "-m", KARAOKE_MODEL]
if KARAOKE_EXTRA:
    args += ["--extra_models", KARAOKE_EXTRA]
print(f"\n=== KARAOKE ensemble ({KARAOKE_MODEL} + {KARAOKE_EXTRA}) ===")
try:
    subprocess.run(["audio-separator", *args, "--output_format", "WAV"], check=True)
except Exception as _e:
    # se l'ensemble (--extra_models) non e' supportato, ricado sul solo principale
    print(f"  !! ensemble fallito ({_e}) -> riprovo col solo modello principale")
    subprocess.run(["audio-separator", INPUT, "-m", KARAOKE_MODEL, "--output_format", "WAV"], check=True)
_new = _wavs() - _before

# (Instrumental) = base + cori PIENA ;  (Vocals) = SOLO lead
_grab(_new, "(Instrumental)", _base, "BASE + CORI piena (karaoke)")
_grab(_new, "(Vocals)", _lead, "SOLO LEAD, per allineamento (cori tolti)")
_cleanup(_new)

if not _lead.exists() or not _base.exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: {[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

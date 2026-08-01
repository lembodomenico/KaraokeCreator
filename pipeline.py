#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR (snella): UNA separazione, voce+base.

RIPRISTINO "voce perfetta" (richiesta utente 2026-08-01). La voce torna a essere
il (Vocals) del Roformer karaoke ENSEMBLE = LEAD SOLISTA PULITO, come quando
"era perfetta" (stile MVSep). Rimosso lo STEP A MDX Inst_HQ_3 (lead+cori) che
sporcava la voce col bleed strumentale: ci si allinea sul lead pulito.
Bonus: una sola passata invece di due = piu' veloce.

  Roformer karaoke ENSEMBLE (aufr33/viperx + gabox_v2):
    - (Instrumental) = BASE + CORI  -> base_piu_cori.wav   (per il KARAOKE)
    - (Vocals)       = LEAD pulito  -> lead_riferimento.wav (per trascrizione/allineamento)

NB: togliendo l'MDX lead+cori, sui ritornelli cantati SOLO dai cori (lead muto)
la voce e' scarsa -> l'allineamento li' puo' soffrire. Sui brani in cui il lead
canta il ritornello (la maggioranza) resta perfetto.

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


# === UNA separazione: ENSEMBLE roformer karaoke (aufr33/viperx + gabox_v2) ===
# (Instrumental) = base+cori (karaoke) ; (Vocals) = LEAD pulito (voce perfetta).
print("\n=== Roformer karaoke ENSEMBLE -> base+cori (Instrumental) + lead pulito (Vocals) ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "--extra_models", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--output_format", "WAV",
], check=True)
_new = _wavs() - _before

for f in sorted(_new):
    if "(Instrumental)" in f.name:
        shutil.copy(f, OUTDIR / "base_piu_cori.wav")
        print("  -> base_piu_cori.wav (base+cori)")
        break
for f in sorted(_new):
    if "(Vocals)" in f.name:
        shutil.copy(f, OUTDIR / "lead_riferimento.wav")
        print("  -> lead_riferimento.wav (lead pulito, voce perfetta)")
        break
for f in _new:
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

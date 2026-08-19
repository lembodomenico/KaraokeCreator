#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR: UNA SOLA separazione (karaoke ensemble).

STORIA / SCELTA (2026-08-19, ordine esplicito utente "TOGLI I CORI DALLA VOCE"):
prima si facevano DUE separazioni -> la seconda (modello vocale standard) rimetteva
i CORI dentro la voce d'allineamento (lead+cori), per non lasciare stem muti dove
cantano solo i cori. MA nel finale corale il LAM ci ammassava le parole ("uh uh") ->
grumo, base inusabile. Ora si TORNA a una voce LEAD PULITA per allineare.

Come, in UNA SOLA passata: la separazione KARAOKE ENSEMBLE produce GIA' i due stem
che servono. Prima si teneva solo (Instrumental) e si BUTTAVA (Vocals): ora si tiene
anche (Vocals), che e' il lead pulito, e si ELIMINA la seconda separazione.

  Roformer KARAOKE ENSEMBLE (aufr33/viperx + gabox_v2), una separazione:
    (Instrumental) = BASE + CORI  -> base_piu_cori.wav   (il karaoke che ascolta l'utente:
                                     cori DENTRO, qualita' ensemble eccellente)
    (Vocals)       = LEAD pulito  -> lead_riferimento.wav (trascrizione DomAI + LAM si
                                     allineano su questo: CORI FUORI dalla voce)

NB: il NOME degli stem resta invariato (`lead_riferimento.wav` -> original_vocals.mp3,
`base_piu_cori.wav` -> original_instrumental.mp3) cosi' il server KC non cambia nulla nel
mapping: cambia solo il CONTENUTO della voce (ora lead pulito, non piu' lead+cori).

Trade-off noto: dove cantano SOLO i cori la voce lead e' muta -> per quello si toglie il
testo dei cori ("uh uh") in editing e c'e' AUTOSEG. Reversibile: git revert.

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


# === KARAOKE ENSEMBLE, UNA SOLA separazione: da qui prendo SIA la base+cori
#     (prodotto) SIA il lead PULITO (allineamento). ===
print("\n=== Roformer KARAOKE ensemble (aufr33/viperx + gabox_v2) -> base+cori E lead pulito ===")
_before = _wavs()
subprocess.run([
    "audio-separator", INPUT,
    "-m", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "--extra_models", "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "--output_format", "WAV",
], check=True)
_new = _wavs() - _before
# (Instrumental) = base+cori -> il karaoke (cori DENTRO, qualita' ensemble)
_take(_new, "(Instrumental)", "base_piu_cori.wav", "base+cori, per il karaoke")
# (Vocals) = LEAD pulito -> voce d'allineamento (cori FUORI). Prima veniva buttato.
_take(_new, "(Vocals)", "lead_riferimento.wav", "LEAD pulito (cori tolti), per trascrizione/allineamento")
_cleanup(_new)

if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

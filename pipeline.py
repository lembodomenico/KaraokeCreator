#!/usr/bin/env python3
"""
pipeline.py — Pipeline KARAOKECREATOR: voce d'allineamento = LEAD PULITO (2 stadi).

SCOPERTA (2026-08-19): il modello KARAOKE applicato al MIX lascia i cori nel suo
(Vocals) -> voce ancora sporca. Come fa mvsep ("Karaoke lead/back" = "Extract vocals
first"), il lead pulito si ottiene a DUE STADI: prima si estrae TUTTO il cantato
(modello vocale), poi si applica il karaoke SUL CANTATO -> lì il karaoke separa
davvero lead da cori. Il PRODOTTO (base+cori) resta la karaoke-ensemble sul mix,
INVARIATO: cambio solo la VOCE d'allineamento.

  A) KARAOKE ENSEMBLE sul MIX (aufr33/viperx + gabox_v2):
       (Instrumental) = BASE + CORI -> base_piu_cori.wav   (prodotto, invariato)
  B) modello VOCALE (BS-Roformer) sul MIX:
       (Vocals) = TUTTO il cantato (lead+cori) -> _full_vocals.wav (temporaneo)
  C) KARAOKE (aufr33/viperx) sul CANTATO _full_vocals.wav:
       (Vocals) = LEAD PULITO (cori tolti) -> lead_riferimento.wav (allineamento)

Nomi stem finali INVARIATI (lead_riferimento.wav -> original_vocals.mp3,
base_piu_cori.wav -> original_instrumental.mp3): il server KC non cambia nulla.

Costo: 3 separazioni (piu' lento del singolo), ma e' l'unico modo per una voce
davvero lead-pulita. Modelli tutti gia' pre-scaricati nell'immagine.

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

KARAOKE_MAIN = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
KARAOKE_XTRA = "mel_band_roformer_karaoke_gabox_v2.ckpt"
VOICE_MODEL = os.environ.get("VOICE_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")


def _wavs():
    return set(Path(".").glob("*.wav"))


def _find(new_files, needle):
    for f in sorted(new_files):
        if needle in f.name:
            return f
    return None


def _take(new_files, needle, dst_name, descr):
    f = _find(new_files, needle)
    if f:
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


def _sep(args, descr):
    print(f"\n=== {descr} ===")
    subprocess.run(["audio-separator", *args, "--output_format", "WAV"], check=True)


# === A) KARAOKE ENSEMBLE sul MIX -> base+cori (PRODOTTO, invariato) ===
_before = _wavs()
_sep([INPUT, "-m", KARAOKE_MAIN, "--extra_models", KARAOKE_XTRA],
     "[A] KARAOKE ensemble sul mix -> base+cori")
_new = _wavs() - _before
_take(_new, "(Instrumental)", "base_piu_cori.wav", "base+cori, per il karaoke")
_cleanup(_new)

# === B) VOCALE sul MIX -> TUTTO il cantato (lead+cori), temporaneo ===
_before = _wavs()
_sep([INPUT, "-m", VOICE_MODEL], f"[B] vocale ({VOICE_MODEL}) sul mix -> cantato completo")
_new = _wavs() - _before
_full = OUTDIR / "_full_vocals.wav"
_vf = _find(_new, "(Vocals)")
if _vf:
    shutil.copy(_vf, _full)
    print("  -> _full_vocals.wav (lead+cori, temporaneo)")
_cleanup(_new)

# === C) KARAOKE sul CANTATO -> LEAD PULITO (cori tolti) ===
if _full.exists():
    _before = _wavs()
    _sep([str(_full), "-m", KARAOKE_MAIN],
         "[C] KARAOKE sul cantato -> LEAD pulito (cori tolti)")
    _new = _wavs() - _before
    _take(_new, "(Vocals)", "lead_riferimento.wav", "LEAD pulito (2-stage), per allineamento")
    _cleanup(_new)
    try:
        _full.unlink()
    except Exception:
        pass
else:
    print("ATTENZIONE: cantato completo non prodotto dallo stadio B -> niente lead pulito")

if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

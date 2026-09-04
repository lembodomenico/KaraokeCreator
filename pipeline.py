# -*- coding: utf-8 -*-
# ⚠️ BOZZA, NON ANCORA NEL REPO DEL WORKER.
#
# Questa e' `pipeline.py` del worker RunPod con IN PIU' la pulizia della base.
# Il push su `main` fa partire da solo il rebuild dell'immagine su RunPod, cioe'
# manda tutto in produzione: per questo la bozza sta qui e non in KC_repo.
#
# ==========================================================================
#  PERCHE'
# ==========================================================================
# Nelle basi del server resta dentro della voce, e si sente. Misurato con
# htdemucs su 5 brani dell'archivio - quanta voce un separatore riesce ancora a
# tirare fuori dalla base, e quanti mezzi secondi stanno sopra -30 dB:
#
#     Nannini    -33,9 dB    29 tratti udibili
#     Ligabue    -30,1 dB    91
#     Britti     -30,4 dB   108
#     Mengoni    -31,0 dB   107
#     Afterhours -33,5 dB    37
#
# Il brano da cui e' partita la segnalazione (Nannini) e' il MENO colpito: non
# e' un caso isolato, e' in tutte le basi.
#
# Due cause che si sommano:
#   1. il roformer karaoke a volte scambia il solista per un coro e lo manda
#      nella base (a 86 s della Nannini si canta "SIAMO PRIGIONIERI DEL MONDO
#      DI IERI" e il (Vocals) del server e' a -99,8 dB: zero assoluto);
#   2. il residuo diffuso che ogni separatore lascia, ~21 dB sotto la base.
#
# ==========================================================================
#  COSA CAMBIA
# ==========================================================================
# La base per il karaoke, PRIMA di essere consegnata, si ripassa in htdemucs e
# si tiene la parte che NON e' voce, riportata al livello di prima.
#
# Risultato sugli stessi 5 brani: da 29-108 tratti di voce udibile a **ZERO**,
# su tutti. La musica resta: somiglianza 0,99 con la base di partenza, 1,000
# nei tratti dove non si canta.
#
# Il lead per l'allineamento NON si tocca: resta quello del roformer.
#
# ==========================================================================
#  DUE COSE PROVATE E SCARTATE (per non rifarle)
# ==========================================================================
# ⛔ RIMETTERE I CORI. La riseparazione toglie anche i cori (misurato:
#    Afterhours -2,9 dB, Ligabue -2,0 dB, su pochissimi tratti). Sembrava
#    risolvibile ricostruendo: strumenti puliti + (tutte le voci - il lead).
#    PEGGIORA, perche' dove il roformer ha PERSO il solista quel canto non sta
#    nel lead, non viene tolto, e rientra nella base coi cori:
#        Ligabue 91 -> 98 | Britti 108 -> 81 | Mengoni 107 -> 252 | After 37 -> 42
#    Su Mengoni la voce udibile piu' che raddoppia.
#
# ⛔ htdemucs_ft (il modello "rifinito"). Costa 4 volte il tempo e NON e'
#    meglio: pulizia identica (0 tratti anche lui), cori uguali (-2,92 contro
#    -2,86), musica un filo meno fedele (0,9926 contro 0,9946). Si usa
#    htdemucs normale.
#
# ==========================================================================
#  COSTO E SICUREZZA
# ==========================================================================
# ~50 secondi di GPU in piu' per brano (audio di 4 minuti).
#
# ⚠️ NEL Dockerfile, se no il modello si scarica a ogni run e allunga il cold
#    start:
#        RUN audio-separator --download_model_only -m htdemucs.yaml || true
#
# Se la pulizia fallisce, la base resta ESATTAMENTE quella di prima: meglio una
# base col brusio che nessuna base. Spegnimento senza rebuild: PULISCI_BASE=0.

import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
stem_name = Path(INPUT).stem
OUTDIR = Path(f"stems_{stem_name}")
OUTDIR.mkdir(exist_ok=True)

# Roformer KARAOKE ensemble: (Vocals)=lead pulito, (Instrumental)=base+cori.
KARAOKE_MODEL = os.environ.get("KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
KARAOKE_EXTRA = os.environ.get("KARAOKE_EXTRA", "mel_band_roformer_karaoke_gabox_v2.ckpt")

# --- pulizia della base -----------------------------------------------------
PULISCI_BASE = os.environ.get("PULISCI_BASE", "1") != "0"
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs.yaml")


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


def _pulisci_base():
    """Toglie dalla base la voce rimasta dentro. (fatto, perche_no)

    La base si ripassa in htdemucs e si tiene la parte senza voce. Il livello
    si riporta a quello di prima: htdemucs restituisce lo stem un paio di dB
    piu' forte, e una base piu' forte non deve passare per "migliore".
    """
    import numpy as np

    SR = 44100

    def leggi(path):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2",
                            "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
        return np.frombuffer(r.stdout, dtype=np.float32).astype(np.float32).reshape(-1, 2)

    base = OUTDIR / "base_piu_cori.wav"
    print(f"\n=== pulizia della base ({DEMUCS_MODEL}) ===")
    _b = _wavs()
    subprocess.run(["audio-separator", str(base), "-m", DEMUCS_MODEL,
                    "--output_format", "WAV"], check=True)
    nuovi = _wavs() - _b

    # ⚠️ PROVATO SUL SERIO, e qui la prima versione sbagliava: audio-separator
    #    con htdemucs NON da' un file "senza voce", da' QUATTRO stem separati -
    #    (Bass), (Drums), (Other), (Vocals). Cercando "instrumental" non si
    #    trovava niente e la pulizia veniva saltata ogni volta, in silenzio.
    #    La base senza voce e' la somma dei tre che non sono la voce.
    #    Verificato: quella somma e' IDENTICA (0,9996) al `no_vocals` che
    #    restituisce demucs chiamato a due stem.
    pezzi = {}
    for f in sorted(nuovi):
        n = f.name.lower()
        for quale in ("bass", "drums", "other", "vocals"):
            if "(%s)" % quale in n:
                pezzi[quale] = f
    mancano = [q for q in ("bass", "drums", "other") if q not in pezzi]
    if mancano:
        _cleanup(nuovi)
        return False, ("htdemucs non ha dato gli stem %s (trovati: %s)"
                       % (mancano, [f.name for f in nuovi]))

    PRIMA = leggi(base)
    DOPO = None
    for quale in ("bass", "drums", "other"):
        pezzo = leggi(pezzi[quale])
        DOPO = pezzo if DOPO is None else DOPO[:min(len(DOPO), len(pezzo))] + pezzo[:min(len(DOPO), len(pezzo))]
    n = min(len(PRIMA), len(DOPO))
    if n < SR:
        _cleanup(nuovi)
        return False, "lo stem senza voce e' troppo corto"
    PRIMA, DOPO = PRIMA[:n], DOPO[:n]

    def liv(x):
        return 20 * np.log10(float(np.sqrt((x ** 2).mean())) + 1e-12)

    DOPO = DOPO * (10 ** ((liv(PRIMA) - liv(DOPO)) / 20.0))
    print("   base %.2f dB -> %.2f dB" % (liv(PRIMA), liv(DOPO)))

    grezzo = np.clip(DOPO, -1.0, 1.0).astype(np.float32).tobytes()
    # ⚠️ 24 bit ESPLICITI. Senza dirlo, ffmpeg scrive il wav in virgola mobile
    #    a 32 bit: il file raddoppia di peso e cambia formato rispetto a quello
    #    che la pipeline aveva prodotto (16 bit). A 24 bit il file resta
    #    leggibile da tutti e la qualita' e' migliore dell'originale.
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR),
                        "-ac", "2", "-i", "-", "-c:a", "pcm_s24le", str(base)],
                       input=grezzo, capture_output=True)
    _cleanup(nuovi)
    if p.returncode != 0 or not base.exists() or base.stat().st_size < 10000:
        return False, "la scrittura della base nuova non e' riuscita"
    return True, ""


if PULISCI_BASE and (OUTDIR / "base_piu_cori.wav").exists():
    # ⚠️ Copia di sicurezza: se qualcosa va storto la base torna quella di
    #    prima. Meglio una base col brusio che una base rotta.
    salvata = OUTDIR / "_base_prima_della_pulizia.wav"
    try:
        shutil.copy(OUTDIR / "base_piu_cori.wav", salvata)
        ok, perche = _pulisci_base()
        if ok:
            print("  -> base ripulita dalla voce rimasta dentro")
            salvata.unlink()
        else:
            shutil.copy(salvata, OUTDIR / "base_piu_cori.wav")
            salvata.unlink()
            print(f"  !! pulizia saltata, base invariata: {perche}")
    except Exception as e:
        try:
            if salvata.exists():
                shutil.copy(salvata, OUTDIR / "base_piu_cori.wav")
                salvata.unlink()
        except Exception:
            pass
        print(f"  !! pulizia saltata, base invariata: {e}")


if not (OUTDIR / "lead_riferimento.wav").exists() or not (OUTDIR / "base_piu_cori.wav").exists():
    print(f"ATTENZIONE: stem mancanti in {OUTDIR}: "
          f"{[p.name for p in OUTDIR.iterdir()]}")

print(f"\nFATTO. Stem in: {OUTDIR.resolve()}")
for f in sorted(OUTDIR.iterdir()):
    print("   ", f.name)

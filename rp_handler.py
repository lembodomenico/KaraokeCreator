"""
rp_handler.py — Handler RunPod per l'endpoint KARAOKECREATOR (snello).
Fa SOLO: separazione voce+strumenti (pipeline.py Roformer) -> export mp3 ->
zip -> upload FTP -> link. NIENTE accordi (madmom non e' in questa immagine).
Input:  { "audio_url": "https://...", "lyrics": "<testo opzionale>" }
Output: { "stems": [...], "download_url": "...", "timestamps": [...] }
Env dell'endpoint: FTP_HOST, FTP_USER, FTP_PASS, FTP_DIR, PUBLIC_BASE_URL

NOTE VELOCITA' (questa versione):
 - I due encode mp3 girano IN PARALLELO (non in sequenza): ~meta' tempo.
 - NIENTE loudnorm qui: KC normalizza gia' lui in fase 2 (_normalize_audio).
   Farlo anche qui era doppio lavoro e rallentava molto l'encode.
 - base_piu_cori -> 320k (la ascolta l'utente).
 - lead_riferimento -> 128k MONO (va SOLO alla trascrizione: 320k era spreco).
"""
import base64
import ftplib
import os
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
import requests
import runpod

# UA "da browser": alcuni hosting (WAF/ModSecurity) bloccano con 403 lo UA
# di default 'python-requests/*'. Con questo lo scaricamento di audio_url passa.
_DL_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; KaraokeCreator/1.0)"}

WORK = Path("/tmp/jobs")
WORK.mkdir(parents=True, exist_ok=True)

# Destinazione FTP dei risultati. Default = VPS NUOVO (karaokecreator.karadom.it).
# Il vecchio server (185.73.8.100 / karaokedom.it / .../karaokecreator) e' DISMESSO
# e non va MAI usato: _no_old() neutralizza ogni suo residuo che arrivasse dall'env
# dell'endpoint. La password NON e' cablata: arriva dall'input del job (repo pubblico).
_NEW_FTP_HOST    = "94.72.100.151"
_NEW_FTP_USER    = "kcftp"
_NEW_FTP_DIR     = "/risultati"
_NEW_PUBLIC_BASE = "https://karaokecreator.karadom.it/risultati"

def _no_old(val, new):
    """Se il valore e' vuoto o punta al vecchio server dismesso, usa quello nuovo."""
    if not val:
        return new
    bad = ("185.73.8.100", "karaokedom.it", "/karaokecreator", "dlembo")
    return new if any(b in val for b in bad) else val

FTP_HOST = _no_old(os.environ.get("FTP_HOST"), _NEW_FTP_HOST)
FTP_USER = _no_old(os.environ.get("FTP_USER"), _NEW_FTP_USER)
FTP_PASS = os.environ.get("FTP_PASS")
FTP_DIR = _no_old(os.environ.get("FTP_DIR"), _NEW_FTP_DIR)
PUBLIC_BASE_URL = _no_old(os.environ.get("PUBLIC_BASE_URL"), _NEW_PUBLIC_BASE)


def _get_audio(job_dir: Path, inp: dict) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    if "audio_url" in inp:
        url = inp["audio_url"]
        suffix = Path(url.split("?")[0]).suffix or ".wav"
        dst = job_dir / f"input{suffix}"
        last = None
        for attempt in range(3):
            try:
                with requests.get(url, timeout=180, headers=_DL_HEADERS, stream=True) as r:
                    r.raise_for_status()
                    with dst.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk:
                                f.write(chunk)
                if dst.exists() and dst.stat().st_size > 0:
                    return dst
                last = "file scaricato vuoto"
            except Exception as e:
                last = e
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"download audio_url fallito: {last}")
    if "audio_base64" in inp:
        dst = job_dir / "input.wav"
        dst.write_bytes(base64.b64decode(inp["audio_base64"]))
        return dst
    raise ValueError("Fornire 'audio_url' oppure 'audio_base64'.")


def _ftp_upload(local_file: Path, remote_name: str) -> str:
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login(FTP_USER, FTP_PASS)
    # crea la cartella di destinazione se non esiste (percorso assoluto)
    try:
        ftp.cwd(FTP_DIR)
    except Exception:
        cur = ""
        for part in [p for p in FTP_DIR.split("/") if p]:
            cur += "/" + part
            try:
                ftp.cwd(cur)
            except Exception:
                try:
                    ftp.mkd(cur)
                    ftp.cwd(cur)
                except Exception:
                    pass
    with local_file.open("rb") as f:
        ftp.storbinary(f"STOR {remote_name}", f)
    ftp.quit()
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{remote_name}"


def _encode_args(wav: Path, mp3: Path):
    """Argomenti ffmpeg per ogni stem.
    - lead_riferimento: 128k MONO (solo trascrizione) -> encode velocissimo.
    - base_piu_cori (e ogni altro): 320k stereo (lo ascolta l'utente).
    NIENTE loudnorm: lo fa KC in fase 2 (evita doppia normalizzazione lenta).
    """
    name = wav.stem.lower()
    if "lead" in name or "vocal" in name or "voce" in name:
        return ["ffmpeg", "-y", "-i", str(wav),
                "-ac", "1", "-ar", "44100", "-b:a", "128k", str(mp3)]
    return ["ffmpeg", "-y", "-i", str(wav),
            "-ar", "44100", "-b:a", "320k", str(mp3)]


# === DomAI: Whisper (faster-whisper) — trascrive la voce isolata quando il
# server NON passa un testo. Modello caricato una sola volta (cache globale),
# sulla GPU del pod (fallback CPU se cuda non disponibile). ===
_WHISPER = None


def _get_whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        try:
            _WHISPER = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
        except Exception:
            _WHISPER = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    return _WHISPER


def whisper_transcribe(vocals_path: str, lang=None) -> str:
    """Ritorna il testo trascritto dalla voce isolata (stringa). lang=None -> auto-detect.
    condition_on_previous_text=False: NON collassa i ritornelli ripetitivi (di default
    Whisper evita di ripetersi e "riassume" i loop ossessivi -> meno parole del cantato
    -> LAM deraglia). Cosi' trascrive ogni segmento indipendente e cattura le ripetizioni."""
    model = _get_whisper()
    segs, _info = model.transcribe(
        vocals_path, language=lang, beam_size=5, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join((s.text or "").strip() for s in segs).strip()


def handler(job: dict) -> dict:
    job_id = job.get("id", uuid.uuid4().hex[:12])
    inp = job.get("input", {}) or {}
    # Override della destinazione FTP dall'input del job (il server passa i valori
    # correnti: cosi' si puo' spostare lo storage senza toccare l'env dell'endpoint).
    global FTP_HOST, FTP_USER, FTP_PASS, FTP_DIR, PUBLIC_BASE_URL
    FTP_HOST = inp.get("ftp_host") or FTP_HOST
    FTP_USER = inp.get("ftp_user") or FTP_USER
    FTP_PASS = inp.get("ftp_pass") or FTP_PASS
    FTP_DIR = inp.get("ftp_dir") or FTP_DIR
    PUBLIC_BASE_URL = inp.get("public_base_url") or PUBLIC_BASE_URL
    d = WORK / job_id
    try:
        audio = _get_audio(d, inp)

        # SEPARAZIONE (solo Roformer voce+strumenti, una passata).
        # Cattura stdout+stderr: se pipeline.py fallisce, l'errore VERO (traceback)
        # torna nel messaggio d'errore del job invece di restare nascosto nei log RunPod.
        _p = subprocess.run(["python", "/app/pipeline.py", str(audio)],
                            cwd=str(d), capture_output=True, text=True)
        # Ristampa comunque l'output nei log del worker (per lo storico su RunPod).
        if _p.stdout:
            print(_p.stdout)
        if _p.stderr:
            print(_p.stderr)
        if _p.returncode != 0:
            _tail = (_p.stderr or _p.stdout or "").strip().splitlines()[-12:]
            return {"error": "Separazione fallita (pipeline.py):\n" + "\n".join(_tail)}
        stems_dir = next(d.glob("stems_*"))

        # === DomAI: trascrizione + allineamento del cantato ===
        # FLUSSO (deciso con l'utente):
        #  1. DomAI (Whisper, GPU del pod) trascrive SEMPRE la voce isolata: e' il
        #     "cantato reale" di riferimento, e conta quante parole ci sono davvero.
        #  2. Il testo ONLINE che il server ha gia' scelto (inp["lyrics"]) si usa
        #     SOLO SE COPRE il cantato, cioe' ha almeno ~quante parole ne ha sentite
        #     DomAI. Cosi' NON si prende piu' un testo con MENO parole del cantato
        #     (il controsenso). Se copre -> parole "pulite" dal testo scritto.
        #  3. Se il testo online NON copre (o non c'e') -> resta DomAI puro.
        #  4. LAM allinea il testo scelto sul cantato.
        # I timestamp DomAI vengono restituiti SEMPRE (anche se l'allineamento non
        # copre tutto): il server li tiene come rete finale + puo' cercare altri
        # testi (Genius/LyricsOvh) se sono segnalati incompleti.
        timestamps = None
        timestamps_incomplete = False
        testo = ""
        text_source = None
        testo_server = (inp.get("lyrics") or "").strip()
        vocals_wav = None
        for wav in stems_dir.glob("*.wav"):
            n = wav.stem.lower()
            if "lead" in n or "vocal" in n or "voce" in n:
                vocals_wav = wav
                break

        def _wc(s):
            return len([w for w in (s or "").split() if w.strip()])

        if vocals_wav is not None:
            import lam_align

            def _align(_txt):
                """Allinea _txt sul cantato. Ritorna (words|None, ok:bool)."""
                if not _txt:
                    return None, False
                try:
                    sd = lam_align.align_as_scribe_data(str(vocals_wav), _txt, "/app/lam")
                    _w = sd.get("words") or []
                    return (_w or None), bool(sd.get("ok") and _w)
                except Exception:
                    return None, False

            # 1) DomAI trascrive SEMPRE (cantato reale)
            testo_domai = ""
            try:
                testo_domai = whisper_transcribe(str(vocals_wav), inp.get("lang") or None)
            except Exception:
                testo_domai = ""
            n_domai = _wc(testo_domai)

            # 2) scelta testo: tra i CANDIDATI passati dal server (delimitati da
            #    @@@CAND@@@), scelgo quello col n. parole PIU' VICINO al cantato reale
            #    (n_domai da DomAI/Whisper) -> non un testo gonfiato ne' uno incompleto,
            #    ma quello che COMBACIA col cantato. Lo uso solo se ragionevolmente
            #    vicino (entro il 35% del conteggio cantato); altrimenti DomAI puro.
            _cand_list = [c.strip() for c in testo_server.split("@@@CAND@@@")
                          if c and c.strip()] if testo_server else []
            _best, _bestd = None, None
            for _c in _cand_list:
                _d = abs(_wc(_c) - n_domai)
                if _bestd is None or _d < _bestd:
                    _bestd, _best = _d, _c
            if _best is not None and (n_domai == 0 or _bestd <= 0.35 * n_domai):
                testo, text_source = _best, "lyrics"
                print(f"[TESTO] scelto candidato {_wc(_best)} parole (cantato ~{n_domai}, "
                      f"scarto {_bestd}) su {len(_cand_list)} candidati", flush=True)
            else:
                testo, text_source = testo_domai, ("whisper" if testo_domai else None)

            # 4) allinea il testo scelto
            _w, _ok = _align(testo)
            if _w:
                timestamps, timestamps_incomplete = _w, (not _ok)

            # se avevo scelto l'online ma NON copre, e DomAI aveva sentito parole,
            # ripiego su DomAI puro (allinea di sicuro cio' che ha sentito)
            if (timestamps is None or timestamps_incomplete) and \
               text_source == "lyrics" and testo_domai:
                _w2, _ok2 = _align(testo_domai)
                if _w2 and (_ok2 or timestamps is None):
                    timestamps = _w2
                    timestamps_incomplete = not _ok2
                    testo, text_source = testo_domai, "whisper"

        # EXPORT mp3 IN PARALLELO (niente loudnorm: lo fa KC)
        final = d / "final"
        final.mkdir(exist_ok=True)
        procs = []
        for wav in sorted(stems_dir.glob("*.wav")):
            mp3 = final / f"{wav.stem}.mp3"
            procs.append((subprocess.Popen(_encode_args(wav, mp3),
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL), wav, mp3))
        for p, wav, mp3 in procs:
            rc = p.wait()
            if rc != 0 or not mp3.exists() or mp3.stat().st_size == 0:
                raise RuntimeError(f"encode fallito per {wav.name} (rc={rc})")

        # ZIP
        zip_path = d / "result.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for mp3 in sorted(final.glob("*.mp3")):
                z.write(mp3, arcname=mp3.name)

        # UPLOAD via FTP -> link col tuo dominio
        url = _ftp_upload(zip_path, f"{job_id}.zip")
        return {
            "stems": [p.name for p in sorted(final.glob("*.mp3"))],
            "chords": False,
            "download_url": url,
            "timestamps": timestamps,                  # tempi DomAI: SEMPRE (rete finale)
            "timestamps_incomplete": bool(timestamps_incomplete),
            "text": testo or None,                     # testo EFFETTIVAMENTE usato
            "text_source": text_source,                # "lyrics" (online) | "whisper" (DomAI)
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"Step fallito: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

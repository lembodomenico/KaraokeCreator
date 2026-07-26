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

FTP_HOST = os.environ.get("FTP_HOST")
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")
FTP_DIR = os.environ.get("FTP_DIR", "/public_html/karaokecreator/risultati")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://karadom.it/karaokecreator/risultati")


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
    ftp.cwd(FTP_DIR)
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
    d = WORK / job_id
    try:
        audio = _get_audio(d, inp)

        # SEPARAZIONE (solo Roformer voce+strumenti, una passata)
        subprocess.run(["python", "/app/pipeline.py", str(audio)],
                       cwd=str(d), check=True)
        stems_dir = next(d.glob("stems_*"))

        # === LAM: allineamento testo sul cantato (solo se il server ha passato
        # "lyrics"). Gira sullo STESSO worker gia' caldo -> nessun cold start in
        # piu'. Fail-safe: qualsiasi problema -> timestamps=None, il server usa
        # il suo fallback (Scribe). ===
        timestamps = None
        testo = (inp.get("lyrics") or "").strip()
        testo_da_whisper = False
        # voce isolata: serve sia a Whisper (trascrizione) sia a LAM (allineamento)
        vocals_wav = None
        for wav in stems_dir.glob("*.wav"):
            n = wav.stem.lower()
            if "lead" in n or "vocal" in n or "voce" in n:
                vocals_wav = wav
                break

        # DomAI: se il server NON ha passato un testo, lo trascrive Whisper (GPU
        # del pod) invece di ripiegare su Scribe. Poi LAM allinea come sempre.
        if not testo and vocals_wav is not None:
            try:
                testo = whisper_transcribe(str(vocals_wav), inp.get("lang") or None)
                testo_da_whisper = bool(testo)
            except Exception as _e:
                testo = ""

        # LAM: allinea il testo (fornito dal server o trascritto da Whisper) sul
        # cantato. Fail-safe: qualsiasi problema -> timestamps=None.
        if testo and vocals_wav is not None:
            try:
                import lam_align
                sd = lam_align.align_as_scribe_data(str(vocals_wav), testo, "/app/lam")
                if sd.get("ok") and sd.get("words"):
                    timestamps = sd["words"]
                else:
                    timestamps = {"ok": False, "reason": sd.get("reason")}
            except Exception as _e:
                timestamps = {"ok": False, "reason": f"LAM errore: {_e}"}

        # DomAI: se il testo FORNITO non ha allineato bene (score basso / testo
        # bucato o incompleto) e non era gia' una trascrizione, DomAI trascrive lui
        # la voce e riallinea -> "rileva da solo le parole" anche col testo scarso.
        _lam_ok = isinstance(timestamps, list) and len(timestamps) > 0
        if not _lam_ok and not testo_da_whisper and vocals_wav is not None:
            try:
                import lam_align
                _t2 = whisper_transcribe(str(vocals_wav), inp.get("lang") or None)
                if _t2:
                    testo = _t2
                    testo_da_whisper = True
                    sd = lam_align.align_as_scribe_data(str(vocals_wav), testo, "/app/lam")
                    if sd.get("ok") and sd.get("words"):
                        timestamps = sd["words"]
                    else:
                        timestamps = {"ok": False, "reason": sd.get("reason")}
            except Exception as _e:
                pass

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
            "timestamps": timestamps,
            "text": testo or None,                     # testo usato (fornito o Whisper)
            "text_source": "whisper" if testo_da_whisper else ("lyrics" if testo else None),
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"Step fallito: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

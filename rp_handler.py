"""
rp_handler.py — Handler RunPod per l'endpoint KARAOKECREATOR (snello).
Fa SOLO: separazione voce+strumenti (pipeline.py Roformer) -> export mp3 ->
zip -> upload FTP -> link. NIENTE accordi (madmom non e' in questa immagine).
Input:  { "audio_url": "https://..." }  oppure  { "audio_base64": "..." }
Output: { "stems": [...], "download_url": "https://.../<job>.zip" }
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
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"Step fallito: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

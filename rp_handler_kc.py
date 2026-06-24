"""
rp_handler.py — Handler RunPod per l'endpoint KARAOKECREATOR (snello).

Fa SOLO: separazione voce+strumenti (pipeline.py Roformer) -> export mp3 320 ->
zip -> upload FTP -> link. NIENTE accordi (madmom non e' in questa immagine).

Input:  { "audio_url": "https://..." }  oppure  { "audio_base64": "..." }
Output: { "stems": [...], "download_url": "https://.../<job>.zip" }

Env dell'endpoint: FTP_HOST, FTP_USER, FTP_PASS, FTP_DIR, PUBLIC_BASE_URL
"""

import base64
import ftplib
import os
import subprocess
import uuid
import zipfile
from pathlib import Path

import requests
import runpod

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
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return dst
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


def handler(job: dict) -> dict:
    job_id = job.get("id", uuid.uuid4().hex[:12])
    inp = job.get("input", {}) or {}
    d = WORK / job_id

    try:
        audio = _get_audio(d, inp)

        # SEPARAZIONE (solo Roformer voce+strumenti)
        subprocess.run(["python", "/app/pipeline.py", str(audio)],
                       cwd=str(d), check=True)
        stems_dir = next(d.glob("stems_*"))

        # EXPORT mp3 320 normalizzato
        final = d / "final"
        final.mkdir(exist_ok=True)
        for wav in sorted(stems_dir.glob("*.wav")):
            mp3 = final / f"{wav.stem}.mp3"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(wav),
                "-af", "loudnorm=I=-14:TP=-1:LRA=11",
                "-ar", "44100", "-b:a", "320k", str(mp3),
            ], check=True)

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

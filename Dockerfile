# Dockerfile — endpoint KARAOKECREATOR (snello): SOLO voce+strumenti.
# Rispetto all'immagine "completa": niente Demucs 6-stem, niente venv accordi/
# madmom. Restano solo i 2 modelli Roformer karaoke, gia' pre-scaricati nel
# build -> niente download a runtime -> cold start molto piu' corto.
#
# Cosa NON c'e' piu' (e quindi cosa NON puo' fare questo endpoint):
#   - drums/bass/guitar/piano/other (Demucs)  -> sta nell'endpoint "Separatore"
#   - accordi (madmom)                        -> sta nell'endpoint "Separatore"

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app
ARG CACHE_BUST=20260814d
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCHAUDIO_USE_BACKEND_DISPATCHER=0

# Sistema: solo ffmpeg (niente Python 3.10/madmom: gli accordi non servono qui)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# torch/torchaudio FISSI (come nell'immagine validata)
RUN pip install --no-cache-dir "torch==2.8.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128

# SOLO audio-separator (NIENTE demucs). + runtime RunPod.
# SOLO audio-separator (NIENTE demucs). + runtime RunPod.
# onnxruntime-gpu per CUDA 12.x (il base image e' cu128): NON fare il downgrade
# a CPU, altrimenti i Roformer girano su CPU (lentissimi). Teniamo la GPU.
# --ignore-installed cryptography: nell'immagine base cryptography e' installata da
# Debian (apt) SENZA file RECORD -> pip non riesce a disinstallarla e la build muore
# (uninstall-no-record-file). Cosi' pip installa la sua versione senza toccare quella di sistema.
# audioread: audio-separator riceve input .mp3; soundfile NON legge mp3 -> librosa
# ripiega su audioread (backend ffmpeg, gia' installato). Senza -> ModuleNotFoundError
# 'audioread' e la separazione muore subito (exit 1 in ~20s). Va installato esplicitamente.
RUN pip install --no-cache-dir --ignore-installed cryptography "audio-separator[gpu]" audioread runpod requests boto3
# Forza onnxruntime-gpu compatibile CUDA 12.x (1.19.2 supporta cu12).
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true \
    && pip install --no-cache-dir onnxruntime-gpu==1.19.2
# Reinstalla torch/torchaudio FISSI in caso audio-separator li abbia toccati.
RUN pip install --no-cache-dir --force-reinstall "torch==2.8.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128

# Pre-download dei modelli NELL'IMMAGINE (no download a runtime).
# STEP A (voce completa lead+cori per la trascrizione): UVR-MDX-NET-Inst_HQ_3.
# STEP B (base+cori per il karaoke): roformer karaoke gabox_v2 (aufr33 di scorta).
RUN audio-separator --download_model_only -m UVR-MDX-NET-Inst_HQ_3.onnx || true
RUN audio-separator --download_model_only -m mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt || true
RUN audio-separator --download_model_only -m mel_band_roformer_karaoke_gabox_v2.ckpt || true
# VOCE di allineamento = lead+CORI (niente buchi nei ritornelli corali): modello
# VOCALE standard BS-Roformer (Vocals = tutte le voci). Pre-scaricato qui.
RUN audio-separator --download_model_only -m model_bs_roformer_ep_317_sdr_12.9755.ckpt || true

# --- LAM: allineamento testo sul cantato (gira sullo STESSO worker gia' caldo:
# nessun cold start in piu'). espeak-ng NATIVO via apt (Ubuntu root) -> niente
# compile/glibc/TMPDIR-noexec come sul CloudLinux. ---
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng     && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers phonemizer sortedcontainers         librosa soundfile tqdm pandas scipy langdetect
COPY lam_local.py lam_align.py /app/
COPY lam/ /app/lam/

# --- DomAI: Whisper (faster-whisper) per TRASCRIVERE quando manca il testo.
# Gira sullo STESSO worker gia' caldo (GPU). Il modello e' pre-scaricato
# nell'immagine -> niente download a runtime (no cold-start extra). ---
RUN pip install --no-cache-dir faster-whisper
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')" || true

# Codice: pipeline snella + handler. NIENTE accordi.py.
COPY pipeline.py rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]

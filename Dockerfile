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
RUN pip install --no-cache-dir "audio-separator[gpu]" runpod requests boto3
# Forza onnxruntime-gpu compatibile CUDA 12.x (1.19.2 supporta cu12).
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true \
    && pip install --no-cache-dir onnxruntime-gpu==1.19.2
# Reinstalla torch/torchaudio FISSI in caso audio-separator li abbia toccati.
RUN pip install --no-cache-dir --force-reinstall "torch==2.8.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128

# Pre-download dei 2 modelli Roformer karaoke NELL'IMMAGINE (no download a runtime)
RUN audio-separator --download_model_only -m mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt || true
RUN audio-separator --download_model_only -m mel_band_roformer_karaoke_gabox_v2.ckpt || true

# Codice: pipeline snella + handler. NIENTE accordi.py.
COPY pipeline.py rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]

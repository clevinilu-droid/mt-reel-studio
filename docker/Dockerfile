FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    AUDIO_SEPARATOR_MODEL_DIR=/opt/audio-separator-models \
    UV_LINK_MODE=copy

# 1. System tools + FFmpeg 8.1 static (NVENC + libass + gblur) + rclone + OpenSSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget xz-utils unzip git libsndfile1 \
    openssh-server tmux htop nano \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && curl -fsSL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz" -o /tmp/ffmpeg.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffmpeg /usr/local/bin/ffmpeg \
    && cp /tmp/ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg* \
    && curl -fsSL "https://downloads.rclone.org/rclone-current-linux-amd64.zip" -o /tmp/rclone.zip \
    && unzip -q /tmp/rclone.zip -d /tmp \
    && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone \
    && chmod +x /usr/local/bin/rclone \
    && rm -rf /tmp/rclone*

# 2. Core AI audio tools on system PyTorch 2.10 + CUDA 12.8 (audio-separator + VieNeu-TTS)
RUN pip install --no-cache-dir -U pip uv \
    && pip install --no-cache-dir -U "audio-separator[gpu]" "vieneu[cuda]" audioread librosa soundfile scipy opencv-python-headless google-genai \
    && pip uninstall -y onnxruntime || true \
    && pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu

# 3. Pre-download Mel-Band-RoFormer (InstVoc Duality V2) so Step 3.1 starts in 0s
RUN mkdir -p /opt/audio-separator-models \
    && python3 -c "from audio_separator.separator import Separator; Separator(model_file_dir='/opt/audio-separator-models', info_only=True).download_model_files('melband_roformer_instvox_duality_v2.ckpt')"

# 4. Pre-install pyVideoTrans v4.19 in isolated uv Python 3.10 venv (for Step 5 dubbing)
RUN git clone --depth 1 https://github.com/jianchang512/pyvideotrans.git /opt/pyvideotrans \
    && cd /opt/pyvideotrans \
    && uv python install 3.10 \
    && uv sync --no-dev \
    && uv cache clean

WORKDIR /workspace
EXPOSE 22 7860
CMD ["bash", "-lc", "service ssh start || true; sleep infinity"]

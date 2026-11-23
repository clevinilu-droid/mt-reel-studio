# MT Reel Studio v2 — Vast.ai 1-Click Multi-Venv Docker Image
# Tách 3 venv hoàn toàn độc lập để FunASR / vLLM / VieNeu + Qwen3-TTS không bao giờ đụng thư viện.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/hf_cache \
    MT_VENV_ASR_SEP=/opt/venvs/asr_sep/bin/python \
    MT_VENV_VLM_LLM=/opt/venvs/vlm_llm/bin/python \
    MT_VENV_TTS_UI=/opt/venvs/tts_ui/bin/python

# 1. Gói hệ thống + FFmpeg + sox + rclone (bản mới) + cloudflared + uv
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      software-properties-common build-essential curl wget git unzip ca-certificates \
      ffmpeg espeak-ng sox libsndfile1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev && \
    curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && \
    unzip -qj /tmp/rclone.zip '*/rclone' -d /usr/local/bin && \
    chmod +x /usr/local/bin/rclone && rm -f /tmp/rclone.zip && \
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    rm -rf /var/lib/apt/lists/*

# 2. Venv 1: ASR + Keyframe + Tách giọng (FunASR + Faster-Whisper + PyAV + BS-RoFormer + pypinyin)
RUN uv venv /opt/venvs/asr_sep --python python3.11 && \
    uv pip install --python /opt/venvs/asr_sep/bin/python \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --python /opt/venvs/asr_sep/bin/python \
      funasr modelscope faster-whisper av "audio-separator[gpu]" \
      rapidocr-onnxruntime "onnxruntime-gpu==1.22.0" \
      pypinyin numpy soundfile scipy pillow

# 3. Venv 2: VLM & LLM (vLLM + bitsandbytes 4-bit cho Qwen3.8-27B & Gemma-4-26B-A4B)
RUN uv venv /opt/venvs/vlm_llm --python python3.11 && \
    uv pip install --python /opt/venvs/vlm_llm/bin/python \
      vllm bitsandbytes hf-transfer transformers accelerate \
      av pillow rapidocr-onnxruntime "onnxruntime-gpu==1.22.0"

# 4. Venv 3: TTS + Step1 Spotcheck + Web UI (VieNeu + Qwen3-TTS-1.7B + FastAPI)
RUN uv venv /opt/venvs/tts_ui --python python3.11 && \
    uv pip install --python /opt/venvs/tts_ui/bin/python \
      torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --python /opt/venvs/tts_ui/bin/python \
      vieneu qwen-tts faster-whisper rapidocr-onnxruntime "onnxruntime-gpu==1.22.0" \
      fastapi "uvicorn[standard]" python-multipart websockets pydantic python-dotenv \
      hf-transfer soundfile numpy

# 5. FFmpeg bản Jellyfin 7.1 (hỗ trợ NVENC API 12.0 cho mọi driver >= 520 + -fps_mode chuẩn)
# Đặt SAU 3 venv để giữ nguyên 100% cache build từ các tầng trên
RUN apt-get update -qq && \
    curl -fsSL https://github.com/jellyfin/jellyfin-ffmpeg/releases/download/v7.1.4-3/jellyfin-ffmpeg7_7.1.4-3-jammy_amd64.deb -o /tmp/jffmpeg.deb && \
    apt-get install -y -qq --no-install-recommends /tmp/jffmpeg.deb && \
    rm -f /tmp/jffmpeg.deb && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe

# 6. Venv 4: VoxCPM2 (giọng ngoại ngữ tự nhiên 48kHz, tách biệt dependency khỏi tts_ui)
RUN uv venv /opt/venvs/voxcpm --python python3.11 && \
    uv pip install --python /opt/venvs/voxcpm/bin/python \
      torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --python /opt/venvs/voxcpm/bin/python \
      voxcpm soundfile hf-transfer

ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/mt_video
COPY _engine /workspace/mt_video/_engine
COPY branding /workspace/mt_video/branding
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /workspace/mt_video/Download /workspace/mt_video/output /workspace/hf_cache

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]

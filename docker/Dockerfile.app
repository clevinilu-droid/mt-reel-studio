FROM ghcr.io/clevinilu-droid/mt-reel-studio:v2

ENV NVIDIA_DRIVER_CAPABILITIES=all

# Cài Jellyfin FFmpeg 7.1 (NVENC API 12.0, chạy được mọi NVIDIA driver >= 520 thay vì BtbN master đòi 610+)
RUN apt-get update -qq && \
    curl -fsSL https://github.com/jellyfin/jellyfin-ffmpeg/releases/download/v7.1.4-3/jellyfin-ffmpeg7_7.1.4-3-jammy_amd64.deb -o /tmp/jffmpeg.deb && \
    apt-get install -y -qq --no-install-recommends /tmp/jffmpeg.deb && \
    rm -f /tmp/jffmpeg.deb && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe

WORKDIR /workspace/mt_video
COPY _engine /workspace/mt_video/_engine
COPY branding /workspace/mt_video/branding
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

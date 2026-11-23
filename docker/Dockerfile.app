FROM ghcr.io/clevinilu-droid/mt-reel-studio:v2

ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/mt_video
COPY _engine /workspace/mt_video/_engine
COPY branding /workspace/mt_video/branding
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

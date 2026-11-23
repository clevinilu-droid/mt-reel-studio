#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.config/rclone /workspace/mt_video/Download /workspace/mt_video/output /workspace/hf_cache

# 0. Tự đồng bộ code mới nhất từ GitHub (~0.5s) để không bao giờ phải rebuild Docker khi sửa Python/UI
if [ "${MT_AUTO_SYNC:-1}" = "1" ]; then
  if curl -fsSL --max-time 8 https://github.com/clevinilu-droid/mt-reel-studio/archive/refs/heads/main.tar.gz \
      | tar -xz --strip-components=1 -C /workspace/mt_video mt-reel-studio-main/_engine mt-reel-studio-main/branding 2>/dev/null; then
    echo "[BOOT] Đã cập nhật code mới nhất từ GitHub."
  fi
fi

# 1. Tự khôi phục rclone.conf nếu được truyền từ Vast Template / UI
if [ -n "${RCLONE_CONF_B64:-}" ]; then
  echo "${RCLONE_CONF_B64}" | base64 -d > /root/.config/rclone/rclone.conf
  echo "[BOOT] Đã nạp cấu hình Google Drive (rclone.conf)."
fi

# 2. Thiết lập đường dẫn thư viện CUDA cho ONNX Runtime (quét mọi venv)
LD_EXTRA=$(python3 - <<'PY'
import glob
from pathlib import Path
dirs = []
for pat in ("/opt/venvs/*/lib/python*/site-packages/nvidia/*/lib", "/opt/venvs/*/lib/python*/site-packages/torch/lib"):
    for p in glob.glob(pat):
        if Path(p).is_dir():
            dirs.append(p)
print(":".join(dict.fromkeys(dirs)))
PY
)
if [ -n "$LD_EXTRA" ]; then
  export LD_LIBRARY_PATH="${LD_EXTRA}:${LD_LIBRARY_PATH:-}"
fi

# 3. Khởi chạy Cloudflare Quick Tunnel ở nền để có link HTTPS mở thẳng Web UI không cần SSH
rm -f /workspace/mt_video/_tunnel.log /workspace/mt_video/_tunnel_url.txt
if command -v cloudflared >/dev/null 2>&1; then
  nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860 > /workspace/mt_video/_tunnel.log 2>&1 &
  (
    for _ in $(seq 1 30); do
      URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /workspace/mt_video/_tunnel.log | head -n 1 || true)
      if [ -n "$URL" ]; then
        echo "$URL" > /workspace/mt_video/_tunnel_url.txt
        echo "========================================================"
        echo "[MT REEL UI TUNNEL] Mở trên trình duyệt: $URL"
        echo "========================================================"
        break
      fi
      sleep 1
    done
  ) &
fi

# 4. Bật Web UI MT Reel Studio v2 tại cổng 7860
exec /opt/venvs/tts_ui/bin/python /workspace/mt_video/_engine/run_ui.py --host 0.0.0.0 --port 7860

"""Điểm khởi chạy Web UI MT Reel Studio v2 (cổng mặc định 7860)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Web UI MT Reel Studio v2")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ bind (dùng 0.0.0.0 trên Vast.ai)")
    parser.add_argument("--port", type=int, default=7860, help="Cổng HTTP/WebSocket (mặc định 7860)")
    args = parser.parse_args()

    import uvicorn

    print(f"[MT REEL UI] Mở trình duyệt tại: http://{args.host}:{args.port}", flush=True)
    uvicorn.run("_engine.ui.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

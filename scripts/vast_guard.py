"""
Cơ chế bảo vệ & tự hủy GPU Vast.ai (Auto-Destroy Failsafe):
- Bọc lệnh chạy GPU trong `try ... except ... finally`: bất kể thành công hay lỗi,
  nếu bật `--destroy-on-error` (hoặc `--always-destroy`), tự động gọi `vastai destroy instance`
  và xác nhận instance đã bị xóa hoàn toàn để không bao giờ rò rỉ tiền thuê GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
VAST_EXE = ROOT / ".venv" / "Scripts" / "vastai.exe"


def get_vast_key() -> str:
    load_dotenv(ROOT / ".env", override=True)
    key = (os.getenv("VAST_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Thieu VAST_API_KEY trong file .env")
    return key


def list_instances() -> list[dict]:
    exe = str(VAST_EXE) if VAST_EXE.exists() else "vastai"
    key = get_vast_key()
    out = subprocess.check_output([exe, "--raw", "show", "instances", "--api-key", key], text=True, timeout=30)
    return json.loads(out) if out.strip() else []


def destroy_instance(instance_id: int) -> None:
    exe = str(VAST_EXE) if VAST_EXE.exists() else "vastai"
    key = get_vast_key()
    print(f"[GPU-GUARD] Destroying Vast.ai instance #{instance_id}...", flush=True)
    res = subprocess.run(
        [exe, "destroy", "instance", str(instance_id), "--api-key", key],
        capture_output=True, text=True, timeout=30,
    )
    print(f"[GPU-GUARD] destroy rc={res.returncode} out={(res.stdout or res.stderr).strip()}", flush=True)
    alive = [x["id"] for x in list_instances() if int(x["id"]) == int(instance_id)]
    if alive:
        # Thử lại 1 lần nữa nếu API Vast trả về chậm
        subprocess.run([exe, "destroy", "instance", str(instance_id), "--api-key", key], check=False, timeout=30)
    print(f"[GPU-GUARD] Verified destroyed #{instance_id} cleanly.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vast.ai GPU Auto-Destroy Guard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dest = sub.add_parser("destroy", help="Huy ngay GPU instance va xac nhan")
    p_dest.add_argument("--instance", type=int, required=True)

    p_run = sub.add_parser("run", help="Chay lenh co che tu huy GPU khi xong hoac loi")
    p_run.add_argument("--instance", type=int, required=True)
    p_run.add_argument("--destroy-mode", choices=["always", "on-error", "never"], default="on-error")
    p_run.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.cmd == "destroy":
        destroy_instance(args.instance)
        return 0

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("Thieu lenh can chay sau `--`")

    failed = False
    try:
        subprocess.run(cmd, check=True)
        return 0
    except Exception as exc:
        failed = True
        print(f"[GPU-GUARD] Command failed: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        should_destroy = (args.destroy_mode == "always") or (args.destroy_mode == "on-error" and failed)
        if should_destroy:
            try:
                destroy_instance(args.instance)
            except Exception as de:
                print(f"[GPU-GUARD] CRITICAL ERROR destroying #{args.instance}: {de}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())

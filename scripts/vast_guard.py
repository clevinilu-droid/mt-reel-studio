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
    for attempt in range(3):
        try:
            res = subprocess.run(
                [exe, "destroy", "instance", str(instance_id), "-y", "--api-key", key],
                capture_output=True, text=True, timeout=60,
            )
            print(f"[GPU-GUARD] destroy rc={res.returncode} out={(res.stdout or res.stderr).strip()}", flush=True)
            alive = [x["id"] for x in list_instances() if int(x["id"]) == int(instance_id)]
            if not alive:
                print(f"[GPU-GUARD] Verified destroyed #{instance_id} cleanly.", flush=True)
                return
        except Exception as e:
            print(f"[GPU-GUARD] Retry destroy #{instance_id} ({attempt+1}/3): {type(e).__name__}", flush=True)
    print(f"[GPU-GUARD] Finished destroy attempts for #{instance_id}.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vast.ai GPU Auto-Destroy Guard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dest = sub.add_parser("destroy", help="Huy ngay GPU instance va xac nhan")
    p_dest.add_argument("--instance", type=int, required=True)

    p_run = sub.add_parser("run", help="Chay lenh co che tu huy GPU khi xong hoac loi")
    p_run.add_argument("--instance", type=int, required=True)
    p_run.add_argument("--destroy-mode", choices=["always", "on-error", "never"], default="on-error")
    p_run.add_argument("command", nargs=argparse.REMAINDER)

    p_det = sub.add_parser("remote-watch", help="Chay lenh detached tren GPU (chong rot SSH) va tu huy GPU neu loi hoac xong")
    p_det.add_argument("--instance", type=int, required=True)
    p_det.add_argument("--host", required=True)
    p_det.add_argument("--port", type=int, required=True)
    p_det.add_argument("--alt-host", default="")
    p_det.add_argument("--alt-port", type=int, default=0)
    p_det.add_argument("--job-name", required=True)
    p_det.add_argument("--remote-cmd", default="")
    p_det.add_argument("--attach", action="store_true", help="Theo doi tiep job da chay truoc do")
    p_det.add_argument("--destroy-mode", choices=["always", "on-error", "never"], default="on-error")
    p_det.add_argument("--poll-sec", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "destroy":
        destroy_instance(args.instance)
        return 0

    if args.cmd == "remote-watch":
        import time
        endpoints = [(args.host, args.port)]
        if args.alt_host and args.alt_port:
            endpoints.append((args.alt_host, args.alt_port))

        def mk_ssh(ep: tuple[str, int]) -> list[str]:
            return [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                "-p", str(ep[1]), f"root@{ep[0]}",
            ]

        log_f = f"/workspace/{args.job_name}.log"
        done_f = f"/workspace/{args.job_name}.exit"
        pid_f = f"/workspace/{args.job_name}.pid"
        failed = False
        ep_idx = 0
        try:
            if not args.attach:
                if not args.remote_cmd:
                    raise ValueError("Can --remote-cmd khi khong dung --attach")
                launch_sh = (
                    f"rm -f {done_f}; "
                    f"nohup bash -lc '({args.remote_cmd}); echo $? > {done_f}' > {log_f} 2>&1 < /dev/null & "
                    f"echo $! > {pid_f}; echo LAUNCHED_PID=$(cat {pid_f})"
                )
                for _i in range(len(endpoints)):
                    try:
                        subprocess.run([*mk_ssh(endpoints[ep_idx]), launch_sh], check=True, timeout=30)
                        break
                    except Exception:
                        ep_idx = (ep_idx + 1) % len(endpoints)
                else:
                    raise RuntimeError("Khong ket noi duoc GPU de khoi chay remote-watch")

            net_errs = 0
            while True:
                time.sleep(args.poll_sec)
                ssh_base = mk_ssh(endpoints[ep_idx])
                try:
                    chk = subprocess.run(
                        [*ssh_base, f"if [ -f {done_f} ]; then echo EXIT=$(cat {done_f}); fi; tail -n 3 {log_f} 2>/dev/null || true"],
                        capture_output=True, text=True, timeout=30,
                    )
                except subprocess.TimeoutExpired:
                    chk = None
                if chk is None or chk.returncode != 0:
                    net_errs += 1
                    ep_idx = (ep_idx + 1) % len(endpoints)
                    print(f"[GPU-GUARD] Transient SSH error ({net_errs}/15) -> switch {endpoints[ep_idx]}", flush=True)
                    if net_errs >= 15:
                        raise RuntimeError("Mat ket noi SSH toi GPU qua 15 lan lien tiep")
                    continue
                net_errs = 0
                out = chk.stdout.strip()
                last_line = out.splitlines()[-1] if out else ""
                print(f"[GPU-WATCH:{args.job_name}] {last_line}", flush=True)
                for line in out.splitlines():
                    if line.startswith("EXIT="):
                        rc = int(line.split("=", 1)[1].strip())
                        if rc != 0:
                            tail = subprocess.run([*ssh_base, f"tail -n 35 {log_f}"], capture_output=True, text=True, timeout=20)
                            raise RuntimeError(f"Remote job {args.job_name} exited rc={rc}:\n{tail.stdout}")
                        print(f"[OK] Remote job {args.job_name} completed rc=0", flush=True)
                        return 0
        except Exception as exc:
            failed = True
            print(f"[GPU-GUARD] Remote watch failed: {exc}", file=sys.stderr, flush=True)
            raise
        finally:
            if (args.destroy_mode == "always") or (args.destroy_mode == "on-error" and failed):
                destroy_instance(args.instance)

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

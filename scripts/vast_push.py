"""Helper upload & download Vast.ai theo file với retry + dual-endpoint failover."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def run_retry(cmd: list[str], timeout: int = 300, tries: int = 8) -> subprocess.CompletedProcess:
    for a in range(tries):
        try:
            r = subprocess.run(cmd, timeout=timeout)
            if r.returncode == 0:
                return r
        except subprocess.TimeoutExpired:
            print(f"[XFER] Timeout ({timeout}s) attempt {a+1}/{tries}", flush=True)
        time.sleep(min(10, 2 * (a + 1)))
    raise RuntimeError(f"Transfer failed after {tries} attempts: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["push", "pull"], required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--alt-host", default="")
    ap.add_argument("--alt-port", type=int, default=0)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    endpoints = [(args.host, args.port)]
    if args.alt_host and args.alt_port:
        endpoints.append((args.alt_host, args.alt_port))

    for attempt in range(8):
        ep = endpoints[attempt % len(endpoints)]
        scp = [
            "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-P", str(ep[1]),
        ]
        try:
            if args.mode == "push":
                run_retry([*scp, args.src, f"root@{ep[0]}:{args.dst}"], timeout=args.timeout, tries=1)
            else:
                Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
                run_retry([*scp, f"root@{ep[0]}:{args.src}", args.dst], timeout=args.timeout, tries=1)
            print(f"[XFER-OK] {args.mode} {args.src} -> {args.dst}", flush=True)
            return 0
        except Exception as e:
            print(f"[XFER] retry {attempt+1}/8 via {ep}: {e}", flush=True)
            time.sleep(3)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

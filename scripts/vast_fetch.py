"""Helper tải file từ Vast.ai chống rớt mạng/timeout: thử lại tối đa 8 lần cho từng file/lô nhỏ."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def run_retry(cmd: list[str], timeout: int = 60, tries: int = 8) -> subprocess.CompletedProcess:
    for a in range(tries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r
            err = (r.stderr or r.stdout or "").strip()[:120]
        except subprocess.TimeoutExpired:
            err = f"TimeoutExpired({timeout}s)"
        print(f"[FETCH] retry {a+1}/{tries}: {err}", flush=True)
        time.sleep(min(10, 2 * (a + 1)))
    raise RuntimeError(f"Failed after {tries} tries: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--alt-host", default="")
    ap.add_argument("--alt-port", type=int, default=0)
    ap.add_argument("--p1-vocals-out", type=Path, required=True)
    ap.add_argument("--p2-step3-dir", type=Path, required=True)
    args = ap.parse_args()

    endpoints = [(args.host, args.port)]
    if args.alt_host and args.alt_port:
        endpoints.append((args.alt_host, args.alt_port))

    def mk_ssh(ep: tuple[str, int]) -> list[str]:
        return [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            "-p", str(ep[1]), f"root@{ep[0]}",
        ]

    def mk_scp(ep: tuple[str, int]) -> list[str]:
        return [
            "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            "-P", str(ep[1]),
        ]

    # 1. P1 vocals chunks (8 parts vpart_00..07)
    if not args.p1_vocals_out.exists() or args.p1_vocals_out.stat().st_size < 200_000_000:
        cdir = args.p1_vocals_out.parent / "_vchunks"
        cdir.mkdir(parents=True, exist_ok=True)
        for idx in range(8):
            fn = f"vpart_{idx:02d}"
            lp = cdir / fn
            if lp.exists() and lp.stat().st_size > 0:
                continue
            for ei in range(8):
                ep = endpoints[ei % len(endpoints)]
                try:
                    run_retry([*mk_scp(ep), f"root@{ep[0]}:/workspace/full/p1/vchunks/{fn}", str(cdir)], timeout=90, tries=1)
                    break
                except Exception:
                    if ei == 7:
                        raise
            print(f"[FETCH] P1 vocal chunk {idx+1}/8 OK", flush=True)
        with open(args.p1_vocals_out, "wb") as f:
            for idx in range(8):
                f.write((cdir / f"vpart_{idx:02d}").read_bytes())
        for p in cdir.glob("*"):
            p.unlink()
        cdir.rmdir()
    print(f"[OK] P1 vocals: {args.p1_vocals_out.stat().st_size / 1e6:.1f} MB", flush=True)

    # 2. P2 hardsub_events.json + sheets (batches of 10 with dual endpoint failover)
    p2g = args.p2_step3_dir / "hardsub_grids"
    p2g.mkdir(parents=True, exist_ok=True)
    ev_p = args.p2_step3_dir / "hardsub_events.json"
    if not ev_p.exists():
        run_retry([*mk_scp(endpoints[0]), f"root@{endpoints[0][0]}:/workspace/full/p2/step3/hardsub_events.json", str(args.p2_step3_dir)], timeout=45)
    ls_res = None
    for ep in endpoints:
        try:
            ls_res = run_retry([*mk_ssh(ep), "ls -1 /workspace/full/p2/step3/hardsub_grids"], timeout=45, tries=3)
            break
        except Exception:
            continue
    if ls_res is None:
        raise RuntimeError("Khong ket noi duoc SSH de liet ke sheets P2")
    all_sheets = sorted(x.strip() for x in ls_res.stdout.splitlines() if x.strip().endswith(".jpg"))
    missing = [fn for fn in all_sheets if not (p2g / fn).exists()]
    print(f"[FETCH] P2 sheets missing: {len(missing)}/{len(all_sheets)}", flush=True)
    for i in range(0, len(missing), 10):
        ch = missing[i : i + 10]
        for ei in range(8):
            ep = endpoints[ei % len(endpoints)]
            srcs = [f"root@{ep[0]}:/workspace/full/p2/step3/hardsub_grids/{fn}" for fn in ch]
            try:
                run_retry([*mk_scp(ep), *srcs, str(p2g)], timeout=75, tries=1)
                break
            except Exception as exc:
                print(f"[FETCH] switch endpoint ({ei+1}/8): {exc}", flush=True)
                if ei == 7:
                    raise
                time.sleep(3)
        print(f"[FETCH] P2 sheets {min(i + 10, len(missing))}/{len(missing)}", flush=True)
    print(f"[OK] P2 sheets local: {len(list(p2g.glob('*.jpg')))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

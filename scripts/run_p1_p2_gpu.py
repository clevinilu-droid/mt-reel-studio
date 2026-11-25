"""
Điều phối tự động toàn bộ Step 5 trên GPU Vast.ai cho cả Full P2 và Full P1:
- Truyền file lớn (>150MB) theo từng mảnh 128MB (resume được nếu rớt mạng, dual-endpoint failover).
- Chồng lịch thông minh: vừa đẩy xong P2 là GPU chạy P2 Step 5 ngay (~35p) trong lúc Laptop đẩy P1 lên;
  sau đó GPU chạy P1 Step 3.1 + Step 5 trong lúc Laptop kéo thành phẩm P2 về.
- Bọc toàn bộ trong `try ... finally: destroy_instance(instance_id)` để TỰ HỦY GPU cả khi xong lẫn khi lỗi!
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vast_guard import destroy_instance, list_instances

ROOT = Path(__file__).resolve().parent.parent
SERIES = ROOT / "projects" / "001_退役归来这个保镖不好惹_CuuDacNhiemLamVeSi"
CHUNK_BYTES = 128 * 1024 * 1024  # 128 MB


def mk_ssh(ep: tuple[str, int]) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-p", str(ep[1]), f"root@{ep[0]}",
    ]


def mk_scp(ep: tuple[str, int]) -> list[str]:
    return [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-P", str(ep[1]),
    ]


def ssh_run(endpoints: list[tuple[str, int]], remote_cmd: str, timeout: int = 60, tries: int = 8) -> str:
    for a in range(tries):
        ep = endpoints[a % len(endpoints)]
        try:
            r = subprocess.run([*mk_ssh(ep), remote_cmd], capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
            err = (r.stderr or r.stdout or "").strip()[:160]
        except subprocess.TimeoutExpired:
            err = f"TimeoutExpired({timeout}s)"
        print(f"[SSH] retry {a+1}/{tries} via {ep}: {err}", flush=True)
        time.sleep(min(10, 2 * (a + 1)))
    raise RuntimeError(f"SSH failed after {tries} tries: {remote_cmd[:120]}")


def push_file(endpoints: list[tuple[str, int]], local_p: Path, remote_p: str) -> None:
    sz = local_p.stat().st_size
    if sz <= CHUNK_BYTES:
        for a in range(8):
            ep = endpoints[a % len(endpoints)]
            try:
                r = subprocess.run([*mk_scp(ep), str(local_p), f"root@{ep[0]}:{remote_p}"], timeout=240)
                if r.returncode == 0:
                    print(f"[PUSH-OK] {local_p.name} ({sz/1e6:.1f} MB)", flush=True)
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2)
        raise RuntimeError(f"Failed push {local_p}")

    # File lớn > 128MB: cắt mảnh 128MB upload từng mảnh + ghép lại trên Vast.ai
    rdir = f"{remote_p}._parts"
    ssh_run(endpoints, f"mkdir -p {rdir}")
    existing = set(x.strip() for x in ssh_run(endpoints, f"ls -1 {rdir} 2>/dev/null || true").splitlines() if x.strip())
    n_parts = (sz + CHUNK_BYTES - 1) // CHUNK_BYTES
    print(f"[PUSH-CHUNK] {local_p.name} ({sz/1e6:.1f} MB) -> {n_parts} parts", flush=True)
    with tempfile.TemporaryDirectory() as td:
        with open(local_p, "rb") as f:
            for idx in range(n_parts):
                pname = f"part_{idx:03d}"
                data = f.read(CHUNK_BYTES)
                if pname in existing:
                    # Kiểm tra size trên remote tránh mảnh dở dang
                    rsz = ssh_run(endpoints, f"stat -c%s {rdir}/{pname} 2>/dev/null || echo 0").strip()
                    if rsz.isdigit() and int(rsz) == len(data):
                        continue
                lp = Path(td) / pname
                lp.write_bytes(data)
                for a in range(8):
                    ep = endpoints[a % len(endpoints)]
                    try:
                        r = subprocess.run([*mk_scp(ep), str(lp), f"root@{ep[0]}:{rdir}/{pname}"], timeout=240)
                        if r.returncode == 0:
                            break
                    except subprocess.TimeoutExpired:
                        pass
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed chunk {pname} of {local_p.name}")
                lp.unlink(missing_ok=True)
                print(f"[PUSH-CHUNK] {local_p.name}: {idx+1}/{n_parts}", flush=True)
    ssh_run(endpoints, f"cat {rdir}/part_* > {remote_p} && rm -rf {rdir}", timeout=180)
    print(f"[PUSH-OK] Assembled {remote_p}", flush=True)


def pull_file(endpoints: list[tuple[str, int]], remote_p: str, local_p: Path) -> None:
    local_p.parent.mkdir(parents=True, exist_ok=True)
    rsz_s = ssh_run(endpoints, f"stat -c%s {remote_p}").strip()
    rsz = int(rsz_s)
    if rsz <= CHUNK_BYTES:
        for a in range(8):
            ep = endpoints[a % len(endpoints)]
            try:
                r = subprocess.run([*mk_scp(ep), f"root@{ep[0]}:{remote_p}", str(local_p)], timeout=240)
                if r.returncode == 0 and local_p.exists() and local_p.stat().st_size == rsz:
                    print(f"[PULL-OK] {local_p.name} ({rsz/1e6:.1f} MB)", flush=True)
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2)
        raise RuntimeError(f"Failed pull {remote_p}")

    rdir = f"{remote_p}._dlparts"
    ssh_run(endpoints, f"mkdir -p {rdir} && split -b {CHUNK_BYTES} -d {remote_p} {rdir}/part_", timeout=180)
    parts = [x.strip() for x in ssh_run(endpoints, f"ls -1 {rdir}").splitlines() if x.strip()]
    cdir = local_p.parent / f"_{local_p.stem}_parts"
    cdir.mkdir(exist_ok=True)
    try:
        for idx, pname in enumerate(sorted(parts)):
            lp = cdir / pname
            er = int(ssh_run(endpoints, f"stat -c%s {rdir}/{pname}").strip())
            if lp.exists() and lp.stat().st_size == er:
                continue
            for a in range(8):
                ep = endpoints[a % len(endpoints)]
                try:
                    r = subprocess.run([*mk_scp(ep), f"root@{ep[0]}:{rdir}/{pname}", str(lp)], timeout=240)
                    if r.returncode == 0 and lp.exists() and lp.stat().st_size == er:
                        break
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(3)
            else:
                raise RuntimeError(f"Failed pulling {pname}")
            print(f"[PULL-CHUNK] {local_p.name}: {idx+1}/{len(parts)}", flush=True)
        with open(local_p, "wb") as out_f:
            for pname in sorted(parts):
                out_f.write((cdir / pname).read_bytes())
    finally:
        for p in cdir.glob("*"):
            p.unlink(missing_ok=True)
        cdir.rmdir()
        ssh_run(endpoints, f"rm -rf {rdir}")
    print(f"[PULL-OK] Assembled {local_p.name} ({local_p.stat().st_size/1e6:.1f} MB)", flush=True)


def launch_remote(endpoints: list[tuple[str, int]], job_name: str, remote_cmd: str) -> None:
    log_f = f"/workspace/{job_name}.log"
    done_f = f"/workspace/{job_name}.exit"
    cmd = f"rm -f {done_f}; nohup bash -lc '({remote_cmd}); echo $? > {done_f}' > {log_f} 2>&1 < /dev/null &"
    ssh_run(endpoints, cmd, timeout=30)
    print(f"[GPU-START] {job_name} launched in background", flush=True)


def wait_remote(endpoints: list[tuple[str, int]], job_name: str, poll_sec: int = 25) -> None:
    log_f = f"/workspace/{job_name}.log"
    done_f = f"/workspace/{job_name}.exit"
    while True:
        time.sleep(poll_sec)
        out = ssh_run(endpoints, f"if [ -f {done_f} ]; then echo EXIT=$(cat {done_f}); fi; tail -n 3 {log_f} 2>/dev/null || true", timeout=35, tries=12)
        lines = [x for x in out.splitlines() if x.strip()]
        if lines:
            print(f"[GPU:{job_name}] {lines[-1]}", flush=True)
        for line in lines:
            if line.startswith("EXIT="):
                rc = int(line.split("=", 1)[1].strip())
                if rc != 0:
                    tail = ssh_run(endpoints, f"tail -n 40 {log_f}", timeout=30)
                    raise RuntimeError(f"Remote job {job_name} failed rc={rc}:\n{tail}")
                print(f"[GPU-DONE] {job_name} finished rc=0", flush=True)
                return


def pull_part_outputs(endpoints: list[tuple[str, int]], part: str, slug: str = "Cuu_Dac_Nhiem_Lam_Ve_Si") -> None:
    out_local = SERIES / part / "04_buoc5_thanh_pham_youtube"
    r_out = f"/workspace/full/{part.lower()}/out"
    stem = f"{slug}_{part}"
    pull_file(endpoints, f"{r_out}/{stem}.mp4", out_local / f"{stem}.mp4")
    for lang in ("vi", "en", "ja", "ko"):
        pull_file(endpoints, f"{r_out}/{stem}.{lang}.m4a", out_local / f"{stem}.{lang}.m4a")
        pull_file(endpoints, f"{r_out}/subtitles/{stem}.{lang}.srt", out_local / "subtitles" / f"{stem}.{lang}.srt")
    pull_file(endpoints, f"{r_out}/subtitles/{stem}.vi.vtt", out_local / "subtitles" / f"{stem}.vi.vtt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--alt-host", required=True)
    ap.add_argument("--alt-port", type=int, required=True)
    args = ap.parse_args()

    endpoints = [(args.host, args.port), (args.alt_host, args.alt_port)]
    try:
        # 1. Verify GPU & setup dirs
        info = ssh_run(
            endpoints,
            "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader && "
            "ffmpeg -y -v error -f lavfi -i color=c=black:s=640x640:d=0.2 -c:v hevc_nvenc -f null - && "
            "echo HEVC_NVENC_OK && python3 -m pip install -q --no-cache-dir edge-tts && "
            "mkdir -p /workspace/scripts /workspace/full/p1/step3 /workspace/full/p1/out /workspace/full/p2/step3 /workspace/full/p2/out",
            timeout=120,
        )
        print("[SETUP-OK]", info.strip(), flush=True)

        # 2. Push scripts & small JSON configs
        push_file(endpoints, ROOT / "scripts" / "vast_step5_dub_and_render.py", "/workspace/scripts/vast_step5_dub_and_render.py")
        push_file(endpoints, ROOT / "scripts" / "vast_step3_separate.py", "/workspace/scripts/vast_step3_separate.py")
        for p in ("P1", "P2"):
            pl = p.lower()
            push_file(endpoints, SERIES / p / "01_buoc2_nhan_vat_va_khung_blur" / "blur_config.json", f"/workspace/full/{pl}/blur_config.json")
            push_file(endpoints, SERIES / p / "03_buoc4_ban_dich_da_ngon_ngu" / "role_lines.json", f"/workspace/full/{pl}/role_lines.json")

        # 3. Push P2 media & immediately start P2 Step 5 on GPU
        push_file(endpoints, SERIES / "P2" / "02_buoc3_hardsub_va_am_thanh" / "vocals_48k_mono.flac", "/workspace/full/p2/step3/vocals_48k_mono.flac")
        push_file(endpoints, SERIES / "P2" / "02_buoc3_hardsub_va_am_thanh" / "instrumental_48k.flac", "/workspace/full/p2/step3/instrumental_48k.flac")
        push_file(endpoints, SERIES / "P2" / "00_buoc1_chuan_hoa" / "source_norm.mp4", "/workspace/full/p2/source_norm.mp4")

        launch_remote(
            endpoints,
            "p2_step5",
            "python3 -u /workspace/scripts/vast_step5_dub_and_render.py "
            "/workspace/full/p2/source_norm.mp4 "
            "/workspace/full/p2/step3/vocals_48k_mono.flac "
            "/workspace/full/p2/step3/instrumental_48k.flac "
            "/workspace/full/p2/role_lines.json "
            "/workspace/full/p2/blur_config.json "
            "/workspace/full/p2/out --name Cuu_Dac_Nhiem_Lam_Ve_Si_P2.mp4 --langs vi,en,ja,ko",
        )

        # 4. While GPU runs P2 Step 5 (~35m), upload P1 source_norm.mp4 + vocals_48k_mono.flac
        push_file(endpoints, SERIES / "P1" / "02_buoc3_hardsub_va_am_thanh" / "vocals_48k_mono.flac", "/workspace/full/p1/step3/vocals_48k_mono.flac")
        push_file(endpoints, SERIES / "P1" / "00_buoc1_chuan_hoa" / "source_norm.mp4", "/workspace/full/p1/source_norm.mp4")

        # 5. Wait for P2 Step 5, then run P1 Step 3.1 + P1 Step 5 while downloading P2 outputs
        wait_remote(endpoints, "p2_step5")
        launch_remote(
            endpoints,
            "p1_step3_and_5",
            "python3 -u /workspace/scripts/vast_step3_separate.py /workspace/full/p1/source_norm.mp4 /workspace/full/p1/step3_tmp && "
            "mv /workspace/full/p1/step3_tmp/instrumental_48k.flac /workspace/full/p1/step3/instrumental_48k.flac && "
            "rm -rf /workspace/full/p1/step3_tmp && "
            "python3 -u /workspace/scripts/vast_step5_dub_and_render.py "
            "/workspace/full/p1/source_norm.mp4 "
            "/workspace/full/p1/step3/vocals_48k_mono.flac "
            "/workspace/full/p1/step3/instrumental_48k.flac "
            "/workspace/full/p1/role_lines.json "
            "/workspace/full/p1/blur_config.json "
            "/workspace/full/p1/out --name Cuu_Dac_Nhiem_Lam_Ve_Si_P1.mp4 --langs vi,en,ja,ko",
        )
        pull_part_outputs(endpoints, "P2")
        # Giải phóng file P2 lớn trên disk local? Không, cứ giữ gọn; đợi P1 xong rồi kéo P1
        wait_remote(endpoints, "p1_step3_and_5")
        pull_part_outputs(endpoints, "P1")
        print("[ALL-GPU-WORK-DONE]", flush=True)
        return 0
    finally:
        # Tự hủy GPU an toàn cả khi thành công lẫn khi gặp bất kỳ lỗi nào
        try:
            alive = [x["id"] for x in list_instances() if int(x["id"]) == int(args.instance)]
            if alive:
                destroy_instance(args.instance)
        except Exception as exc:
            print(f"[GPU-GUARD] Cleanup err: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

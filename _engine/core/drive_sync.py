"""Đồng bộ Google Drive (rclone) theo Cách B gọn nhẹ & giữ chất lượng cao nhất.

Cách B đóng gói đúng 3 thành phần tinh gọn thay vì đẩy file 01_cfr_raw.mp4 nặng 3.5GB:
  1. Video gốc nguyên bản (`source.mp4`, giữ nguyên 1 đời nén)
  2. Audio gốc nén lossless FLAC (`01_orig_48k_stereo.flac`, giảm ~50% dung lượng, giữ 100% bit)
  3. Hồ sơ kiểm tra & checkpoint (`step1_audit.json`, `manifest.json`)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

LogFn = Callable[[str], None]
ProgFn = Callable[[float, str], None]

RCLONE_PCT_RE = re.compile(r"(\d+)%")


def find_rclone_bin() -> str | None:
    """Tìm rclone trong PATH hoặc thư mục WinGet trên Windows."""
    found = shutil.which("rclone")
    if found:
        return found
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        pkg_root = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if pkg_root.exists():
            matches = sorted(pkg_root.glob("Rclone*/*/rclone.exe")) + sorted(pkg_root.rglob("rclone.exe"))
            for m in matches:
                if m.is_file():
                    return str(m)
    return None


def check_gdrive_remote() -> tuple[bool, str]:
    """Kiểm tra rclone và remote `gdrive:` đã sẵn sàng chưa."""
    rclone = find_rclone_bin()
    if not rclone:
        return False, "Chưa tìm thấy rclone trên máy"
    try:
        out = subprocess.check_output([rclone, "listremotes"], text=True, timeout=5)
        remotes = [x.strip() for x in out.splitlines() if x.strip()]
        if "gdrive:" in remotes:
            return True, rclone
        return False, f"Chưa cấu hình remote gdrive: (hiện có: {', '.join(remotes) or 'trống'})"
    except Exception as exc:
        return False, f"Lỗi kiểm tra rclone: {exc}"


def ensure_lossless_flac(wav_48k: Path, flac_out: Path, log_cb: LogFn | None = None) -> Path:
    """Nén 01_orig_48k_stereo.wav sang FLAC lossless 48kHz stereo để giảm ~50% dung lượng upload."""
    if flac_out.exists() and flac_out.stat().st_size > 1_000_000:
        if log_cb:
            log_cb(f"[FLAC] Dùng lại file FLAC có sẵn ({flac_out.stat().st_size / 1024 / 1024:.1f} MB)")
        return flac_out
    if log_cb:
        log_cb("[FLAC] Đang nén lossless 01_orig_48k_stereo.wav -> .flac (không mất 1 bit nào)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(wav_48k),
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        str(flac_out),
    ]
    subprocess.run(cmd, check=True)
    if log_cb:
        mb = flac_out.stat().st_size / 1024 / 1024
        log_cb(f"[FLAC] Xong -> {flac_out.name} ({mb:.1f} MB)")
    return flac_out


def push_method_b_to_drive(
    work_dir: Path,
    work_rel: str,
    source_mp4: Path,
    remote_root: str = "gdrive:MT-Video",
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
) -> dict:
    """Đóng gói gọn (source.mp4 + audio_48k.flac + audit/manifest) và đẩy lên Google Drive/MT-Video/<Series>/<Part>/."""
    ok, rclone_or_err = check_gdrive_remote()
    if not ok:
        raise RuntimeError(rclone_or_err)
    rclone = rclone_or_err

    if not source_mp4.exists():
        raise FileNotFoundError(f"Không tìm thấy video gốc: {source_mp4}")

    stage_dir = work_dir / "_drive_bundle_b"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Danh sách (file_local, tên_file_chuẩn_trên_Drive)
    ext = source_mp4.suffix.lower() or ".mp4"
    files_to_push: list[tuple[Path, str]] = [(source_mp4, f"source{ext}")]

    wav_48k = work_dir / "01_orig_48k_stereo.wav"
    flac_path = stage_dir / "audio_48k.flac"
    if wav_48k.exists() and wav_48k.stat().st_size > 100_000:
        if prog_cb:
            prog_cb(10.0, "Đang nén audio WAV -> audio_48k.flac...")
        ensure_lossless_flac(wav_48k, flac_path, log_cb=log_cb)
        files_to_push.append((flac_path, "audio_48k.flac"))

    for meta_name in ("step1_audit.json", "manifest.json"):
        mp = work_dir / meta_name
        if mp.exists():
            dst_meta = stage_dir / meta_name
            shutil.copy2(mp, dst_meta)
            files_to_push.append((dst_meta, meta_name))

    remote_dest = f"{remote_root.rstrip('/')}/{work_rel.strip('/')}"
    total_mb = sum(p.stat().st_size for p, _ in files_to_push) / (1024 * 1024)
    if log_cb:
        log_cb(f"[DRIVE] Upload ({len(files_to_push)} file, {total_mb:.1f} MB) -> {remote_dest}")

    for idx, (fpath, remote_name) in enumerate(files_to_push, start=1):
        base_pct = 20.0 + ((idx - 1) / len(files_to_push)) * 75.0
        span = 75.0 / len(files_to_push)
        if prog_cb:
            prog_cb(base_pct, f"Upload ({idx}/{len(files_to_push)}): {remote_name}")
        if log_cb:
            log_cb(f"[DRIVE] ({idx}/{len(files_to_push)}) {remote_name} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)...")

        cmd = [
            rclone,
            "copyto",
            str(fpath),
            f"{remote_dest}/{remote_name}",
            "--stats",
            "2s",
            "--stats-one-line",
            "-v",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.strip()
            if not clean:
                continue
            m = RCLONE_PCT_RE.search(clean)
            if m and prog_cb:
                sub_p = float(m.group(1))
                prog_cb(min(98.0, base_pct + (sub_p / 100.0) * span), f"Upload {remote_name}: {clean[:72]}")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"rclone copyto thất bại với {remote_name} (exit={rc})")

    if prog_cb:
        prog_cb(100.0, f"Hoàn tất {total_mb:.1f} MB -> {remote_dest}")
    if log_cb:
        log_cb(f"[DRIVE] HOÀN TẤT ({total_mb:.1f} MB) -> {remote_dest}")

    return {
        "remote": remote_dest,
        "total_mb": round(total_mb, 2),
        "files": [rname for _, rname in files_to_push],
    }


def pull_from_drive(
    work_rel: str,
    work_dir: Path,
    remote_root: str = "gdrive:MT-Video",
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
) -> Path:
    """Kéo gói Cách B từ `gdrive:MT-Video/<Series>/<Part>/` về máy Vast.ai và giải nén FLAC -> 2 WAV trong vài giây."""
    ok, rclone_or_err = check_gdrive_remote()
    if not ok:
        raise RuntimeError(rclone_or_err)
    rclone = rclone_or_err

    work_dir.mkdir(parents=True, exist_ok=True)
    remote_src = f"{remote_root.rstrip('/')}/{work_rel.strip('/')}"
    if prog_cb:
        prog_cb(5.0, f"Đang kết nối {remote_src}...")
    if log_cb:
        log_cb(f"[DRIVE PULL] Kéo dữ liệu từ {remote_src} -> {work_dir}")

    proc = subprocess.Popen(
        [rclone, "copy", remote_src, str(work_dir), "--transfers", "4", "--stats=1s", "--stats-one-line"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            m = RCLONE_PCT_RE.search(line)
            if m:
                pct = float(m.group(1))
                if prog_cb:
                    prog_cb(min(72.0, 5.0 + pct * 0.67), f"Đang tải từ Drive: {int(pct)}% — {line}")
            elif log_cb:
                log_cb(f"[DRIVE PULL] {line}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"rclone pull thất bại (mã {rc})")

    flac_in = work_dir / "audio_48k.flac"
    wav_48k = work_dir / "01_orig_48k_stereo.wav"
    wav_16k = work_dir / "01_orig_16k_ai.wav"
    if flac_in.exists() and not wav_48k.exists():
        if prog_cb:
            prog_cb(80.0, "Đang giải nén audio_48k.flac -> 2 bản WAV chuẩn...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(flac_in),
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                str(wav_48k),
                "-map",
                "0:a:0",
                "-af",
                "highpass=f=80,dynaudnorm",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_16k),
            ],
            check=True,
        )
    if prog_cb:
        prog_cb(100.0, f"Đã tải xong từ {remote_src}")
    return work_dir


def list_drive_projects(remote_root: str = "gdrive:MT-Video") -> list[dict]:
    """Liệt kê các gói phim (<Series>/<Part>) đã có trên Google Drive để Vast.ai kéo về 1 click."""
    ok, rclone_or_err = check_gdrive_remote()
    if not ok:
        raise RuntimeError(rclone_or_err)
    rclone = rclone_or_err

    proc = subprocess.run(
        [rclone, "lsf", remote_root, "-R", "--files-only", "--max-depth", "4"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Không đọc được {remote_root}")

    projects: dict[str, dict] = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip().lstrip("/")
        if not line or "/" not in line:
            continue
        parts = line.split("/")
        # Chấp nhận <Series>/<Part>/<file>
        if len(parts) < 3:
            continue
        work_rel = "/".join(parts[:2])
        fname = parts[-1]
        entry = projects.setdefault(work_rel, {"work_rel": work_rel, "files": []})
        if len(parts) == 3 and fname not in entry["files"]:
            entry["files"].append(fname)

    out = []
    for work_rel, info in sorted(projects.items()):
        files = set(info["files"])
        ready = ("source.mp4" in files or "audio_48k.flac" in files)
        if not ready:
            continue
        out.append({"work_rel": work_rel, "files": sorted(files), "has_manifest": "manifest.json" in files})
    return out


def push_output_to_drive(
    work_rel: str,
    output_dir: Path,
    remote_root: str = "gdrive:MT-Video",
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
) -> dict:
    """Đẩy thành phẩm render (`output/<Series>/<Part>/`) lên Google Drive để về máy local tải nhẹ."""
    ok, rclone_or_err = check_gdrive_remote()
    if not ok:
        raise RuntimeError(rclone_or_err)
    rclone = rclone_or_err
    if not output_dir.exists():
        raise FileNotFoundError(f"Chưa có thư mục output để đẩy Drive: {output_dir}")

    remote_dest = f"{remote_root.rstrip('/')}/{work_rel.strip('/')}/output"
    if prog_cb:
        prog_cb(10.0, f"Đang upload thành phẩm lên {remote_dest}...")
    if log_cb:
        log_cb(f"[DRIVE OUTPUT] Đẩy {output_dir} -> {remote_dest}")

    proc = subprocess.Popen(
        [rclone, "copy", str(output_dir), remote_dest, "--transfers", "4", "--stats=2s", "--stats-one-line"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            m = RCLONE_PCT_RE.search(line)
            if m and prog_cb:
                prog_cb(min(98.0, max(12.0, float(m.group(1)))), f"Upload output: {m.group(1)}%")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"rclone upload output lỗi (code={rc})")
    if prog_cb:
        prog_cb(100.0, f"Đã upload thành phẩm lên {remote_dest}")
    return {"remote": remote_dest}

    return work_dir / "source.mp4"

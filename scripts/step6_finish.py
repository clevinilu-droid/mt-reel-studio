"""
Bước 6: Upload toàn bộ thành phẩm từng Phần (P1, P2...) lên Google Drive đúng cấu trúc:
  gdrive:MT-Video/<TenPhim>/<Phan>/
    ├── <Phan>_vi.mp4
    ├── audio_tracks/ (audio_vi.m4a, audio_en.m4a, audio_ja.m4a, audio_ko.m4a)
    ├── subtitles/    (vi.srt, en.srt, ja.srt, ko.srt, vi.vtt)
    └── youtube_seo/  (thumb_1280x720.jpg, thumb_poster.jpg, metadata_4langs.json, youtube_*.txt)

Sau đó tự động hủy GPU Vast.ai (nếu còn) và tùy chọn đưa Windows vào trạng thái Sleep (Standby S0).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from vast_guard import destroy_instance, list_instances


def verify_part_outputs(part_dir: Path, part_name: str) -> None:
    req = [
        part_dir / f"{part_name}_vi.mp4",
        part_dir / "audio_tracks" / "audio_vi.m4a",
        part_dir / "audio_tracks" / "audio_en.m4a",
        part_dir / "audio_tracks" / "audio_ja.m4a",
        part_dir / "audio_tracks" / "audio_ko.m4a",
        part_dir / "subtitles" / "vi.srt",
        part_dir / "subtitles" / "en.srt",
        part_dir / "subtitles" / "ja.srt",
        part_dir / "subtitles" / "ko.srt",
        part_dir / "youtube_seo" / "thumb_1280x720.jpg",
        part_dir / "youtube_seo" / "metadata_4langs.json",
    ]
    missing = [str(p) for p in req if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Thieu file thanh pham {part_name}: {missing}")


def upload_part_to_gdrive(part_out_dir: Path, drive_dest: str) -> None:
    print(f"[DRIVE] Uploading {part_out_dir} -> {drive_dest} ...", flush=True)
    cmd = [
        "rclone", "copy", str(part_out_dir), drive_dest,
        "--transfers", "4",
        "--checkers", "4",
        "--retries", "5",
        "--low-level-retries", "10",
        "--drive-chunk-size", "64M",
        "--exclude", "*.tmp*",
        "--exclude", "*_preview*",
        "--exclude", "test_10m/**",
        "--exclude", "*_config.json",
        "-P",
    ]
    subprocess.run(cmd, check=True)
    print(f"[DRIVE] Verifying remote files in {drive_dest} ...", flush=True)
    subprocess.run(["rclone", "ls", drive_dest], check=True)


def windows_sleep(delay_sec: int = 5) -> None:
    print(f"[SLEEP] Windows will enter Standby in {delay_sec}s...", flush=True)
    ps_cmd = (
        f"Start-Sleep -Seconds {delay_sec}; "
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[System.Windows.Forms.Application]::SetSuspendState([System.Windows.Forms.PowerState]::Suspend, $false, $false)"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload P1/P2 Drive, auto-destroy Vast GPU, optional Windows Sleep")
    ap.add_argument("--series-dir", type=Path, required=True, help="projects/<TenPhim>")
    ap.add_argument("--parts", nargs="+", required=True, help="Vi du: P1 P2")
    ap.add_argument("--drive-root", default="gdrive:MT-Video", help="Goc Drive")
    ap.add_argument("--instance", type=int, default=0, help="Vast.ai instance ID can tu huy")
    ap.add_argument("--sleep", action="store_true", help="Dua Windows ve che do Sleep sau khi xong")
    ap.add_argument("--sleep-delay", type=int, default=6)
    args = ap.parse_args()

    series_name = args.series_dir.name
    try:
        for p in args.parts:
            p_out = args.series_dir / p / "04_buoc5_thanh_pham_youtube"
            verify_part_outputs(p_out, p)

        for p in args.parts:
            p_out = args.series_dir / p / "04_buoc5_thanh_pham_youtube"
            dest = f"{args.drive_root}/{series_name}/{p}"
            upload_part_to_gdrive(p_out, dest)
    finally:
        if args.instance:
            alive = [x["id"] for x in list_instances() if int(x["id"]) == int(args.instance)]
            if alive:
                destroy_instance(args.instance)

    if args.sleep:
        windows_sleep(args.sleep_delay)
    print("[OK] Step 6 completed cleanly!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

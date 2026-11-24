"""
Bước 1.2: Chuẩn hóa video gốc (thường ghép từ hàng chục/trăm tập có thể lệch 1–4px và trôi FPS)
về 1 file master duy nhất khóa cứng:
  - Độ phân giải: 1080x1920 (SAR 1:1) — giữ nguyên phần 1080x1920, cắt gọn đều mép nếu lệch vài pixel
  - Tốc độ khung hình: 30 fps chuẩn (CFR) — chống trôi timeline trên phim dài 2–3 tiếng
  - Âm thanh: 48kHz Stereo AAC + aresample=async=1:first_pts=0 — chống lệch tiếng ở mối ghép tập

File gốc trong Download/ được giữ nguyên 100%, chỉ đọc; file chuẩn xuất vào:
  projects/<Series>/<Part>/00_buoc1_chuan_hoa/source_norm.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def probe_summary(video: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels",
        "-of", "json", str(video),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def detect_best_encoder(preferred: str = "auto") -> tuple[str, list[str]]:
    if preferred != "auto":
        return preferred, _encoder_flags(preferred)

    # 1. Check NVIDIA NVENC
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=4)
        if r.returncode == 0 and _test_encoder("h264_nvenc", ["-cq", "23", "-preset", "p4"]):
            return "h264_nvenc", _encoder_flags("h264_nvenc")
    except Exception:
        pass

    # 2. Check Intel Quick Sync (QSV)
    if _test_encoder("h264_qsv", ["-global_quality", "26", "-preset", "faster"]):
        return "h264_qsv", _encoder_flags("h264_qsv")

    # 3. CPU fallback
    return "libx264", _encoder_flags("libx264")


def _encoder_flags(enc: str) -> list[str]:
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "23", "-preset", "p4", "-pix_fmt", "yuv420p"]
    if enc == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "26", "-preset", "p4", "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "26", "-preset", "faster"]
    if enc == "hevc_qsv":
        return ["-c:v", "hevc_qsv", "-global_quality", "25", "-preset", "faster"]
    return ["-c:v", "libx264", "-crf", "21", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def _test_encoder(enc: str, extra: list[str]) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=black:s=640x640:d=0.2",
        "-vf", "format=nv12",
        "-c:v", enc, *extra, "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=10)
    return res.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Normalize concatenated source video to rock-solid 1080x1920 30fps CFR + 48kHz audio.")
    ap.add_argument("input_video", type=Path, help="Original source .mp4 in Download/")
    ap.add_argument("out_dir", type=Path, help="Target 00_buoc1_chuan_hoa directory")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--encoder", default="auto", choices=["auto", "h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "libx264"])
    ap.add_argument("--force", action="store_true", help="Re-encode even if source_norm.mp4 already exists")
    args = ap.parse_args()

    if not args.input_video.is_file():
        raise FileNotFoundError(f"Missing input video: {args.input_video}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = args.out_dir / "source_norm.mp4"
    tmp_mp4 = args.out_dir / "source_norm.tmp.mp4"
    report_json = args.out_dir / "norm_report.json"

    if out_mp4.exists() and not args.force:
        print(f"[SKIP] Already exists: {out_mp4} ({out_mp4.stat().st_size / 1e6:.1f} MB). Use --force to overwrite.")
        return

    before_probe = probe_summary(args.input_video)
    enc_name, enc_flags = detect_best_encoder(args.encoder)
    print(f"[1.2] Input: {args.input_video.name} ({args.input_video.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"[1.2] Encoder: {enc_name} | Target: {args.width}x{args.height} @ {args.fps}fps CFR | Audio: 48kHz Stereo", flush=True)

    vf = (
        f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
        f"crop={args.width}:{args.height},setsar=1,fps={args.fps},format=nv12"
    )
    if enc_name in ("h264_nvenc", "hevc_nvenc", "libx264"):
        vf = (
            f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
            f"crop={args.width}:{args.height},setsar=1,fps={args.fps}"
        )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        "-i", str(args.input_video),
        "-vf", vf,
        *enc_flags,
        "-fps_mode", "cfr",
        "-af", "aresample=async=1:first_pts=0",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(tmp_mp4),
    ]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    if out_mp4.exists():
        out_mp4.unlink()
    tmp_mp4.rename(out_mp4)
    dt = time.time() - t0

    after_probe = probe_summary(out_mp4)
    report = {
        "source": str(args.input_video),
        "normalized": str(out_mp4),
        "encoder": enc_name,
        "target": {"width": args.width, "height": args.height, "fps": args.fps},
        "elapsed_seconds": round(dt, 1),
        "size_mb": round(out_mp4.stat().st_size / (1024 * 1024), 2),
        "before": before_probe,
        "after": after_probe,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Normalized in {dt/60:.1f}m -> {out_mp4} ({report['size_mb']} MB)", flush=True)


if __name__ == "__main__":
    main()

"""
Bước 1: Chuẩn hóa video CFR bằng GPU + Trích xuất 2 file WAV đồng bộ + Spot-Check 5 điểm.

Đầu ra trong `_engine/work/<movie_name>/`:
  - `01_cfr_raw.mp4`        : Video chuẩn 30fps CFR, đồng nhất 1080x1920, không audio
  - `01_orig_48k_stereo.wav`: Âm thanh gốc 48kHz stereo PCM (dùng tách giọng & giữ SFX)
  - `01_orig_16k_ai.wav`    : Âm thanh 16kHz mono lọc nhẹ cho AI VAD/ASR
  - `step1_audit.json`      : Thông số FFprobe, kết quả Spot-Check 5 điểm & thời gian chạy từng pha

Cách dùng:
  python _engine/scripts/step1_prepare_and_audit.py <input_mp4> <movie_name>
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def clean_cjk(text: str) -> str:
    """Giữ lại ký tự chữ Hán để so khớp nội dung nhanh."""
    return "".join(CJK_RE.findall(text or ""))


def parse_ratio(r: str) -> float:
    """Chuyển chuỗi tỷ lệ FFprobe như '30000/1001' sang float."""
    if not r or "/" not in r:
        return float(r or 0.0)
    a, b = r.split("/", 1)
    den = float(b)
    return float(a) / den if den else 0.0


def probe_media(src_path: Path) -> dict:
    """Kiểm tra codec, VFR, lệch start_time và thời lượng video gốc bằng ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(src_path),
    ]
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8")
    data = json.loads(raw)
    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})

    r_fps = parse_ratio(v_stream.get("r_frame_rate", "0/1"))
    avg_fps = parse_ratio(v_stream.get("avg_frame_rate", "0/1"))
    v_start = float(v_stream.get("start_time", 0.0) or 0.0)
    a_start = float(a_stream.get("start_time", 0.0) or 0.0)
    duration = float(fmt.get("duration") or v_stream.get("duration") or a_stream.get("duration") or 0.0)

    return {
        "video_codec": v_stream.get("codec_name", ""),
        "width": int(v_stream.get("width", 0) or 0),
        "height": int(v_stream.get("height", 0) or 0),
        "r_fps": round(r_fps, 4),
        "avg_fps": round(avg_fps, 4),
        "is_vfr": abs(r_fps - avg_fps) > 0.05,
        "video_start_s": round(v_start, 6),
        "audio_start_s": round(a_start, 6),
        "av_offset_ms": round((v_start - a_start) * 1000.0, 2),
        "audio_sample_rate": int(a_stream.get("sample_rate", 0) or 0),
        "duration_s": round(duration, 3),
    }


def detect_gpu_encoder() -> tuple[str, list[str], list[str]]:
    """Tự phát hiện GPU encoder khả dụng (NVENC CUDA > Intel QSV D3D11VA > CPU libx264)."""
    candidates = [
        (
            "h264_nvenc",
            ["-hwaccel", "cuda"],
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"],
        ),
        (
            "h264_qsv",
            ["-hwaccel", "d3d11va"],
            ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "20"],
        ),
    ]
    for name, hw_args, enc_args in candidates:
        test_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.3:size=320x240:rate=30",
            "-pix_fmt",
            "nv12",
            *enc_args,
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(test_cmd, capture_output=True)
        if res.returncode == 0:
            return name, hw_args, enc_args
    return "libx264", [], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]


def normalize_and_extract_wavs(
    src_mp4: Path,
    cfr_mp4: Path,
    wav_48k: Path,
    wav_16k: Path,
    hw_args: list[str],
    enc_args: list[str],
    target_w: int = 1080,
    target_h: int = 1920,
    target_fps: int = 30,
):
    """Chạy 1 lượt FFmpeg GPU: chuẩn hóa video CFR + xuất 2 bản WAV đồng bộ trục thời gian."""
    vf = (
        f"fps={target_fps},"
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        "format=nv12"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-stats",
        "-fflags",
        "+genpts",
        *hw_args,
        "-i",
        str(src_mp4),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-fps_mode",
        "cfr",
        *enc_args,
        "-an",
        str(cfr_mp4),
        "-map",
        "0:a:0",
        "-af",
        "aresample=async=1:first_pts=0",
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
        "aresample=async=1:first_pts=0,highpass=f=80,dynaudnorm",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_16k),
    ]
    subprocess.run(cmd, check=True)


def _read_wav_slice(wav_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    """Đọc chính xác lát audio float32 16kHz từ file WAV bằng seek O(1)."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_total = wf.getnframes()
        i0 = max(0, min(n_total, int(round(start_s * sr))))
        cnt = max(0, min(n_total - i0, int(round(dur_s * sr))))
        wf.setpos(i0)
        raw = wf.readframes(cnt)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _ocr_window(
    video_mp4: Path,
    ocr_engine,
    start_s: float,
    dur_s: float,
    w: int,
    h: int,
    fps: float = 2.0,
) -> list[dict]:
    """Quét nhanh khung phụ đề nửa dưới màn hình trong cửa sổ spot-check."""
    cy = int(h * 0.55)
    ch = h - cy
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur_s:.3f}",
        "-i",
        str(video_mp4),
        "-vf",
        f"fps={fps},format=bgr24,crop={w}:{ch}:0:{cy}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * ch * 3 * 4)
    frame_bytes = w * ch * 3
    samples = []
    idx = 0
    step_dt = 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((ch, w, 3))
            res, _ = ocr_engine(frame)
            lines = []
            if res:
                for item in res:
                    txt, conf = item[1], float(item[2])
                    if conf >= 0.55 and clean_cjk(txt):
                        lines.append(txt.strip())
            samples.append({"t": start_s + idx * step_dt, "text": " ".join(lines)})
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()

    segs: list[dict] = []
    for s in samples:
        t_clean = clean_cjk(s["text"])
        if not t_clean:
            continue
        if (
            segs
            and s["t"] - segs[-1]["end"] <= step_dt * 1.6
            and difflib.SequenceMatcher(None, clean_cjk(segs[-1]["text"]), t_clean).ratio() >= 0.75
        ):
            segs[-1]["end"] = s["t"] + step_dt
            if len(t_clean) > len(clean_cjk(segs[-1]["text"])):
                segs[-1]["text"] = s["text"]
        else:
            segs.append({"start": s["t"], "end": s["t"] + step_dt, "text": s["text"]})
    return segs


def run_spotcheck(
    cfr_mp4: Path,
    wav_16k: Path,
    duration_s: float,
    w: int,
    h: int,
    whisper_model_size: str = "small",
) -> dict:
    """Spot-check 5 lát cắt đều nhau trong phim: đo lệch timeline & đối chiếu nội dung ASR vs OCR."""
    from faster_whisper import WhisperModel
    from rapidocr_onnxruntime import RapidOCR

    t_load0 = time.perf_counter()
    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    whisper = WhisperModel(
        whisper_model_size,
        device="cuda" if use_cuda else "cpu",
        compute_type="float16" if use_cuda else "int8",
    )
    ocr = RapidOCR()
    load_s = time.perf_counter() - t_load0

    window_dur = min(30.0, max(10.0, duration_s * 0.05))
    ratios = [0.005, 0.25, 0.50, 0.75, 0.93] if duration_s >= 240 else [0.02, 0.50]
    windows = [round(max(0.0, min(duration_s - window_dur, duration_s * r)), 2) for r in ratios]

    all_offsets: list[float] = []
    total_asr = 0
    matched_asr = 0
    window_reports = []

    t_run0 = time.perf_counter()
    for ws in windows:
        audio_arr = _read_wav_slice(wav_16k, ws, window_dur)
        segs_iter, _ = whisper.transcribe(
            audio_arr,
            language="zh",
            vad_filter=True,
            beam_size=1,
        )
        asr_segs = [
            {"start": round(ws + s.start, 2), "end": round(ws + s.end, 2), "text": s.text.strip()}
            for s in segs_iter
            if clean_cjk(s.text)
        ]
        ocr_segs = _ocr_window(cfr_mp4, ocr, ws, window_dur, w, h, fps=2.0)

        win_matches = []
        for a in asr_segs:
            total_asr += 1
            best = None
            best_dist = 999.0
            for o in ocr_segs:
                # Chấp nhận ghép cặp nếu hai mốc giao nhau hoặc cách nhau <= 1.2s
                dist = max(0.0, max(a["start"] - o["end"], o["start"] - a["end"]))
                if dist <= 1.2 and dist < best_dist:
                    best = o
                    best_dist = dist
            if best is not None:
                matched_asr += 1
                off = round(best["start"] - a["start"], 3)
                all_offsets.append(off)
                win_matches.append(
                    {
                        "asr_t": [a["start"], a["end"]],
                        "ocr_t": [best["start"], best["end"]],
                        "offset_s": off,
                        "asr": a["text"],
                        "ocr": best["text"],
                    }
                )

        window_reports.append(
            {
                "window_start_s": ws,
                "asr_lines": len(asr_segs),
                "ocr_lines": len(ocr_segs),
                "matched_lines": len(win_matches),
                "samples": win_matches[:3],
            }
        )
    run_s = time.perf_counter() - t_run0

    med_offset = round(float(np.median(all_offsets)), 3) if all_offsets else None
    match_pct = round((matched_asr / total_asr) * 100.0, 1) if total_asr else 0.0
    return {
        "model_load_s": round(load_s, 2),
        "spotcheck_exec_s": round(run_s, 2),
        "windows_checked": len(windows),
        "total_asr_lines": total_asr,
        "matched_ocr_lines": matched_asr,
        "timeline_match_pct": match_pct,
        "median_ocr_minus_asr_offset_s": med_offset,
        "windows": window_reports,
    }


def infer_work_rel(src_mp4: Path, engine_dir: Path) -> Path:
    """Suy ra đường dẫn tương đối `<Series>/<Part>` nếu nằm trong `Download/`, ngược lại lấy tên file."""
    download_dir = (engine_dir.parent / "Download").resolve()
    try:
        rel_parts = src_mp4.resolve().relative_to(download_dir).parts
        if len(rel_parts) >= 3:
            return Path(rel_parts[0]) / rel_parts[1]
    except ValueError:
        pass
    return Path(src_mp4.stem)


def main():
    parser = argparse.ArgumentParser(description="Bước 1: Chuẩn hóa CFR GPU + Xuất WAV đồng bộ + Spot-Check 5 điểm.")
    parser.add_argument("input_mp4", type=Path, help="Đường dẫn file video gốc")
    parser.add_argument("movie_name", nargs="?", default=None, type=str, help="Đường dẫn work (mặc định tự lấy <Series>/<Part> theo Download/)")
    parser.add_argument("--skip-spotcheck", action="store_true", help="Bỏ qua bước Spot-Check")
    args = parser.parse_args()

    src_mp4 = args.input_mp4.resolve()
    if not src_mp4.exists():
        raise FileNotFoundError(f"Không tìm thấy video gốc: {src_mp4}")

    engine_dir = Path(__file__).resolve().parent.parent
    work_rel = Path(args.movie_name) if args.movie_name else infer_work_rel(src_mp4, engine_dir)
    work_dir = engine_dir / "work" / work_rel
    work_dir.mkdir(parents=True, exist_ok=True)

    cfr_mp4 = work_dir / "01_cfr_raw.mp4"
    wav_48k = work_dir / "01_orig_48k_stereo.wav"
    wav_16k = work_dir / "01_orig_16k_ai.wav"
    audit_json = work_dir / "step1_audit.json"

    t_total0 = time.perf_counter()

    # 1A. FFprobe
    t0 = time.perf_counter()
    src_info = probe_media(src_mp4)
    enc_name, hw_args, enc_args = detect_gpu_encoder()
    t_probe = round(time.perf_counter() - t0, 2)
    print(
        f"[STEP 1A] Probe: {src_info['duration_s']}s | {src_info['width']}x{src_info['height']} | "
        f"VFR={src_info['is_vfr']} (r={src_info['r_fps']}, avg={src_info['avg_fps']}) | "
        f"AV_offset={src_info['av_offset_ms']}ms | Encoder={enc_name} ({t_probe}s)",
        flush=True,
    )

    # 1B. Chuẩn hóa CFR bằng GPU + xuất 2 file WAV đồng bộ
    t1 = time.perf_counter()
    normalize_and_extract_wavs(
        src_mp4,
        cfr_mp4,
        wav_48k,
        wav_16k,
        hw_args,
        enc_args,
        target_w=1080,
        target_h=1920,
        target_fps=30,
    )
    out_info = probe_media(cfr_mp4)
    t_gpu_norm = round(time.perf_counter() - t1, 2)
    print(
        f"[STEP 1B] GPU Normalize + Dual WAV xong trong {t_gpu_norm}s -> "
        f"{out_info['width']}x{out_info['height']} @ {out_info['avg_fps']}fps (VFR={out_info['is_vfr']}, dur={out_info['duration_s']}s)",
        flush=True,
    )

    # 1C. Spot-Check 5 điểm
    spot = {}
    if not args.skip_spotcheck:
        spot = run_spotcheck(cfr_mp4, wav_16k, out_info["duration_s"], out_info["width"], out_info["height"])
        print(
            f"[STEP 1C] Spot-Check {spot['windows_checked']} điểm: load={spot['model_load_s']}s, "
            f"scan={spot['spotcheck_exec_s']}s | Khớp timeline={spot['timeline_match_pct']}% "
            f"({spot['matched_ocr_lines']}/{spot['total_asr_lines']}) | Offset trung vị={spot['median_ocr_minus_asr_offset_s']}s",
            flush=True,
        )

    t_total = round(time.perf_counter() - t_total0, 2)
    report = {
        "source": str(src_mp4),
        "encoder_used": enc_name,
        "source_probe": src_info,
        "normalized_probe": out_info,
        "timings_s": {
            "probe_s": t_probe,
            "gpu_normalize_and_wav_s": t_gpu_norm,
            "spotcheck_model_load_s": spot.get("model_load_s", 0.0),
            "spotcheck_exec_s": spot.get("spotcheck_exec_s", 0.0),
            "total_step1_s": t_total,
        },
        "spotcheck": spot,
    }
    audit_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE STEP 1] Tổng thời gian: {t_total}s -> {audit_json}", flush=True)


if __name__ == "__main__":
    main()

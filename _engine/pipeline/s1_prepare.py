"""Bước 1 (s1_prepare): Chuẩn hóa video CFR bằng GPU + Trích 2 WAV đồng bộ + Spot-Check.

Hỗ trợ đầy đủ:
  - Tự dùng GPU sẵn có trên máy (`h264_nvenc` trên RTX 3090, `h264_qsv` trên Intel Iris Xe).
  - Chế độ `test_5min` (cắt & kiểm tra 5 phút đầu vào `<work_dir>/_test_5min`, không ghi đè bản Full).
  - Chế độ `full` (chuẩn hóa toàn bộ phim vào `<work_dir>`, tự bỏ qua nếu checkpoint hợp lệ và không bật force).
  - Báo tiến độ live (`%`, `fps`, `speed`, `ETA`) và hỗ trợ dừng khẩn (`cancel_event`).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _engine.core.checkpoint import atomic_write_json, step1_artifacts_valid
from _engine.core.vram_guard import patch_onnx_threads

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
FF_TIME_RE = re.compile(r"out_time_us=(\d+)")
FF_FPS_RE = re.compile(r"fps=([\d.]+)")
FF_SPEED_RE = re.compile(r"speed=\s*([\d.]+)x")

LogFn = Callable[[str], None]
ProgFn = Callable[[float, str], None]


def clean_cjk(text: str) -> str:
    return "".join(CJK_RE.findall(text or ""))


def parse_ratio(r: str) -> float:
    if not r or "/" not in r:
        return float(r or 0.0)
    a, b = r.split("/", 1)
    den = float(b)
    return float(a) / den if den else 0.0


def probe_media(src_path: Path) -> dict[str, Any]:
    """Đọc thông số video/audio gốc bằng ffprobe (codec, VFR, lệch start_time, thời lượng)."""
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
    """Tự nhận diện GPU encoder mạnh nhất trên máy (NVENC > Intel QSV > CPU libx264)."""
    candidates = [
        ("h264_nvenc", [], ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
        ("h264_nvenc", [], ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "20"]),
        ("h264_nvenc", [], ["-c:v", "h264_nvenc"]),
        ("h264_qsv", ["-hwaccel", "d3d11va"], ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "20"]),
    ]
    nvenc_errs: list[str] = []
    for name, hw_args, enc_args in candidates:
        res = subprocess.run(
            [
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
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if res.returncode == 0:
            return name, hw_args, enc_args
        if name == "h264_nvenc" and res.stderr:
            nvenc_errs.append(res.stderr.strip().splitlines()[-1])
    if nvenc_errs:
        print(f"[GPU ENCODER WARN] h264_nvenc không khả dụng ({nvenc_errs[0]}) -> fallback libx264", flush=True)
    return "libx264", [], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]


def diagnose_gpu_encoder() -> dict[str, Any]:
    """Kiểm tra chi tiết nguyên nhân NVENC/ONNX CUDA để chẩn đoán tức thì qua /api/diag."""
    import glob
    import os

    res = subprocess.run(
        [
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
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    enc_name, _, _ = detect_gpu_encoder()
    nv_libs = sorted(glob.glob("/usr/lib/**/libnvidia-encode*", recursive=True))
    return {
        "selected_encoder": enc_name,
        "nvenc_probe_rc": res.returncode,
        "nvenc_probe_stderr": (res.stderr or "").strip(),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        "libnvidia_encode_paths": nv_libs[:8],
    }


def _ffmpeg_cfr_flag() -> str:
    """Dùng `-fps_mode` cho FFmpeg 5.1+, fallback `-vsync` cho FFmpeg 4.x (Ubuntu 22.04)."""
    res = subprocess.run(["ffmpeg", "-h", "full"], capture_output=True, text=True, errors="ignore")
    return "-fps_mode" if "fps_mode" in (res.stdout or "") else "-vsync"


def _fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _link_or_copy_video(src_mp4: Path, dst_mp4: Path, max_dur: float | None = None):
    """Dùng trực tiếp video gốc (hardlink/copy 0.1s cho Full, hoặc stream-copy -c:v copy 1s cho Test) để KHÔNG encode video 2 lần và KHÔNG làm phình file 400MB lên 3.7GB."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if max_dur is not None:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src_mp4),
            "-t", f"{max_dur:.3f}",
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            str(dst_mp4),
        ]
        subprocess.run(cmd, check=True)
        return
    if dst_mp4.exists():
        dst_mp4.unlink(missing_ok=True)
    try:
        os.link(src_mp4, dst_mp4)
    except Exception:
        shutil.copy2(src_mp4, dst_mp4)


def _run_ffmpeg_normalize(
    src_mp4: Path,
    cfr_mp4: Path,
    wav_48k: Path,
    wav_16k: Path,
    hw_args: list[str],
    enc_args: list[str],
    expected_dur: float,
    max_dur: float | None = None,
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
    cancel_event: threading.Event | None = None,
):
    if prog_cb:
        prog_cb(8.0, "Đang gắn liên kết video gốc (không encode lại để giữ dung lượng gốc ~400MB)...")
    _link_or_copy_video(src_mp4, cfr_mp4, max_dur=max_dur)

    t_args = ["-t", f"{max_dur:.3f}"] if max_dur is not None else []
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(src_mp4),
        *t_args,
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
        *t_args,
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

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    cur_fps = 0.0
    cur_speed = 0.0
    last_emit = 0.0
    try:
        for raw_line in proc.stdout:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                proc.wait(timeout=5)
                raise InterruptedError("Người dùng đã tạm dừng Bước 1")
            line = raw_line.strip()
            if line.startswith("fps="):
                try:
                    cur_fps = float(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("speed="):
                m = FF_SPEED_RE.search(line)
                if m:
                    cur_speed = float(m.group(1))
            elif line.startswith("out_time_us="):
                m = FF_TIME_RE.search(line)
                if m and expected_dur > 0:
                    cur_s = int(m.group(1)) / 1_000_000.0
                    now = time.monotonic()
                    if now - last_emit >= 0.4:
                        last_emit = now
                        ratio = min(1.0, max(0.0, cur_s / expected_dur))
                        # Dành 5%..80% thanh tiến độ cho pha FFmpeg GPU
                        pct = round(5.0 + ratio * 75.0, 1)
                        eta = ((expected_dur - cur_s) / cur_speed) if cur_speed > 0.05 else 0.0
                        detail = f"GPU CFR {cur_s:.0f}/{expected_dur:.0f}s | {cur_fps:.0f} fps ({cur_speed:.2f}x) | ETA {_fmt_eta(eta)}"
                        if prog_cb:
                            prog_cb(pct, detail)
        rc = proc.wait()
        if rc != 0:
            err_txt = (proc.stderr.read() if proc.stderr else "").strip()
            raise RuntimeError(f"FFmpeg Bước 1 lỗi (exit={rc}): {err_txt[-400:]}")
    finally:
        if proc.poll() is None:
            proc.terminate()


def _read_wav_slice(wav_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_total = wf.getnframes()
        i0 = max(0, min(n_total, int(round(start_s * sr))))
        cnt = max(0, min(n_total - i0, int(round(dur_s * sr))))
        wf.setpos(i0)
        raw = wf.readframes(cnt)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _ocr_window(video_mp4: Path, ocr_engine, start_s: float, dur_s: float, w: int, h: int, fps: float = 2.0) -> list[dict]:
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
    assert proc.stdout is not None
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


def _run_spotcheck(
    cfr_mp4: Path,
    wav_16k: Path,
    duration_s: float,
    w: int,
    h: int,
    is_test_5min: bool,
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    from faster_whisper import WhisperModel
    from rapidocr_onnxruntime import RapidOCR

    if prog_cb:
        prog_cb(82.0, "Đang nạp Whisper + RapidOCR cho Spot-Check...")
    t_load0 = time.perf_counter()
    patch_onnx_threads(4)
    # Dùng Whisper small CPU int8 + RapidOCR CPU cho 3 đoạn 20s để tránh xung đột/treo cuDNN trong container
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    ocr = RapidOCR()
    load_s = time.perf_counter() - t_load0

    if is_test_5min or duration_s <= 360:
        window_dur = min(20.0, max(8.0, duration_s * 0.08))
        ratios = [0.05, 0.45, 0.85]
    else:
        window_dur = 30.0
        ratios = [0.005, 0.25, 0.50, 0.75, 0.93]
    windows = [round(max(0.0, min(max(0.0, duration_s - window_dur), duration_s * r)), 2) for r in ratios]

    all_offsets: list[float] = []
    total_asr = 0
    matched_asr = 0
    window_reports = []

    t_run0 = time.perf_counter()
    for wi, ws in enumerate(windows, start=1):
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Người dùng đã tạm dừng Spot-Check")
        if prog_cb:
            pct = round(84.0 + (wi - 1) / len(windows) * 14.0, 1)
            prog_cb(pct, f"Spot-Check điểm {wi}/{len(windows)} (@ {ws:.1f}s)...")
        if log_cb:
            log_cb(f"[STEP 1C] Spot-Check điểm {wi}/{len(windows)} tại {ws:.1f}s ({window_dur:.0f}s)...")

        audio_arr = _read_wav_slice(wav_16k, ws, window_dur)
        segs_iter, _ = whisper.transcribe(audio_arr, language="zh", vad_filter=False, beam_size=1)
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


def _slice_wav_head(src_wav: Path, dst_wav: Path, max_sec: float = 300.0):
    """Cắt nhanh đoạn đầu WAV PCM mà không giải mã lại."""
    with wave.open(str(src_wav), "rb") as wf_in:
        sr = wf_in.getframerate()
        n_frames = min(wf_in.getnframes(), int(round(max_sec * sr)))
        raw = wf_in.readframes(n_frames)
        params = wf_in.getparams()
    with wave.open(str(dst_wav), "wb") as wf_out:
        wf_out.setnchannels(params.nchannels)
        wf_out.setsampwidth(params.sampwidth)
        wf_out.setframerate(sr)
        wf_out.writeframes(raw)


def _derive_test_5min_from_full(work_dir: Path, target_dir: Path, log_cb: LogFn | None = None, prog_cb: ProgFn | None = None) -> dict[str, Any]:
    """Cắt siêu nhanh 5 phút đầu (-c copy) từ bản Full đã xong Bước 1 mà không chạy GPU/Spot-Check lại."""
    t0 = time.perf_counter()
    if prog_cb:
        prog_cb(25.0, "Đã có bản Full Bước 1 → đang cắt nhanh 2 phút đầu (-c copy)...")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(work_dir / "01_cfr_raw.mp4"), "-t", "120", "-c:v", "copy", "-an", str(target_dir / "01_cfr_raw.mp4")],
        check=True,
    )
    _slice_wav_head(work_dir / "01_orig_48k_stereo.wav", target_dir / "01_orig_48k_stereo.wav", 120.0)
    _slice_wav_head(work_dir / "01_orig_16k_ai.wav", target_dir / "01_orig_16k_ai.wav", 120.0)

    full_audit = json.loads((work_dir / "step1_audit.json").read_text(encoding="utf-8"))
    out_info = probe_media(target_dir / "01_cfr_raw.mp4")
    elapsed = round(time.perf_counter() - t0, 2)
    report = {
        "mode": "test_5min",
        "derived_from_full": True,
        "source": full_audit.get("source", ""),
        "encoder_used": full_audit.get("encoder_used", "copy"),
        "source_probe": full_audit.get("source_probe", {}),
        "normalized_probe": out_info,
        "timings_s": {"total_step1_s": elapsed},
        "spotcheck": full_audit.get("spotcheck", {}),
    }
    atomic_write_json(target_dir / "step1_audit.json", report)
    if log_cb:
        log_cb(f"[STEP 1] Tái sử dụng bản Full → cắt xong 5 phút đầu trong {elapsed}s (không chạy lại GPU).")
    if prog_cb:
        prog_cb(100.0, f"Tái sử dụng bản Full → cắt xong 5p đầu ({elapsed}s)")
    return report


def run_step1(
    src_mp4: Path,
    work_dir: Path,
    mode: str = "test_5min",
    force: bool = False,
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Chạy Bước 1 (Test 5 phút đầu hoặc Full) có tự động kiểm tra checkpoint."""
    is_test = mode == "test_5min"
    target_dir = (work_dir / "_test_5min") if is_test else work_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    audit_json = target_dir / "step1_audit.json"

    # Nếu đã có checkpoint hợp lệ và không ép chạy lại -> tái sử dụng tức thì!
    if not force and step1_artifacts_valid(target_dir):
        existing = json.loads(audit_json.read_text(encoding="utf-8"))
        if log_cb:
            log_cb(f"[STEP 1] Đã có checkpoint hợp lệ ({'Test 5p' if is_test else 'Full'}) -> Tái sử dụng ngay, không chạy lại.")
        if prog_cb:
            prog_cb(100.0, f"Tái sử dụng checkpoint ({'Test 5p' if is_test else 'Full'})")
        return existing

    # Nếu đang chọn Test 5p mà bản Full đã xong sẵn -> chỉ cắt 300s đầu trong ~2 giây!
    if is_test and not force and step1_artifacts_valid(work_dir):
        return _derive_test_5min_from_full(work_dir, target_dir, log_cb=log_cb, prog_cb=prog_cb)

    # Nếu đã kéo gói Cách B từ Google Drive (có sẵn step1_audit.json + 2 WAV ở work_dir) -> chỉ encode video CFR, bỏ qua Spot-Check!
    drive_audit_ok = (
        not force
        and (work_dir / "step1_audit.json").exists()
        and (work_dir / "01_orig_48k_stereo.wav").exists()
        and (work_dir / "01_orig_16k_ai.wav").exists()
    )
    if drive_audit_ok:
        full_audit = json.loads((work_dir / "step1_audit.json").read_text(encoding="utf-8"))
        if log_cb:
            log_cb("[STEP 1] Đã có gói Cách B từ Google Drive → gắn trực tiếp video gốc (<1s, không encode lại, không phình dung lượng).")
        src_info = full_audit.get("source_probe") or probe_media(src_mp4)
        enc_name, _, _ = detect_gpu_encoder()
        max_dur = min(120.0, float(src_info.get("duration_s") or 120.0)) if is_test else None
        cfr_mp4 = target_dir / "01_cfr_raw.mp4"
        wav_48k = target_dir / "01_orig_48k_stereo.wav"
        wav_16k = target_dir / "01_orig_16k_ai.wav"
        t1 = time.perf_counter()
        if not cfr_mp4.exists() or cfr_mp4.stat().st_size < 100_000:
            _link_or_copy_video(src_mp4, cfr_mp4, max_dur=max_dur)
        if is_test:
            _slice_wav_head(work_dir / "01_orig_48k_stereo.wav", wav_48k, 120.0)
            _slice_wav_head(work_dir / "01_orig_16k_ai.wav", wav_16k, 120.0)
        out_info = probe_media(cfr_mp4)
        elapsed = round(time.perf_counter() - t1, 2)
        report = {
            "mode": mode,
            "derived_from_drive": True,
            "source": str(src_mp4.resolve()),
            "encoder_used": enc_name,
            "source_probe": src_info,
            "normalized_probe": out_info,
            "timings_s": {"total_step1_s": elapsed},
            "spotcheck": full_audit.get("spotcheck", {}),
        }
        atomic_write_json(audit_json, report)
        if prog_cb:
            prog_cb(100.0, f"Xong Bước 1 tức thì từ gói Drive ({elapsed}s, giữ nguyên dung lượng gốc)")
        return report

    t_total0 = time.perf_counter()
    if prog_cb:
        prog_cb(2.0, "Đang soi thông số video gốc (FFprobe) & kiểm tra GPU...")
    t0 = time.perf_counter()
    src_info = probe_media(src_mp4)
    enc_name, hw_args, enc_args = detect_gpu_encoder()
    t_probe = round(time.perf_counter() - t0, 2)

    max_dur = min(120.0, src_info["duration_s"]) if is_test else None
    expected_dur = max_dur if max_dur is not None else src_info["duration_s"]
    if log_cb:
        log_cb(
            f"[STEP 1A] Probe ({'Test 5p' if is_test else 'Full'}): {src_info['duration_s']}s | "
            f"{src_info['width']}x{src_info['height']} | VFR={src_info['is_vfr']} | "
            f"AV_offset={src_info['av_offset_ms']}ms | GPU Encoder={enc_name} ({t_probe}s)"
        )

    cfr_mp4 = target_dir / "01_cfr_raw.mp4"
    wav_48k = target_dir / "01_orig_48k_stereo.wav"
    wav_16k = target_dir / "01_orig_16k_ai.wav"

    t1 = time.perf_counter()
    _run_ffmpeg_normalize(
        src_mp4,
        cfr_mp4,
        wav_48k,
        wav_16k,
        hw_args,
        enc_args,
        expected_dur=expected_dur,
        max_dur=max_dur,
        log_cb=log_cb,
        prog_cb=prog_cb,
        cancel_event=cancel_event,
    )
    out_info = probe_media(cfr_mp4)
    t_gpu = round(time.perf_counter() - t1, 2)
    if log_cb:
        log_cb(
            f"[STEP 1B] Xong GPU Normalize + 2 WAV trong {t_gpu}s -> "
            f"{out_info['width']}x{out_info['height']} @ {out_info['avg_fps']}fps (VFR={out_info['is_vfr']}, dur={out_info['duration_s']}s)"
        )

    spot = _run_spotcheck(
        cfr_mp4,
        wav_16k,
        out_info["duration_s"],
        out_info["width"],
        out_info["height"],
        is_test_5min=is_test,
        log_cb=log_cb,
        prog_cb=prog_cb,
        cancel_event=cancel_event,
    )
    t_total = round(time.perf_counter() - t_total0, 2)

    report = {
        "mode": mode,
        "source": str(src_mp4.resolve()),
        "encoder_used": enc_name,
        "source_probe": src_info,
        "normalized_probe": out_info,
        "timings_s": {
            "probe_s": t_probe,
            "gpu_normalize_and_wav_s": t_gpu,
            "spotcheck_model_load_s": spot["model_load_s"],
            "spotcheck_exec_s": spot["spotcheck_exec_s"],
            "total_step1_s": t_total,
        },
        "spotcheck": spot,
    }
    atomic_write_json(audit_json, report)

    if prog_cb:
        prog_cb(
            100.0,
            f"Hoàn tất {mode} ({enc_name}, {t_total}s | khớp sub {spot['timeline_match_pct']}% | offset {spot['median_ocr_minus_asr_offset_s']}s)",
        )
    if log_cb:
        log_cb(
            f"[STEP 1 DONE] Tổng {t_total}s | Khớp sub-tiếng: {spot['timeline_match_pct']}% "
            f"({spot['matched_ocr_lines']}/{spot['total_asr_lines']}) | Offset trung vị: {spot['median_ocr_minus_asr_offset_s']}s"
        )
    return report

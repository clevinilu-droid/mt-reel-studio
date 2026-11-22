"""
Step 1c (Hardsub OCR Extractor): Extract frame-accurate Chinese subtitles directly
from burned-in hardsubs using FFmpeg crop streaming + RapidOCR (ONNX CPU).
Outputs _engine/work/<movie>/zh_ocr.srt (or --out path).

Usage:
  python _engine/scripts/step1c_ocr_hardsub.py <movie_out_dir> [--fps 5] [--crop X:Y:W:H]
"""
import argparse
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from rapidocr_onnxruntime import RapidOCR

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step1_merge_and_transcribe import format_ts, probe_duration

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_STRIP_RE = re.compile(r"[\s，。！？、；：“”‘’\"'.,!?;:\-…·~()（）【】]+")


def clean_cjk_key(text: str) -> str:
    return PUNCT_STRIP_RE.sub("", text)


def similar_text(a: str, b: str) -> float:
    ka, kb = clean_cjk_key(a), clean_cjk_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    return SequenceMatcher(None, ka, kb).ratio()


def read_subtitle_text(ocr: RapidOCR, frame_bgr: np.ndarray, min_conf: float) -> tuple[str, float]:
    """Run RapidOCR on cropped subtitle strip and return (joined_text, avg_conf)."""
    res, _ = ocr(frame_bgr)
    if not res:
        return "", 0.0

    kept = []
    for item in res:
        box, txt, conf = item[0], item[1].strip(), float(item[2])
        if conf < min_conf or not CJK_RE.search(txt):
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        box_h = max(ys) - min(ys)
        # Reject oversized decorative title banners while keeping normal 1-2 line subs
        if box_h > 115:
            continue
        kept.append((min(ys), min(xs), txt, conf))

    if not kept:
        return "", 0.0

    # Group into visual lines (top-to-bottom), then left-to-right within each line
    kept.sort(key=lambda k: (round(k[0] / 35), k[1]))
    text = " ".join(k[2] for k in kept).strip()
    avg_conf = sum(k[3] for k in kept) / len(kept)
    return text, avg_conf


def _get_rec_providers(ocr: RapidOCR) -> list[str]:
    """Return active ONNX Runtime providers for RapidOCR's recognizer across 1.2.x and 1.4.x layouts."""
    rec_obj = getattr(ocr, "text_rec", None) or getattr(ocr, "text_recognizer", None)
    if rec_obj is None:
        return []
    sess = getattr(rec_obj, "session", None) or getattr(getattr(rec_obj, "infer", None), "session", None)
    if hasattr(sess, "session"):
        sess = sess.session
    return list(sess.get_providers()) if hasattr(sess, "get_providers") else []


def _preload_ort_cuda():
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def probe_rapidocr_cuda() -> bool:
    """Return True only if RapidOCR can genuinely run with CUDAExecutionProvider active."""
    try:
        import onnxruntime as ort
        _preload_ort_cuda()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        ocr = _build_ocr_engine(use_cuda=True)
        return "CUDAExecutionProvider" in _get_rec_providers(ocr)
    except Exception:
        return False


def _build_ocr_engine(use_cuda: bool) -> RapidOCR:
    """Create RapidOCR with angle cls disabled and optional CUDA acceleration on both Det and Rec."""
    if use_cuda:
        _preload_ort_cuda()
    ocr = RapidOCR(
        use_angle_cls=False,
        det_use_cuda=use_cuda,
        det_model_path=None,
        rec_use_cuda=use_cuda,
        rec_model_path=None,
    )
    if use_cuda:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                rec_providers = _get_rec_providers(ocr)
                if rec_providers and "CUDAExecutionProvider" not in rec_providers:
                    # rapidocr_onnxruntime 1.2.x forgets to strip 'rec_' on rec_use_cuda
                    from rapidocr_onnxruntime.rapid_ocr_api import root_dir
                    from rapidocr_onnxruntime.utils import concat_model_path, read_yaml
                    cfg = concat_model_path(read_yaml(str(root_dir / "config.yaml")))
                    cfg["Rec"]["use_cuda"] = True
                    RecCls = ocr.init_module(cfg["Rec"]["module_name"], cfg["Rec"]["class_name"])
                    if hasattr(ocr, "text_rec"):
                        ocr.text_rec = RecCls(cfg["Rec"])
                    else:
                        ocr.text_recognizer = RecCls(cfg["Rec"])
        except Exception as exc:
            print(f"[WARN] CUDA init check failed ({exc}); using default providers.", flush=True)
    return ocr


def _scan_slice(
    video_path_str: str,
    crop: tuple[int, int, int, int],
    fps: float,
    diff_thresh: float,
    min_conf: float,
    ss: float,
    dur: float | None,
    slice_idx: int,
    use_cuda: bool = False,
) -> list[dict]:
    """Stream a time slice [ss, ss+dur] via FFmpeg bgr24 crop and run RapidOCR on changed frames."""
    cx, cy, cw, ch = crop
    ocr = _build_ocr_engine(use_cuda=use_cuda)

    cmd = ["ffmpeg", "-v", "error"]
    if ss > 0:
        cmd += ["-ss", f"{ss:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += [
        "-i", video_path_str,
        # Force bgr24 BEFORE crop so odd dimensions are never clipped by YUV420 rounding
        "-vf", f"fps={fps},format=bgr24,crop={cw}:{ch}:{cx}:{cy}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=cw * ch * 3 * 8)
    frame_bytes = cw * ch * 3

    samples: list[dict] = []
    prev_mask = None
    last_ocr_idx = -999
    prev_text = ""
    prev_conf = 0.0
    ocr_calls = 0
    idx = 0
    recheck_interval = max(1, int(round(fps * 5.0)))
    x_pad = cw // 10

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((ch, cw, 3))

            # Inspect central subtitle zone with strict bright-white threshold to skip side lighting/clothes
            core = frame[:, x_pad : cw - x_pad]
            white_mask = (core.min(axis=2) > 218).astype(np.uint8)
            white_ratio = float(white_mask.mean())
            if white_ratio < 0.0012:
                text, conf = "", 0.0
                prev_text, prev_conf = "", 0.0
            else:
                need_ocr = prev_mask is None or not prev_text or (idx - last_ocr_idx) >= recheck_interval
                if not need_ocr:
                    changed = float(np.mean(white_mask != prev_mask))
                    if changed >= diff_thresh:
                        need_ocr = True
                if need_ocr:
                    text, conf = read_subtitle_text(ocr, frame, min_conf)
                    ocr_calls += 1
                    last_ocr_idx = idx
                    prev_text, prev_conf = text, conf
                else:
                    text, conf = prev_text, prev_conf

            prev_mask = white_mask
            samples.append({"t": ss + (idx / fps), "text": text, "conf": conf})
            idx += 1
            if idx % 500 == 0:
                print(f"  [OCR w{slice_idx}] {idx} frames ({ss + idx/fps:.0f}s mark, {ocr_calls} calls)...", flush=True)
    finally:
        proc.stdout.close()
        proc.wait()

    print(f"  [OCR w{slice_idx}] Finished slice: {idx} frames @ {fps} fps ({ocr_calls} OCR calls).", flush=True)
    return samples


def extract_ocr_segments(
    video_path: Path,
    crop: tuple[int, int, int, int],
    fps: float,
    diff_thresh: float,
    min_conf: float,
    workers: int = 1,
    total_dur: float | None = None,
    use_cuda: bool = False,
) -> list[dict]:
    """Stream cropped subtitle frames via FFmpeg (optionally parallel slices) and emit merged text segments."""
    if workers <= 1 or not total_dur or total_dur < 60.0:
        samples = _scan_slice(str(video_path), crop, fps, diff_thresh, min_conf, 0.0, None, 0, use_cuda=use_cuda)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        slice_dur = total_dur / workers
        tasks = [
            (str(video_path), crop, fps, diff_thresh, min_conf, i * slice_dur, slice_dur if i < workers - 1 else None, i, use_cuda)
            for i in range(workers)
        ]
        samples = []
        # Use spawn so each worker initializes CUDA/ONNX cleanly without fork deadlocks.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for part in ex.map(_scan_slice, *zip(*tasks)):
                samples.extend(part)
        samples.sort(key=lambda s: s["t"])

    print(f"[OCR] Total scanned {len(samples)} frames @ {fps} fps ({workers} worker(s)).", flush=True)

    # Bridge single-frame OCR dropouts between identical subtitle lines
    for i in range(1, len(samples) - 1):
        if not samples[i]["text"] and samples[i - 1]["text"] and samples[i + 1]["text"]:
            if similar_text(samples[i - 1]["text"], samples[i + 1]["text"]) >= 0.82:
                samples[i]["text"] = samples[i - 1]["text"]
                samples[i]["conf"] = samples[i - 1]["conf"]

    # Merge contiguous frames with matching subtitle text
    step_dt = 1.0 / fps
    segments: list[dict] = []
    for s in samples:
        txt = s["text"]
        if not txt:
            continue
        if segments and s["t"] - segments[-1]["end"] <= step_dt * 1.5 and similar_text(segments[-1]["text"], txt) >= 0.80:
            segments[-1]["end"] = s["t"] + step_dt
            segments[-1]["variants"].append((txt, s["conf"]))
        else:
            segments.append({
                "start": s["t"],
                "end": s["t"] + step_dt,
                "text": txt,
                "variants": [(txt, s["conf"])],
            })

    # Pick cleanest/longest high-confidence variant and discard tiny noise flashes
    cleaned: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        best_txt = max(seg["variants"], key=lambda v: (len(clean_cjk_key(v[0])), v[1]))[0]
        if dur < 0.25 and len(clean_cjk_key(best_txt)) < 2:
            continue
        cleaned.append({"start": seg["start"], "end": seg["end"], "text": best_txt})

    return cleaned


def write_srt(segments: list[dict], out_srt: Path):
    with out_srt.open("w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments, 1):
            fh.write(f"{i}\n")
            fh.write(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}\n")
            fh.write(f"{seg['text']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Extract burned-in Chinese hardsubs via RapidOCR.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--fps", type=float, default=2.5, help="Sampling rate in frames/second (default: 2.5)")
    parser.add_argument("--crop", type=str, default="40:1430:1000:160", help="Subtitle region X:Y:W:H on 1080x1920")
    parser.add_argument("--diff-thresh", type=float, default=0.020, help="White mask change ratio threshold to trigger OCR")
    parser.add_argument("--min-conf", type=float, default=0.55, help="Minimum OCR confidence threshold")
    parser.add_argument("--workers", type=int, default=4, help="Parallel FFmpeg+OCR workers across video slices (default: 4)")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA GPU execution for RapidOCR (use on Colab/Kaggle GPU)")
    parser.add_argument("--out", type=Path, default=None, help="Output SRT path (default: work/<movie>/zh_ocr.srt)")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    work_dir = ENGINE_DIR / "work" / out_dir.name
    merged_mp4 = work_dir / "01_merged_raw.mp4"
    if not merged_mp4.exists():
        raise FileNotFoundError(f"Missing merged video: {merged_mp4}")

    crop_parts = tuple(int(x) for x in args.crop.split(":"))
    if len(crop_parts) != 4:
        raise ValueError(f"Invalid --crop '{args.crop}', expected X:Y:W:H")

    workers = max(1, args.workers)
    t0 = time.time()
    dur = probe_duration(merged_mp4)
    print(f"[INFO] Extracting hardsubs from {merged_mp4.name} ({dur:.1f}s) crop={crop_parts} fps={args.fps} workers={workers} cuda={args.cuda}...", flush=True)
    segments = extract_ocr_segments(merged_mp4, crop_parts, args.fps, args.diff_thresh, args.min_conf, workers=workers, total_dur=dur, use_cuda=args.cuda)

    out_srt = args.out or (work_dir / "zh_ocr.srt")
    write_srt(segments, out_srt)
    print(f"[DONE] Extracted {len(segments)} hardsub lines in {time.time() - t0:.1f}s -> {out_srt}", flush=True)


if __name__ == "__main__":
    main()

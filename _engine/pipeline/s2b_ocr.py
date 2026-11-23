"""Bước 2B (s2b_ocr): Lọc Keyframe bằng PyAV dẫn đường theo Audio + Đọc chữ theo Batch (Qwen3.5 VLM / RapidOCR).

Cải tiến lớn so với bản cũ:
  - Dùng PyAV nhảy/đọc C native theo bản đồ thoại từ 2A (`4 fps` trong vùng nói, `1 fps` vùng im lặng).
  - Bỏ ngưỡng trắng cứng `>218`, thay bằng so lệch ảnh grayscale + biên (`L1 diff`) trong dải phụ đề hẹp.
  - Lưu micro-checkpoint JSONL sau mỗi batch 64 ảnh (`02b_ocr_raw.jsonl`), resume-safe 100%.
  - Hỗ trợ gọi Qwen3.5/Qwen3-VL qua vLLM OpenAI API (hoặc fallback RapidOCR local).
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from _engine.core.checkpoint import atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, ensure_onnx_cuda_libs, load_pipeline_config, patch_onnx_threads, release_vram

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def clean_cjk(text: str) -> str:
    return "".join(CJK_RE.findall(text or ""))


def _merge_intervals(segs: list[dict[str, Any]], pad: float = 0.55) -> list[tuple[float, float]]:
    blocks: list[list[float]] = []
    for s in segs:
        a = max(0.0, float(s["start"]) - pad)
        b = float(s["end"]) + pad
        if blocks and a <= blocks[-1][1] + 0.5:
            blocks[-1][1] = max(blocks[-1][1], b)
        else:
            blocks.append([a, b])
    return [(x[0], x[1]) for x in blocks]


def _in_intervals(t: float, blocks: list[tuple[float, float]]) -> bool:
    for a, b in blocks:
        if a <= t <= b:
            return True
        if a > t:
            return False
    return False


def extract_guided_keyframes(
    cfr_mp4: Path,
    asr_segs: list[dict[str, Any]],
    kf_dir: Path,
    manifest_kf_path: Path,
    crop_y0_ratio: float = 0.60,
    crop_y1_ratio: float = 0.90,
    speech_fps: float = 4.0,
    idle_fps: float = 1.2,
    diff_thresh: float = 9.5,
) -> list[dict[str, Any]]:
    """Dùng PyAV quét video, chỉ trích lưu các ảnh Keyframe khi dải phụ đề thực sự đổi chữ."""
    existing = read_json_safe(manifest_kf_path)
    if existing and existing.get("keyframes"):
        print(f"[S2B] Tái sử dụng {len(existing['keyframes'])} keyframe đã lọc sẵn.", flush=True)
        return existing["keyframes"]

    kf_dir.mkdir(parents=True, exist_ok=True)
    speech_blocks = _merge_intervals(asr_segs, pad=0.6)

    container = av.open(str(cfr_mp4))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    total_dur = float(container.duration / av.time_base) if container.duration else 0.0
    time_base = float(stream.time_base)

    keyframes: list[dict[str, Any]] = []
    prev_small: np.ndarray | None = None
    last_sample_t = -999.0
    last_emit = 0.0
    kf_idx = 0

    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = frame.pts * time_base
            in_speech = _in_intervals(t, speech_blocks)
            min_dt = (1.0 / speech_fps) if in_speech else (1.0 / idle_fps)
            if t - last_sample_t < min_dt - 0.005:
                continue
            last_sample_t = t

            rgb = frame.to_ndarray(format="rgb24")
            h, w, _ = rgb.shape
            y0, y1 = int(h * crop_y0_ratio), int(h * crop_y1_ratio)
            strip = rgb[y0:y1, :, :]

            # Tính ảnh grayscale thu nhỏ để đo biến đổi chữ nhanh không lệ thuộc ngưỡng màu trắng cứng
            gray = strip.mean(axis=2)
            small = gray[::4, ::4]
            changed = True
            if prev_small is not None and prev_small.shape == small.shape:
                mad = float(np.mean(np.abs(small - prev_small)))
                changed = mad >= diff_thresh

            if changed or not keyframes:
                kf_idx += 1
                fname = f"kf_{kf_idx:06d}.jpg"
                Image.fromarray(strip).save(kf_dir / fname, format="JPEG", quality=88)
                keyframes.append(
                    {
                        "id": kf_idx,
                        "t": round(t, 3),
                        "end": round(t + min_dt, 3),
                        "in_speech": in_speech,
                        "file": fname,
                    }
                )
                prev_small = small
            else:
                keyframes[-1]["end"] = round(t + min_dt, 3)

            now = time.monotonic()
            if total_dur > 0 and now - last_emit >= 0.5:
                last_emit = now
                pct = min(48.0, 4.0 + (t / total_dur) * 44.0)
                emit_progress(pct, f"PyAV lọc Keyframe: {t:.0f}/{total_dur:.0f}s ({len(keyframes)} ảnh)")
    finally:
        container.close()

    atomic_write_json(manifest_kf_path, {"count": len(keyframes), "keyframes": keyframes})
    print(f"[S2B] PyAV đã lọc xong {len(keyframes)} keyframe (từ {total_dur:.1f}s phim).", flush=True)
    return keyframes


def _load_ocr_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done[int(rec["id"])] = rec
        except Exception:
            continue
    return done


def _ocr_batch_vlm_openai(
    batch_items: list[dict[str, Any]],
    kf_dir: Path,
    endpoint: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Đọc batch ảnh keyframe qua vLLM OpenAI-compatible server (Qwen3.5 / Qwen3-VL)."""
    import urllib.request

    out: list[dict[str, Any]] = []
    prompt = "Chỉ chép chính xác dòng phụ đề tiếng Trung xuất hiện ở phần dưới ảnh. Không giải thích, bỏ chữ trên áo/biển hiệu. Nếu không có phụ đề trả về chuỗi rỗng."
    for item in batch_items:
        img_bytes = (kf_dir / item["file"]).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        body = json.dumps(
            {
                "model": model_name,
                "temperature": 0.0,
                "max_tokens": 96,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        txt = (data["choices"][0]["message"]["content"] or "").strip()
        out.append({"id": item["id"], "t": item["t"], "end": item["end"], "text": txt, "conf": 0.95 if clean_cjk(txt) else 0.0})
    return out


def _ocr_batch_rapidocr(batch_items: list[dict[str, Any]], kf_dir: Path, ocr_engine) -> list[dict[str, Any]]:
    """Đọc batch ảnh bằng RapidOCR và lọc bỏ nhiễu chữ áo/quần đùi/góc màn hình."""
    out: list[dict[str, Any]] = []
    for item in batch_items:
        arr = np.array(Image.open(kf_dir / item["file"]).convert("RGB"))
        sh, sw, _ = arr.shape
        # RapidOCR dùng BGR
        bgr = arr[:, :, ::-1].copy()
        res, _ = ocr_engine(bgr)
        kept = []
        if res:
            for r in res:
                box, txt, conf = r[0], r[1].strip(), float(r[2])
                ck = clean_cjk(txt)
                if conf < 0.60 or not ck:
                    continue
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                if (max(ys) - min(ys)) > max(90, int(sh * 0.34)):
                    continue
                # Hard-sub thoại luôn căn giữa khung hình; chữ 1-2 ký tự lệch biên (logo áo/quần) hoặc tự tin thấp phải loại bỏ
                cx = (min(xs) + max(xs)) / 2.0
                off_center = abs(cx - sw / 2.0) / max(1.0, sw)
                if len(ck) == 1 and (conf < 0.82 or off_center > 0.14):
                    continue
                if len(ck) == 2 and (off_center > 0.22 or (off_center > 0.15 and conf < 0.78)):
                    continue
                if off_center > 0.36:
                    continue
                kept.append((min(ys), min(xs), txt, conf))
        if kept:
            kept.sort(key=lambda k: (round(k[0] / 35), k[1]))
            joined = " ".join(k[2] for k in kept)
            avg_c = round(sum(k[3] for k in kept) / len(kept), 3)
        else:
            joined, avg_c = "", 0.0
        out.append({"id": item["id"], "t": item["t"], "end": item["end"], "text": joined, "conf": avg_c})
    return out


def _merge_ocr_lines(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gom các keyframe liên tiếp cùng 1 câu phụ đề thành segment gọn."""
    segs: list[dict[str, Any]] = []
    for r in sorted(records, key=lambda x: x["t"]):
        ck = clean_cjk(r["text"])
        if not ck:
            continue
        if (
            segs
            and r["t"] - segs[-1]["end"] <= 0.65
            and difflib.SequenceMatcher(None, clean_cjk(segs[-1]["text"]), ck).ratio() >= 0.78
        ):
            segs[-1]["end"] = max(segs[-1]["end"], r["end"])
            if len(ck) > len(clean_cjk(segs[-1]["text"])):
                segs[-1]["text"] = r["text"]
                segs[-1]["conf"] = max(segs[-1]["conf"], r["conf"])
        else:
            segs.append(
                {
                    "start": round(r["t"], 3),
                    "end": round(r["end"], 3),
                    "text": r["text"],
                    "conf": r["conf"],
                }
            )
    for idx, s in enumerate(segs, start=1):
        s["id"] = idx
    return segs


def run_step2b(target_dir: Path, force: bool = False, batch_size: int = 16) -> dict[str, Any]:
    out_json = target_dir / "02b_ocr_raw.json"
    jsonl_path = target_dir / "02b_ocr_raw.jsonl"
    kf_dir = target_dir / "keyframes"
    kf_manifest = target_dir / "02b_keyframes.json"
    cfr_mp4 = target_dir / "01_cfr_raw.mp4"
    asr_json = target_dir / "02a_audio_asr.json"

    if not cfr_mp4.exists():
        raise FileNotFoundError(f"Thiếu video CFR Bước 1: {cfr_mp4}")

    if not force:
        existing = read_json_safe(out_json)
        if existing and existing.get("segments"):
            emit_progress(100.0, f"Tái sử dụng checkpoint 2B ({len(existing['segments'])} dòng sub)")
            return existing
    else:
        jsonl_path.unlink(missing_ok=True)
        kf_manifest.unlink(missing_ok=True)

    t0 = time.perf_counter()
    asr_data = read_json_safe(asr_json) or {}
    asr_segs = asr_data.get("segments", [])

    emit_progress(3.0, "PyAV đang lọc Keyframe theo bản đồ thoại 2A...")
    keyframes = extract_guided_keyframes(cfr_mp4, asr_segs, kf_dir, kf_manifest)

    done_map = _load_ocr_jsonl(jsonl_path)
    todo = [k for k in keyframes if k["id"] not in done_map]

    cfg = load_pipeline_config()
    vlm_url = os.environ.get("MT_VLM_BASE_URL", "").strip()
    vlm_model = cfg.get("models", {}).get("ocr_vlm_model", "Qwen/Qwen3.5-4B")
    engine_used = "vlm_openai" if vlm_url else "rapidocr"

    ocr_engine = None
    if not vlm_url:
        emit_progress(48.5, "Đang nạp RapidOCR (kiểm tra GPU CUDA)...")
        ensure_onnx_cuda_libs()
        patch_onnx_threads(4)
        import onnxruntime as ort
        from rapidocr_onnxruntime import RapidOCR

        has_cuda_ep = "CUDAExecutionProvider" in ort.get_available_providers()
        if has_cuda_ep:
            try:
                gpu_ocr = RapidOCR(
                    use_angle_cls=False,
                    det_use_cuda=True,
                    det_model_path=None,
                    rec_use_cuda=True,
                    rec_model_path=None,
                )
                # Thử 1 ảnh nhỏ để xác nhận cuDNN/CUDA ExecutionProvider chạy thật trước khi quét batch
                gpu_ocr(np.zeros((64, 160, 3), dtype=np.uint8))
                ocr_engine = gpu_ocr
                engine_used = "rapidocr_cuda"
                batch_size = max(batch_size, 64)
            except Exception as exc:
                print(f"[S2B CUDA WARN] RapidOCR CUDA lỗi ({exc}) -> fallback rapidocr_cpu", flush=True)

        if ocr_engine is None:
            ocr_engine = RapidOCR(
                use_angle_cls=False,
                det_use_cuda=False,
                det_model_path=None,
                rec_use_cuda=False,
                rec_model_path=None,
            )
            engine_used = "rapidocr_cpu"

    total_todo = len(todo)
    processed = 0
    for i in range(0, total_todo, batch_size):
        batch = todo[i : i + batch_size]
        if vlm_url:
            res_batch = _ocr_batch_vlm_openai(batch, kf_dir, vlm_url, vlm_model)
        else:
            res_batch = _ocr_batch_rapidocr(batch, kf_dir, ocr_engine)

        with jsonl_path.open("a", encoding="utf-8") as fh:
            for r in res_batch:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                done_map[r["id"]] = r

        processed += len(batch)
        pct = 50.0 + (processed / max(1, total_todo)) * 46.0
        emit_progress(pct, f"OCR ({engine_used}): {len(done_map)}/{len(keyframes)} ảnh (đã lưu checkpoint)")

    del ocr_engine
    release_vram()

    merged = _merge_ocr_lines(list(done_map.values()))
    elapsed = round(time.perf_counter() - t0, 2)
    payload = {
        "engine": engine_used,
        "keyframes_count": len(keyframes),
        "segments_count": len(merged),
        "elapsed_s": elapsed,
        "segments": merged,
    }
    atomic_write_json(out_json, payload)
    emit_progress(100.0, f"Xong 2B ({engine_used}): {len(merged)} dòng sub từ {len(keyframes)} ảnh ({elapsed}s)")
    print(f"[S2B DONE] {engine_used}: {len(merged)} dòng sub trong {elapsed}s -> {out_json.name}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Bước 2B: PyAV lọc Keyframe + Batch OCR (Qwen3.5 VLM / RapidOCR).")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu")
    parser.add_argument("--force", action="store_true", help="Quét lại từ đầu")
    parser.add_argument("--batch-size", type=int, default=64, help="Số ảnh mỗi micro-checkpoint JSONL")
    args = parser.parse_args()
    run_step2b(args.target_dir.resolve(), force=args.force, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

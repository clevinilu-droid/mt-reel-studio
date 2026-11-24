"""
Bước 3.3 (Tự động hóa tốc độ cao + đối chiếu Gemini Vision cho câu mờ/đọc yếu):
Đọc toàn bộ dải ảnh trong `hardsub_grids/sub_sheet_*.jpg` khớp 1:1 với `hardsub_events.json`,
cắt bỏ thanh số thứ tự bên trái (`x >= 230`), nhận diện bằng RapidOCR (~0.02s/dải),
dải nào trống hoặc confidence < 0.75 sẽ tự động gửi Gemini Vision kiểm tra chéo,
rồi xuất thẳng `zh_hardsub_raw.srt` + `ocr_audit.json`.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent


def fmt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _imread_utf8(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Khong doc duoc anh: {path}")
    return img


def load_strips(step3_dir: Path, events: list[dict], bar_w: int = 230, per_sheet: int = 25) -> list[np.ndarray]:
    sheets = sorted((step3_dir / "hardsub_grids").glob("sub_sheet_*.jpg"))
    if not sheets:
        raise FileNotFoundError(f"Khong tim thay sub_sheet_*.jpg trong {step3_dir / 'hardsub_grids'}")
    first = _imread_utf8(sheets[0])
    strip_h = first.shape[0] // min(per_sheet, len(events))
    strips: list[np.ndarray] = []
    for s_path in sheets:
        img = _imread_utf8(s_path)
        n_rows = img.shape[0] // strip_h
        for r in range(n_rows):
            crop = img[r * strip_h : (r + 1) * strip_h, bar_w:]
            strips.append(crop)
            if len(strips) >= len(events):
                return strips
    return strips


def gemini_fallback_batch(items: list[tuple[int, np.ndarray]]) -> dict[int, str]:
    """Dùng Gemini Vision kiểm tra các dải phụ đề RapidOCR không chắc chắn."""
    if not items:
        return {}
    load_dotenv(ROOT / ".env", override=True)
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return {}
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    out: dict[int, str] = {}
    for idx in range(0, len(items), 15):
        chunk = items[idx : idx + 15]
        # Ghép dọc các dải nhỏ đánh số rõ để Gemini đọc 1 lần
        h = chunk[0][1].shape[0]
        w = chunk[0][1].shape[1]
        canvas = np.full((h * len(chunk), w + 140, 3), 20, dtype=np.uint8)
        id_map = []
        for r_i, (ev_id, img) in enumerate(chunk):
            canvas[r_i * h : (r_i + 1) * h, 140:] = img
            cv2.putText(canvas, f"#{ev_id}", (12, r_i * h + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)
            id_map.append(ev_id)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue
        prompt = (
            "Read the Simplified Chinese movie subtitle on each numbered row (#ID). "
            "Return STRICT JSON mapping string ID (without #) to exact Simplified Chinese text "
            "(empty string if no Chinese subtitle on that row)."
        )
        for model_name in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
                )
                parsed = json.loads(resp.text)
                for ev_id in id_map:
                    val = parsed.get(str(ev_id)) or parsed.get(f"#{ev_id}") or ""
                    out[ev_id] = str(val).strip()
                break
            except Exception as e:
                print(f"[OCR-GEMINI] {model_name} retry/err: {str(e)[:120]}", flush=True)
                time.sleep(1.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR rapid + Gemini fallback on Step 3 hardsub strips")
    ap.add_argument("step3_dir", type=Path, help="Thu muc 02_buoc3_hardsub_va_am_thanh")
    ap.add_argument("--conf-thresh", type=float, default=0.72)
    ap.add_argument("--no-gemini", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    events = json.loads((args.step3_dir / "hardsub_events.json").read_text(encoding="utf-8"))
    strips = load_strips(args.step3_dir, events)
    if len(strips) != len(events):
        raise RuntimeError(f"Mismatch strips ({len(strips)}) vs events ({len(events)})")

    ocr = RapidOCR()
    results: list[dict] = []
    need_gemini: list[tuple[int, np.ndarray]] = []

    for ev, strip in zip(events, strips):
        res, _ = ocr(strip)
        lines_txt = []
        min_score = 1.0
        if res:
            # Sắp xếp từ trên xuống dưới theo tọa độ Y tâm box
            res_sorted = sorted(res, key=lambda x: (sum(pt[1] for pt in x[0]) / 4.0, x[0][0][0]))
            for box, txt, score in res_sorted:
                t_clean = str(txt).strip()
                sc = float(score)
                if t_clean:
                    lines_txt.append(t_clean)
                    min_score = min(min_score, sc)
        joined = " ".join(lines_txt).strip()
        if (not joined or min_score < args.conf_thresh) and not args.no_gemini:
            need_gemini.append((ev["id"], strip))
        results.append({
            "id": ev["id"],
            "start": ev["start"],
            "end": ev["end"],
            "text": joined,
            "score": round(min_score if joined else 0.0, 3),
        })

    if need_gemini:
        print(f"[OCR] Sending {len(need_gemini)}/{len(results)} uncertain/empty strips to Gemini Vision...", flush=True)
        fixed = gemini_fallback_batch(need_gemini)
        for item in results:
            if item["id"] in fixed and fixed[item["id"]]:
                item["text"] = fixed[item["id"]]
                item["gemini_fixed"] = True

    kept = []
    for item in results:
        if not item["text"]:
            continue
        kept.append(item)

    srt_blocks = []
    for idx, item in enumerate(kept, 1):
        srt_blocks.append(f"{idx}\n{fmt_ts(item['start'])} --> {fmt_ts(item['end'])}\n{item['text']}\n")

    out_srt = args.step3_dir / "zh_hardsub_raw.srt"
    out_srt.write_text("\n".join(srt_blocks), encoding="utf-8")
    (args.step3_dir / "ocr_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] OCR {len(events)} strips -> {len(kept)} non-empty lines in {time.time() - t0:.1f}s -> {out_srt.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

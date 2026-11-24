"""
Bước 3.3 bản siêu tốc (Đọc theo từng Tờ ảnh ghép + Đa luồng CPU + Đối chiếu Gemini Vision):
- Chạy RapidOCR trực tiếp trên cả tờ `sub_sheet_*.jpg` (chỉ ~6.5s cho 25 câu thay vì tách từng câu),
  kết hợp `ProcessPoolExecutor` chia đều cho nhiều nhân CPU (nhanh gấp 8-10 lần).
- Gom chữ từng hàng theo tọa độ tâm Y (`row = int(y_center // strip_h)`).
- Câu nào trống hoặc confidence < `--conf-thresh` (mặc định 0.58) sẽ gửi Gemini Vision đọc bổ sung.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_OCR = None


def _imread_utf8(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Khong doc duoc anh: {path}")
    return img


def fmt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _process_sheet(task: tuple[int, str, int, int, int]) -> list[tuple[int, str, float]]:
    sheet_idx, sheet_path_str, strip_h, per_sheet, bar_w = task
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()

    img = _imread_utf8(Path(sheet_path_str))
    n_rows = img.shape[0] // strip_h
    res, _ = _OCR(img[:, bar_w:])

    rows: dict[int, list[tuple[float, float, str, float]]] = {r: [] for r in range(n_rows)}
    for item in res or []:
        box, txt, sc = item[0], str(item[1]).strip(), float(item[2])
        if not txt:
            continue
        yc = sum(p[1] for p in box) / 4.0
        xc = sum(p[0] for p in box) / 4.0
        r = int(yc // strip_h)
        if 0 <= r < n_rows:
            rows[r].append((yc, xc, txt, sc))

    out: list[tuple[int, str, float]] = []
    base_idx = sheet_idx * per_sheet
    for r in range(n_rows):
        items = sorted(rows[r], key=lambda x: (round(x[0] / 25.0), x[1]))
        txt = " ".join(x[2] for x in items).strip()
        min_sc = min((x[3] for x in items), default=0.0)
        out.append((base_idx + r, txt, round(min_sc, 3)))
    return out


def gemini_fallback_batch(items: list[tuple[int, np.ndarray]]) -> dict[int, str]:
    if not items:
        return {}
    from dotenv import load_dotenv
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
        h, w = chunk[0][1].shape[:2]
        canvas = np.full((h * len(chunk), w + 140, 3), 20, dtype=np.uint8)
        id_map = []
        for r_i, (ev_id, img) in enumerate(chunk):
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            canvas[r_i * h : (r_i + 1) * h, 140:] = img
            cv2.putText(canvas, f"#{ev_id}", (12, r_i * h + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)
            id_map.append(ev_id)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue
        prompt = (
            "Read the Simplified Chinese movie subtitle on each numbered row (#ID). "
            "Return STRICT JSON object mapping each string ID (digits only, without #) to exact Simplified Chinese text "
            "(empty string if no Chinese subtitle on that row)."
        )
        for model_name in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
                )
                parsed = json.loads(resp.text)
                if isinstance(parsed, list):
                    norm_map = {}
                    for entry in parsed:
                        if isinstance(entry, dict):
                            k = str(entry.get("id") or entry.get("row") or "").lstrip("#").strip()
                            v = entry.get("text") or entry.get("subtitle") or ""
                            if k:
                                norm_map[k] = str(v)
                    parsed = norm_map
                for ev_id in id_map:
                    out[ev_id] = str(parsed.get(str(ev_id)) or parsed.get(f"#{ev_id}") or "").strip()
                print(f"[OCR-GEMINI] Batch {idx//15 + 1}/{(len(items)+14)//15} OK", flush=True)
                break
            except Exception as e:
                print(f"[OCR-GEMINI] {model_name} err: {str(e)[:120]}", flush=True)
                time.sleep(1.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sheet-level multiprocess OCR + Gemini fallback")
    ap.add_argument("step3_dir", type=Path)
    ap.add_argument("--conf-thresh", type=float, default=0.58)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-sheet", type=int, default=25)
    ap.add_argument("--bar-w", type=int, default=230)
    ap.add_argument("--no-gemini", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    events = json.loads((args.step3_dir / "hardsub_events.json").read_text(encoding="utf-8"))
    sheets = sorted(
        (args.step3_dir / "hardsub_grids").glob("sub_sheet_*.jpg"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    first = _imread_utf8(sheets[0])
    strip_h = first.shape[0] // min(args.per_sheet, len(events))

    cache_path = args.step3_dir / "ocr_rapid_raw_cache.json"
    by_idx: dict[int, tuple[str, float]] = {}
    if cache_path.exists():
        raw_c = json.loads(cache_path.read_text(encoding="utf-8"))
        if raw_c.get("count") == len(events):
            by_idx = {int(k): (v[0], float(v[1])) for k, v in raw_c["items"].items()}
            print(f"[OCR-FAST] Loaded raw cache for {len(by_idx)} strips!", flush=True)

    if len(by_idx) != len(events):
        tasks = [(i, str(sp), strip_h, args.per_sheet, args.bar_w) for i, sp in enumerate(sheets)]
        done_sheets = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res_list in ex.map(_process_sheet, tasks):
                for idx, txt, sc in res_list:
                    by_idx[idx] = (txt, sc)
                done_sheets += 1
                if done_sheets % 20 == 0 or done_sheets == len(sheets):
                    print(f"[OCR-FAST] Sheets: {done_sheets}/{len(sheets)} ({time.time() - t0:.1f}s)", flush=True)
        cache_path.write_text(
            json.dumps({"count": len(events), "items": {str(k): list(v) for k, v in by_idx.items()}}, ensure_ascii=False),
            encoding="utf-8",
        )

    results = []
    need_gemini_meta: list[tuple[int, int]] = []  # (ev_id, idx)
    for idx, ev in enumerate(events):
        txt, sc = by_idx.get(idx, ("", 0.0))
        if (not txt or sc < args.conf_thresh) and not args.no_gemini:
            need_gemini_meta.append((ev["id"], idx))
        results.append({
            "id": ev["id"], "start": ev["start"], "end": ev["end"], "text": txt, "score": sc
        })

    if need_gemini_meta:
        print(f"[OCR-FAST] Checking {len(need_gemini_meta)}/{len(results)} low-conf strips via Gemini...", flush=True)
        # Chỉ load đúng những tờ chứa câu yếu
        need_items = []
        cache_sheet = {}
        for ev_id, idx in need_gemini_meta:
            s_i = idx // args.per_sheet
            r_i = idx % args.per_sheet
            if s_i not in cache_sheet:
                cache_sheet[s_i] = _imread_utf8(sheets[s_i])
            im = cache_sheet[s_i]
            need_items.append((ev_id, im[r_i * strip_h : (r_i + 1) * strip_h, args.bar_w:]))
        fixed = gemini_fallback_batch(need_items)
        for item in results:
            if item["id"] in fixed and fixed[item["id"]]:
                item["text"] = fixed[item["id"]]
                item["gemini_fixed"] = True

    kept = [r for r in results if r["text"]]
    srt_blocks = [f"{i}\n{fmt_ts(r['start'])} --> {fmt_ts(r['end'])}\n{r['text']}\n" for i, r in enumerate(kept, 1)]
    out_srt = args.step3_dir / "zh_hardsub_raw.srt"
    out_srt.write_text("\n".join(srt_blocks), encoding="utf-8")
    (args.step3_dir / "ocr_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Sheet-OCR {len(events)} strips -> {len(kept)} non-empty lines in {time.time() - t0:.1f}s -> {out_srt.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

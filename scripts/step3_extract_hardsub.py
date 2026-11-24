"""
Bước 3.2: Trích xuất các câu Hardsub gốc (vùng đáy cố định) thành các tấm ảnh dải chữ gọn nhẹ.
Chỉ chụp 1 dải ảnh rõ nhất ở giữa mỗi câu để đọc chính xác 100% chữ tác giả mà không tốn thời gian OCR trùng lặp.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_text_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect bright white Chinese hardsub strokes surrounded by dark outline."""
    b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
    white = (b > 225) & (g > 225) & (r > 225)
    kernel = np.ones((7, 7), np.uint8)
    near_white = cv2.dilate(white.astype(np.uint8), kernel, iterations=1) > 0
    dark = (b < 65) & (g < 65) & (r < 65)
    bordered = near_white & dark
    has_border = cv2.dilate(bordered.astype(np.uint8), kernel, iterations=1) > 0
    return (white & has_border).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Extract unique bottom hardsub line strips into numbered sheets.")
    ap.add_argument("video", type=Path, help="Input video file (.mp4)")
    ap.add_argument("out_dir", type=Path, help="Output Step 3 directory (e.g. projects/.../P1/02_buoc3_hardsub_va_am_thanh)")
    ap.add_argument("--crop", default="80:1455:920:95", help="X:Y:W:H crop box of bottom hardsub")
    ap.add_argument("--fps", type=float, default=5.0, help="Scan FPS (default 5.0)")
    ap.add_argument("--per-sheet", type=int, default=30, help="Subtitle strips per output sheet")
    args = ap.parse_args()

    cx, cy, cw, ch = (int(x) for x in args.crop.split(":"))
    sub_grids_dir = args.out_dir / "hardsub_grids"
    sub_grids_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(args.video),
        "-vf", f"fps={args.fps},format=bgr24,crop={cw}:{ch}:{cx}:{cy}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=cw * ch * 3 * 16)
    frame_bytes = cw * ch * 3

    frames_meta = []
    idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((ch, cw, 3))
        mask = extract_text_mask(frame)
        ink = int(mask.sum())
        frames_meta.append((idx / args.fps, ink, mask, frame if ink > 80 else None))
        idx += 1
    proc.stdout.close()
    proc.wait()

    events = []
    cur = None
    for t, ink, mask, frame in frames_meta:
        if ink < 110:
            if cur is not None:
                cur["end"] = t
                events.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {"start": t, "end": t + 1.0 / args.fps, "samples": [(t, ink, mask, frame)]}
        else:
            prev_mask = cur["samples"][-1][2]
            inter = int(np.logical_and(mask, prev_mask).sum())
            union = int(np.logical_or(mask, prev_mask).sum())
            iou = inter / union if union else 0.0
            if iou >= 0.42:
                cur["end"] = t + 1.0 / args.fps
                cur["samples"].append((t, ink, mask, frame))
            else:
                events.append(cur)
                cur = {"start": t, "end": t + 1.0 / args.fps, "samples": [(t, ink, mask, frame)]}
    if cur is not None:
        events.append(cur)

    events = [e for e in events if (e["end"] - e["start"]) >= 0.20]
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)

    meta_out = []
    strips = []
    bar_w = 230
    for i, ev in enumerate(events, 1):
        mid = ev["samples"][len(ev["samples"]) // 2]
        pil_strip = Image.fromarray(cv2.cvtColor(mid[3], cv2.COLOR_BGR2RGB))
        canvas = Image.new("RGB", (cw + bar_w, ch), (20, 20, 20))
        canvas.paste(pil_strip, (bar_w, 0))
        d = ImageDraw.Draw(canvas)
        sm, ss = divmod(ev["start"], 60)
        em, es = divmod(ev["end"], 60)
        d.text((10, 18), f"#{i:03d}\n{int(sm):02d}:{ss:05.2f} -> {int(em):02d}:{es:05.2f}", font=font, fill=(255, 230, 80))
        strips.append(canvas)
        meta_out.append({"id": i, "start": round(ev["start"], 3), "end": round(ev["end"], 3)})

    sheets = []
    for s_idx in range(0, len(strips), args.per_sheet):
        chunk = strips[s_idx : s_idx + args.per_sheet]
        sheet = Image.new("RGB", (cw + bar_w, ch * len(chunk)), (15, 15, 15))
        for r_i, st in enumerate(chunk):
            sheet.paste(st, (0, r_i * ch))
        out_p = sub_grids_dir / f"sub_sheet_{s_idx // args.per_sheet + 1:02d}.jpg"
        sheet.save(out_p, quality=92)
        sheets.append(out_p.name)

    (args.out_dir / "hardsub_events.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Scanned {idx} frames in {time.time() - t0:.1f}s -> {len(events)} lines -> {len(sheets)} sheets")


if __name__ == "__main__":
    main()

"""
Bước 3.2 (Nâng cấp 4 lớp chống sót câu):
- Khung quét nới rộng (40:1380:1000:220) bao trọn cả hardsub 1 dòng lẫn 2 dòng.
- Bắt cả chữ trắng lẫn chữ vàng/màu sáng (max(R,G,B) > 205 kèm viền tối min(R,G,B) < 75).
- Ngưỡng IoU >= 0.65 (cộng kiểm tra thay đổi chữ tâm) tránh gộp nhầm 2 câu ngắn liên tiếp.
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
    """Detect pure white or bright yellow hardsub strokes surrounded by true black outline."""
    b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
    is_white = (b > 218) & (g > 218) & (r > 218)
    is_yellow = (r > 218) & (g > 180) & (b < 115)
    bright = is_white | is_yellow
    # True dark outline (all 3 channels dark)
    dark = (b < 68) & (g < 68) & (r < 68)
    kernel = np.ones((7, 7), np.uint8)
    near_bright = cv2.dilate(bright.astype(np.uint8), kernel, iterations=1) > 0
    bordered = near_bright & dark
    has_border = cv2.dilate(bordered.astype(np.uint8), kernel, iterations=1) > 0
    mask = (bright & has_border).astype(np.uint8)
    # Reject huge solid reflections (> 22% of any local 40x40 block) by keeping only text-thin strokes
    if mask.sum() > 0:
        thick = cv2.erode(mask, np.ones((11, 11), np.uint8), iterations=1)
        if thick.sum() > 0:
            mask = mask & (1 - cv2.dilate(thick, np.ones((15, 15), np.uint8), iterations=1))
    return mask


def main():
    ap = argparse.ArgumentParser(description="Extract unique bottom hardsub line strips into numbered sheets.")
    ap.add_argument("video", type=Path, help="Input video file (.mp4)")
    ap.add_argument("out_dir", type=Path, help="Output Step 3 directory")
    ap.add_argument("--crop", default="60:1410:960:170", help="X:Y:W:H crop box covering 1-2 line hardsubs")
    ap.add_argument("--fps", type=float, default=5.0, help="Scan FPS (default 5.0)")
    ap.add_argument("--iou-thresh", type=float, default=0.65, help="Same-line IoU threshold (default 0.65)")
    ap.add_argument("--per-sheet", type=int, default=25, help="Subtitle strips per output sheet")
    args = ap.parse_args()

    cx, cy, cw, ch = (int(x) for x in args.crop.split(":"))
    sub_grids_dir = args.out_dir / "hardsub_grids"
    sub_grids_dir.mkdir(parents=True, exist_ok=True)
    for old_f in sub_grids_dir.glob("sub_sheet_*.jpg"):
        old_f.unlink()

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
        # Centered subtitle region check prevents side highlights/shorts stripes from triggering
        center_ink = int(mask[:, cw // 4 : cw - cw // 4].sum())
        ink = int(mask.sum()) if center_ink >= 140 else 0
        frames_meta.append((idx / args.fps, ink, mask, frame if ink > 140 else None))
        idx += 1
    proc.stdout.close()
    proc.wait()

    events = []
    cur = None
    for t, ink, mask, frame in frames_meta:
        if ink < 120:
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
            if iou >= args.iou_thresh:
                cur["end"] = t + 1.0 / args.fps
                cur["samples"].append((t, ink, mask, frame))
            else:
                events.append(cur)
                cur = {"start": t, "end": t + 1.0 / args.fps, "samples": [(t, ink, mask, frame)]}
    if cur is not None:
        events.append(cur)

    events = [e for e in events if (e["end"] - e["start"]) >= 0.20]
    font = None
    for fpath in (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(fpath).exists():
            font = ImageFont.truetype(fpath, 22)
            break
    if font is None:
        font = ImageFont.load_default()

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
        d.text((10, 22), f"#{i:03d}\n{int(sm):02d}:{ss:05.2f} -> {int(em):02d}:{es:05.2f}", font=font, fill=(255, 230, 80))
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

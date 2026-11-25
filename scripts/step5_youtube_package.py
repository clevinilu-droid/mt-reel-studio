"""
Đóng gói thành phẩm YouTube chuẩn hàng loạt (Bước 5.4):
1. Tạo Thumbnail 16:9 (1280x720, < 2MB) từ poster.jpg gốc:
   - Nền lấy từ poster làm mờ + làm tối nhẹ để khớp chuẩn 16:9 (không viền đen thô).
   - Poster gốc sắc nét căn giữa, thêm duy nhất 2 huy hiệu rõ nét: "HOT" và "P1"/"P2"
     đặt ở góc trên (tránh che mặt nhân vật và tránh góc dưới-phải bị YouTube chèn thời lượng).
   - Xuất kèm thumb_poster.jpg (dọc) nếu cần lưu kho.
2. Tạo bộ Tiêu đề, Mô tả, Chapters & Hashtags riêng cho 4 ngôn ngữ (vi, en, ja, ko):
   - KHÔNG nhồi 4 ngôn ngữ vào 1 tiêu đề hay 1 đoạn mô tả (tránh bị YouTube đánh spam / Reused Content).
   - Xuất `metadata_4langs.json` (tương thích YouTube API `localizations`) và 4 file `.txt`
     để copy/dán trực tiếp vào mục Languages trong YouTube Studio hoặc dùng khi upload tự động.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Chọn font đậm có sẵn trên Windows/Linux để vẽ huy hiệu HOT và P1/P2."""
    candidates = [
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_badge(
    base: Image.Image,
    text: str,
    center_xy: tuple[int, int],
    fill_rgba: tuple[int, int, int, int],
    text_rgb: tuple[int, int, int],
    border_rgba: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    pad_x: int = 34,
    pad_y: int = 16,
    radius: int = 18,
    tilt_deg: float = 0.0,
) -> None:
    """Vẽ huy hiệu có bóng đổ + viền rõ nét, không che vùng trung tâm."""
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    margin = 28
    stamp = Image.new("RGBA", (bw + margin * 2, bh + margin * 2), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)

    # Bóng đổ mềm phía sau huy hiệu
    sd.rounded_rectangle(
        [margin + 4, margin + 8, margin + bw + 4, margin + bh + 8],
        radius=radius,
        fill=(0, 0, 0, 175),
    )
    # Nền + viền huy hiệu
    sd.rounded_rectangle(
        [margin, margin, margin + bw, margin + bh],
        radius=radius,
        fill=fill_rgba,
        outline=border_rgba,
        width=5,
    )
    # Căn chữ chính xác theo textbbox
    tx = margin + (bw - tw) // 2 - bbox[0]
    ty = margin + (bh - th) // 2 - bbox[1]
    sd.text((tx + 2, ty + 3), text, font=font, fill=(0, 0, 0, 140))
    sd.text((tx, ty), text, font=font, fill=text_rgb + (255,))

    if tilt_deg:
        stamp = stamp.rotate(tilt_deg, resample=Image.Resampling.BICUBIC, expand=True)

    px = center_xy[0] - stamp.width // 2
    py = center_xy[1] - stamp.height // 2
    base.alpha_composite(stamp, (px, py))


def build_thumbnails(poster_path: Path, part_label: str, out_dir: Path) -> tuple[Path, Path]:
    """Tạo thumbnail 16:9 (1280x720) chuẩn YouTube và bản dọc (thumb_poster.jpg)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    poster = Image.open(poster_path).convert("RGB")
    part_text = part_label.upper().strip()

    # 1. Canvas 1280x720 (16:9 YouTube)
    W, H = 1280, 720
    # Nền blur phủ kín chiều ngang từ chính poster để giữ tông màu lửa/cháy đồng bộ
    bg_scale = max(W / poster.width, H / poster.height)
    bg_w = int(round(poster.width * bg_scale))
    bg_h = int(round(poster.height * bg_scale))
    bg = poster.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    left = (bg_w - W) // 2
    top = (bg_h - H) // 2
    bg = bg.crop((left, top, left + W, top + H)).filter(ImageFilter.GaussianBlur(radius=24))
    # Làm tối nền 45% để poster giữa và 2 badge bật nổi
    dark_overlay = Image.new("RGBA", (W, H), (8, 8, 12, 115))
    canvas = Image.alpha_composite(bg.convert("RGBA"), dark_overlay)

    # Đặt poster sắc nét ở giữa (chừa mép nhẹ tạo chiều sâu)
    fg_h = H
    fg_w = int(round(poster.width * (fg_h / poster.height)))
    fg = poster.resize((fg_w, fg_h), Image.Resampling.LANCZOS).convert("RGBA")
    fg_x = (W - fg_w) // 2

    # Đổ bóng nhẹ 2 bên poster
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh_draw = ImageDraw.Draw(shadow)
    sh_draw.rectangle([fg_x - 12, 0, fg_x + fg_w + 12, H], fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(fg, (fg_x, 0))

    # Vẽ 2 huy hiệu to rõ ở 2 khoảng trống trên-trái và trên-phải (không đè mặt nhân vật & không đè thời lượng YouTube)
    font_big = _pick_font(124)
    _draw_badge(
        canvas,
        "HOT",
        center_xy=(198, 132),
        fill_rgba=(225, 24, 24, 248),
        text_rgb=(255, 255, 255),
        border_rgba=(255, 215, 64, 255),
        font=font_big,
        pad_x=38,
        pad_y=18,
        radius=22,
        tilt_deg=-4.0,
    )
    _draw_badge(
        canvas,
        part_text,
        center_xy=(W - 198, 132),
        fill_rgba=(250, 204, 21, 250),
        text_rgb=(17, 17, 17),
        border_rgba=(255, 255, 255, 255),
        font=font_big,
        pad_x=44,
        pad_y=18,
        radius=22,
        tilt_deg=4.0,
    )

    thumb_16x9 = out_dir / "thumb_1280x720.jpg"
    canvas.convert("RGB").save(thumb_16x9, format="JPEG", quality=93, optimize=True)

    # 2. Bản dọc từ poster gốc (chỉ thêm HOT nhỏ góc trên-trái và P1/P2 góc trên-phải)
    pw, ph = 900, int(round(900 * poster.height / poster.width))
    v_canvas = poster.resize((pw, ph), Image.Resampling.LANCZOS).convert("RGBA")
    font_v = _pick_font(78)
    _draw_badge(
        v_canvas,
        "HOT",
        center_xy=(125, 84),
        fill_rgba=(225, 24, 24, 245),
        text_rgb=(255, 255, 255),
        border_rgba=(255, 215, 64, 255),
        font=font_v,
        pad_x=24,
        pad_y=12,
        radius=16,
        tilt_deg=-4.0,
    )
    _draw_badge(
        v_canvas,
        part_text,
        center_xy=(pw - 115, 84),
        fill_rgba=(250, 204, 21, 248),
        text_rgb=(17, 17, 17),
        border_rgba=(255, 255, 255, 255),
        font=font_v,
        pad_x=26,
        pad_y=12,
        radius=16,
        tilt_deg=4.0,
    )
    thumb_vert = out_dir / "thumb_poster.jpg"
    v_canvas.convert("RGB").save(thumb_vert, format="JPEG", quality=92, optimize=True)
    return thumb_16x9, thumb_vert


def format_description(
    lang: str,
    summary_lines: list[str],
    chapters: list[tuple[str, str]],
    audio_note: str,
    hashtags: list[str],
) -> str:
    """Tạo phần mô tả sạch, đúng chính sách YouTube, tách riêng từng ngôn ngữ."""
    labels = {
        "vi": ("Nội dung:", "Mốc thời gian:", "Âm thanh & Phụ đề:"),
        "en": ("Synopsis:", "Chapters:", "Audio & Subtitles:"),
        "ja": ("あらすじ：", "チャプター：", "音声・字幕："),
        "ko": ("줄거리:", "챕터:", "오디오 및 자막:"),
    }[lang]
    parts: list[str] = [labels[0], *summary_lines, "", labels[1]]
    for ts, chap in chapters:
        parts.append(f"{ts} {chap}")
    parts.extend(["", f"{labels[2]} {audio_note}", "", " ".join(hashtags[:3])])
    return "\n".join(parts).strip() + "\n"


def save_custom_16x9_thumbnail(custom_src: Path, out_dir: Path) -> Path:
    """Chuẩn hóa thumbnail 16:9 thiết kế riêng (không dùng poster gốc có chữ Trung) về đúng 1280x720 (<2MB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(custom_src).convert("RGB")
    tw, th = 1280, 720
    scale = max(tw / im.width, th / im.height)
    rw, rh = int(round(im.width * scale)), int(round(im.height * scale))
    res = im.resize((rw, rh), Image.Resampling.LANCZOS)
    left = (rw - tw) // 2
    top = (rh - th) // 2
    out = res.crop((left, top, left + tw, top + th))
    thumb_16x9 = out_dir / "thumb_1280x720.jpg"
    out.save(thumb_16x9, format="JPEG", quality=94, optimize=True)
    return thumb_16x9


def main() -> int:
    parser = argparse.ArgumentParser(description="Đóng gói Thumbnail & Metadata YouTube 4 ngôn ngữ")
    parser.add_argument("--poster", default="", help="Đường dẫn tới poster.jpg gốc (dự phòng khi chưa có --custom-thumb)")
    parser.add_argument("--custom-thumb", default="", help="Ảnh thumbnail 16:9 thiết kế riêng (ưu tiên, không dùng poster gốc)")
    parser.add_argument("--part", default="P1", help="Nhãn tập/phần, ví dụ P1 hoặc P2")
    parser.add_argument("--config", required=True, help="File JSON chứa title/plot/chapters 4 ngôn ngữ")
    parser.add_argument("--out-dir", required=True, help="Thư mục xuất thumbnail & SEO txt/json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    thumb_vert = None
    existing_thumb = out_dir / "thumb_1280x720.jpg"
    if args.custom_thumb:
        thumb_16x9 = save_custom_16x9_thumbnail(Path(args.custom_thumb), out_dir)
    elif existing_thumb.is_file() and not args.poster:
        thumb_16x9 = existing_thumb
    elif args.poster:
        thumb_16x9, thumb_vert = build_thumbnails(Path(args.poster), args.part, out_dir)
    else:
        raise SystemExit("Cần truyền --custom-thumb hoặc --poster")

    localizations: dict[str, dict[str, str]] = {}
    for lang in ("vi", "en", "ja", "ko"):
        lc = cfg["languages"][lang]
        desc = format_description(
            lang=lang,
            summary_lines=lc["summary_lines"],
            chapters=[(c["time"], c["title"]) for c in lc["chapters"]],
            audio_note=lc["audio_note"],
            hashtags=lc["hashtags"],
        )
        localizations[lang] = {
            "title": lc["title"],
            "description": desc,
        }
        txt_out = out_dir / f"youtube_{lang}.txt"
        txt_out.write_text(
            f"TITLE ({len(lc['title'])} chars):\n{lc['title']}\n\nDESCRIPTION:\n{desc}",
            encoding="utf-8",
        )

    thumbs_meta = {"youtube_16x9": str(thumb_16x9)}
    if thumb_vert is not None:
        thumbs_meta["poster_vertical"] = str(thumb_vert)

    payload = {
        "part": args.part,
        "defaultLanguage": cfg.get("defaultLanguage", "vi"),
        "defaultAudioLanguage": cfg.get("defaultAudioLanguage", "vi"),
        "privacyStatus": cfg.get("privacyStatus", "private"),
        "categoryId": cfg.get("categoryId", "24"),
        "thumbnails": thumbs_meta,
        "snippet": {
            "title": localizations[cfg.get("defaultLanguage", "vi")]["title"],
            "description": localizations[cfg.get("defaultLanguage", "vi")]["description"],
            "tags": cfg.get("tags", []),
        },
        "localizations": localizations,
    }
    meta_path = out_dir / "metadata_4langs.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"THUMB_16X9={thumb_16x9} ({thumb_16x9.stat().st_size / 1024:.1f} KB)")
    if thumb_vert is not None:
        print(f"THUMB_POSTER={thumb_vert} ({thumb_vert.stat().st_size / 1024:.1f} KB)")
    print(f"METADATA_JSON={meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

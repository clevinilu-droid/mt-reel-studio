"""
Master Film Library Scanner & Exporter for MT Reel.
Scans Download/*/_SERIES.json + live disk folders (P1/P1_END/_END), output/, and _engine/work/,
prints the terminal dashboard, and generates:
  - DANH_SACH_PHIM.csv  (UTF-8 BOM, opens directly in Excel / Google Sheets)
  - DANH_SACH_PHIM.md   (Readable Markdown master catalog in repo root)

Usage:
  python _engine/scripts/scan_series.py
"""
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOWNLOAD_DIR = ROOT / "Download"
OUTPUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "_engine" / "work"

CSV_PATH = ROOT / "DANH_SACH_PHIM.csv"
MD_PATH = ROOT / "DANH_SACH_PHIM.md"

CSV_HEADERS = [
    "Ma_Series",
    "Ma_Phan",
    "Phan",
    "Ten_Goc_TQ",
    "Ten_Tieng_Viet",
    "The_Loai",
    "Nguon_Tai",
    "ID_Nguon",
    "Link_Nguon",
    "So_Tap",
    "Thoi_Luong_Phut",
    "Trang_Thai_Nguon_TQ",
    "Tien_Do_Ky_Thuat",
    "Trang_Thai_Folder",
    "Thu_Muc_Nguon",
    "Thanh_Pham_Output",
    "Glossary_Chung",
    "Ngon_Ngu",
    "Ngay_Tai",
    "Ngay_Up_YT",
    "Tieu_De_YT_Du_Kien",
    "Link_YouTube",
    "Nhu_Cau_Viewer",
    "Ghi_Chu",
]


def detect_live_part_status(sdir: Path, part_key: str, part_meta: dict) -> tuple[str, str, str]:
    """Return (folder_state, actual_rel_path, detected_output_str)."""
    p_end = sdir / f"{part_key}_END"
    p_cur = sdir / part_key
    work_name = part_meta.get("work_name", "")

    if p_end.exists() and p_end.is_dir():
        folder_state = f"{part_key}_END"
        rel_src = f"Download/{sdir.name}/{part_key}_END"
    elif p_cur.exists() and p_cur.is_dir():
        mp4s = list(p_cur.glob("*.mp4"))
        folder_state = f"{part_key} (co {len(mp4s)} mp4)" if mp4s else f"{part_key} (trong)"
        rel_src = f"Download/{sdir.name}/{part_key}"
    else:
        folder_state = "CHUA_TAI"
        rel_src = f"Download/{sdir.name}/{part_key}"

    out_str = "Chưa xuất"
    # Check nested output/<Series>/<P1|P1_END>/ first, then legacy flat output/<work_name>/
    nested_candidates = [OUTPUT_DIR / sdir.name / part_key, OUTPUT_DIR / sdir.name / f"{part_key}_END"]
    found_mp4 = None
    for ndir in nested_candidates:
        if ndir.is_dir():
            mp4s = sorted(f for f in ndir.glob("*.mp4") if f.stat().st_size > 1_000_000)
            if mp4s:
                found_mp4 = mp4s[0]
                break
    if found_mp4 is None and work_name:
        cand = OUTPUT_DIR / work_name / f"{work_name}.mp4"
        if cand.exists():
            found_mp4 = cand

    if found_mp4 is not None:
        mb = found_mp4.stat().st_size / (1024 * 1024)
        out_str = f"{found_mp4.relative_to(ROOT).as_posix()} ({mb:.1f} MB)"
    elif work_name and (WORK_DIR / work_name).exists():
        out_str = f"_engine/work/{work_name}/ (đang làm dở)"

    return folder_state, rel_src, out_str


def build_catalog():
    if not DOWNLOAD_DIR.exists():
        raise FileNotFoundError(f"Missing Download dir: {DOWNLOAD_DIR}")

    series_dirs = sorted(p for p in DOWNLOAD_DIR.iterdir() if p.is_dir())
    series_rows = []
    part_rows = []

    for sdir in series_dirs:
        meta_path = sdir / "_SERIES.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            # Fallback if user drops a new movie folder without _SERIES.json yet
            meta = {
                "series_id": sdir.name.split("_")[0],
                "title_orig": sdir.name,
                "title_vi": "(Chưa khai báo _SERIES.json)",
                "platform": "",
                "source_id": "",
                "known_parts_at_source": 1,
                "series_ended_at_source": False,
                "series_completed_all": sdir.name.upper().endswith("_END"),
                "parts": {},
            }

        sid = meta.get("series_id", "")
        title_orig = meta.get("title_orig", sdir.name)
        title_vi = meta.get("title_vi", "")
        genre = meta.get("genre", "")
        platform = meta.get("platform", "")
        source_id = meta.get("source_id", "")
        source_url = meta.get("source_url", "")
        known_parts = int(meta.get("known_parts_at_source", 1) or 1)
        src_ended = bool(meta.get("series_ended_at_source", False))
        all_end = sdir.name.upper().endswith("_END") or bool(meta.get("series_completed_all", False))
        glossary = meta.get("shared_glossary", "")

        sub_parts = sorted(p.name for p in sdir.iterdir() if p.is_dir() and p.name.upper().startswith("P"))
        ended_parts = [n for n in sub_parts if n.upper().endswith("_END")]
        active_parts = [n for n in sub_parts if not n.upper().endswith("_END")]

        if all_end:
            src_status = f"Đã END trọn bộ ({known_parts} phần)"
            next_action = "ĐÃ END TRỌN BỘ"
        elif src_ended:
            src_status = f"Nguồn TQ đã hết ({known_parts} phần)"
            next_action = f"Làm tiếp {', '.join(active_parts)}" if active_parts else f"Còn phần chờ tải (tổng {known_parts} phần)"
        else:
            src_status = f"Còn ra tiếp (hiện có {known_parts} phần)"
            next_action = f"Đang/Chờ làm {', '.join(active_parts)}" if active_parts else "Theo dõi phần mới bên TQ"

        series_rows.append({
            "sid": sid,
            "folder": sdir.name,
            "title_orig": title_orig,
            "title_vi": title_vi,
            "platform": platform,
            "source_id": source_id,
            "genre": genre,
            "known_parts": known_parts,
            "ended_parts": len(ended_parts),
            "sub_parts": ", ".join(sub_parts) if sub_parts else "(chưa có)",
            "src_status": src_status,
            "next_action": next_action,
            "glossary": glossary,
            "synopsis_vi": meta.get("synopsis_vi", ""),
            "main_characters": meta.get("main_characters", ""),
        })

        parts_dict = meta.get("parts") or {}
        if not parts_dict and sub_parts:
            for sp in sub_parts:
                pkey = sp.replace("_END", "").replace("_end", "")
                parts_dict[pkey] = {"folder": sp}

        for pkey, pmeta in sorted(parts_dict.items()):
            folder_state, rel_src, out_str = detect_live_part_status(sdir, pkey, pmeta)
            part_rows.append({
                "Ma_Series": sid,
                "Ma_Phan": f"{sid}-{pkey}",
                "Phan": pkey,
                "Ten_Goc_TQ": pmeta.get("title_orig_part") or title_orig,
                "Ten_Tieng_Viet": pmeta.get("title_vi_part") or f"{title_vi} — {pkey}",
                "The_Loai": genre,
                "Nguon_Tai": platform,
                "ID_Nguon": source_id,
                "Link_Nguon": source_url,
                "So_Tap": pmeta.get("episodes", ""),
                "Thoi_Luong_Phut": pmeta.get("duration_min", ""),
                "Trang_Thai_Nguon_TQ": src_status,
                "Tien_Do_Ky_Thuat": pmeta.get("pipeline_stage", ""),
                "Trang_Thai_Folder": folder_state,
                "Thu_Muc_Nguon": rel_src,
                "Thanh_Pham_Output": out_str,
                "Glossary_Chung": glossary,
                "Ngon_Ngu": pmeta.get("languages", ""),
                "Ngay_Tai": pmeta.get("downloaded_at", ""),
                "Ngay_Up_YT": pmeta.get("uploaded_at", ""),
                "Tieu_De_YT_Du_Kien": pmeta.get("youtube_title_vi", ""),
                "Link_YouTube": pmeta.get("youtube_url", ""),
                "Nhu_Cau_Viewer": pmeta.get("viewer_demand", ""),
                "Ghi_Chu": pmeta.get("notes", ""),
            })

    return series_rows, part_rows


def export_csv(part_rows: list[dict]):
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in part_rows:
            writer.writerow(row)


def export_markdown(series_rows: list[dict], part_rows: list[dict]):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 🎬 Danh Sách Quản Lý Toàn Bộ Kho Phim — MT Reel",
        "",
        f"> Tự động tổng hợp lúc `{now_str}` từ `Download/*/_SERIES.json` bằng `_engine/scripts/scan_series.py`.",
        "> Bản mở bằng Excel / Google Sheets: **[DANH_SACH_PHIM.csv](DANH_SACH_PHIM.csv)**.",
        "",
        "---",
        "",
        "## 1. Bảng Theo Dõi Series Tổng (Biết Bộ Nào Còn Ra Phần Mới)",
        "",
        "| Mã | Tên gốc TQ & Tên Tiếng Việt | Nguồn & ID | Thể loại | Thư mục hiện có | Trạng thái gốc TQ | Hành động tiếp theo |",
        "| :---: | :--- | :--- | :--- | :---: | :--- | :--- |",
    ]
    for s in series_rows:
        lines.append(
            f"| **{s['sid']}** | **{s['title_orig']}**<br>*{s['title_vi']}* | {s['platform']}<br>`{s['source_id']}` | {s['genre']} | `{s['sub_parts']}` | {s['src_status']} | **{s['next_action']}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Bảng Chi Tiết Từng Phần (`P1 / P2 / P3 / END`)",
        "",
        "| Mã Phần | Tên Phần (TQ & Việt) | Nguồn & ID | Tập / Thời lượng | Trạng thái Folder | Tiến độ kỹ thuật | Thành phẩm Output | Link YouTube | Nhu cầu Viewer |",
        "| :---: | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- |",
    ]
    for p in part_rows:
        dur = f"{p['Thoi_Luong_Phut']}p" if p["Thoi_Luong_Phut"] else ""
        ep_dur = f"{p['So_Tap']} ({dur})" if (p["So_Tap"] and dur) else (p["So_Tap"] or dur)
        yt = p["Link_YouTube"] or "Chưa up"
        lines.append(
            f"| **{p['Ma_Phan']}** | **{p['Ten_Goc_TQ']}**<br>*{p['Ten_Tieng_Viet']}* | {p['Nguon_Tai']}<br>`{p['ID_Nguon']}` | {ep_dur} | `{p['Trang_Thai_Folder']}` | {p['Tien_Do_Ky_Thuat']} | `{p['Thanh_Pham_Output']}` | {yt} | {p['Nhu_Cau_Viewer']} |"
        )

    lines += ["", "---", "", "## 3. Hồ Sơ Chi Tiết Từng Series", ""]
    for s in series_rows:
        lines += [
            f"### #{s['sid']} — {s['title_orig']} ({s['title_vi']})",
            f"- **Thư mục:** `Download/{s['folder']}/`",
            f"- **Nguồn tải:** {s['platform']} — ID: `{s['source_id']}`",
            f"- **Nhân vật chính:** {s['main_characters']}",
            f"- **Tóm tắt:** {s['synopsis_vi']}",
            f"- **Glossary dùng chung cho P1 → P2 → P3:** `{s['glossary']}`",
            "",
        ]

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    series_rows, part_rows = build_catalog()
    export_csv(part_rows)
    export_markdown(series_rows, part_rows)

    print("=" * 84)
    print(f"{'MA':<5} {'SERIES (TEN VIET)':<38} {'PHAN HIEN CO':<16} {'VIEC TIEP THEO'}")
    print("=" * 84)
    for s in series_rows:
        print(f"{s['sid']:<5} {s['title_vi'][:37]:<38} {s['sub_parts'][:15]:<16} {s['next_action']}")
    print("=" * 84)
    print(f"[OK] Da xuat: {CSV_PATH.name} ({len(part_rows)} dong) + {MD_PATH.name}")


if __name__ == "__main__":
    main()

"""Quản lý Checkpoint Atomic & Nhận diện vân tay video nguồn (Fingerprint).

Bảo đảm:
  - Mọi ghi đè `manifest.json` đều ghi ra file tạm rồi `os.replace()` (không hỏng file khi tắt ngang).
  - Tự nhận diện các thư mục `_engine/work/<Series>/<Part>/` đã chạy xong Bước 1 từ trước
    (kể cả bộ `001 / P1` vừa chạy) để không bao giờ bắt chạy lại khi kéo thả hoặc chọn lại file.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENGINE_DIR = ROOT_DIR / "_engine"
WORK_ROOT = ENGINE_DIR / "work"
DOWNLOAD_ROOT = ROOT_DIR / "Download"
OUTPUT_ROOT = ROOT_DIR / "output"

PIPELINE_STEPS = [
    ("s1", "Bước 1 — Chuẩn hóa CFR GPU + WAV + Audit gốc"),
    ("s2a", "Bước 2A — Nghe thoại & Gom người nói (FunASR)"),
    ("s2b", "Bước 2B — Lọc Keyframe & Đọc sub (PyAV + Qwen)"),
    ("s3", "Bước 3 — Đối chiếu Pinyin & Cứu hộ câu sót"),
    ("s4", "Bước 4 — Tách giọng vùng thoại (RoFormer) & 2 mẫu Clone"),
    ("s5", "Bước 5 — Phân vai (Qwen) & Dịch Việt/Ngoại ngữ (Gemma 4)"),
    ("s6", "Bước 6 — Lồng tiếng (VieNeu + Qwen3-TTS) & Render"),
]


def now_iso() -> str:
    """Trả về chuỗi thời gian địa phương dạng ISO ngắn gọn."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_json(path: Path, data: dict[str, Any]):
    """Ghi JSON an toàn qua file tạm cùng thư mục rồi đổi tên atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, path)


def read_json_safe(path: Path) -> dict[str, Any] | None:
    """Đọc file JSON an toàn (chấp nhận cả UTF-8 có hoặc không có BOM), trả về None nếu lỗi."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def compute_source_fingerprint(src_path: Path) -> dict[str, Any]:
    """Tính vân tay nhanh (tên, dung lượng, SHA256 64KB đầu + cuối) không phải băm cả file nhiều GB."""
    st = src_path.stat()
    size = st.st_size
    chunk = 65536
    h_head = hashlib.sha256()
    h_tail = hashlib.sha256()
    with src_path.open("rb") as fh:
        h_head.update(fh.read(min(chunk, size)))
        if size > chunk:
            fh.seek(max(0, size - chunk))
            h_tail.update(fh.read(chunk))
        else:
            h_tail = h_head
    return {
        "name": src_path.name,
        "size": size,
        "head_sha256": h_head.hexdigest()[:16],
        "tail_sha256": h_tail.hexdigest()[:16],
    }


def fingerprints_match(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Kiểm tra 2 vân tay file nguồn có trùng khớp tuyệt đối không."""
    if not a or not b:
        return False
    return (
        a.get("size") == b.get("size")
        and a.get("head_sha256") == b.get("head_sha256")
        and a.get("tail_sha256") == b.get("tail_sha256")
    )


def infer_work_rel(src_mp4: Path) -> str:
    """Tự suy ra `<Series>/<Part>` nếu video nằm trong `Download/`, ngược lại dùng tên file sạch."""
    try:
        rel_parts = src_mp4.resolve().relative_to(DOWNLOAD_ROOT.resolve()).parts
        if len(rel_parts) >= 3:
            return f"{rel_parts[0]}/{rel_parts[1]}"
    except ValueError:
        pass
    stem = src_mp4.stem.strip() or f"video_{int(time.time())}"
    return stem


def step1_artifacts_valid(target_dir: Path) -> bool:
    """Kiểm tra bộ sản phẩm Bước 1 (MP4 CFR + 2 WAV + audit JSON) đã đầy đủ hợp lệ chưa."""
    req = [
        (target_dir / "01_cfr_raw.mp4", 100_000),
        (target_dir / "01_orig_48k_stereo.wav", 100_000),
        (target_dir / "01_orig_16k_ai.wav", 20_000),
        (target_dir / "step1_audit.json", 50),
    ]
    return all(p.exists() and p.stat().st_size >= min_sz for p, min_sz in req)


def default_steps_state() -> dict[str, dict[str, Any]]:
    """Khởi tạo trạng thái mặc định cho 7 chặng (Bước 1 -> Bước 6)."""
    return {
        key: {
            "key": key,
            "label": label,
            "status": "pending",
            "progress_pct": 0.0,
            "detail": "",
            "elapsed_s": 0.0,
            "updated_at": "",
        }
        for key, label in PIPELINE_STEPS
    }


def make_job_id(work_rel: str) -> str:
    """Tạo ID ổn định cho từng thư mục work `<Series>/<Part>`."""
    norm = work_rel.replace("\\", "/").strip("/")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def load_or_create_manifest(
    work_dir: Path,
    work_rel: str,
    source_path: Path | None = None,
    mode: str = "test_5min",
    langs: list[str] | None = None,
    voice_mode: str = "clone_2",
    auto_sync_drive: bool = False,
) -> dict[str, Any]:
    """Đọc `manifest.json` nếu có, hoặc khởi tạo mới và tự động phát hiện checkpoint Bước 1 sẵn có."""
    work_dir.mkdir(parents=True, exist_ok=True)
    mpath = work_dir / "manifest.json"
    data = read_json_safe(mpath)
    norm_rel = work_rel.replace("\\", "/").strip("/")

    if data is None:
        data = {
            "job_id": make_job_id(norm_rel),
            "work_rel": norm_rel,
            "work_dir": str(work_dir),
            "source_path": str(source_path.resolve()) if source_path and source_path.exists() else "",
            "source_fingerprint": compute_source_fingerprint(source_path) if source_path and source_path.exists() else None,
            "mode": mode,
            "langs": langs or ["vi", "en", "ja", "ko", "es"],
            "voice_mode": voice_mode,
            "auto_sync_drive": auto_sync_drive,
            "status": "idle",
            "current_step": "",
            "steps": default_steps_state(),
            "step1_full_ready": False,
            "step1_test_ready": False,
            "audit_full": None,
            "audit_test": None,
            "drive_sync": {"status": "idle", "remote": "", "detail": "", "updated_at": ""},
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
    else:
        # Bổ sung step mới nếu schema mở rộng
        base_steps = default_steps_state()
        for k, v in base_steps.items():
            data.setdefault("steps", {}).setdefault(k, v)
        if source_path and source_path.exists():
            data["source_path"] = str(source_path.resolve())
            if not data.get("source_fingerprint"):
                data["source_fingerprint"] = compute_source_fingerprint(source_path)

    # Đồng bộ trạng thái thực tế của Bước 1 (cả bản Full và bản Test 5 phút)
    full_ok = step1_artifacts_valid(work_dir)
    test_ok = step1_artifacts_valid(work_dir / "_test_5min")
    data["step1_full_ready"] = full_ok
    data["step1_test_ready"] = test_ok

    if full_ok:
        audit_full = read_json_safe(work_dir / "step1_audit.json")
        data["audit_full"] = audit_full
        if audit_full:
            src_str = audit_full.get("source", "")
            if src_str and not data.get("source_path") and Path(src_str).exists():
                data["source_path"] = str(Path(src_str).resolve())
            if data.get("source_path") and not data.get("source_fingerprint") and Path(data["source_path"]).exists():
                data["source_fingerprint"] = compute_source_fingerprint(Path(data["source_path"]))
            t_s = audit_full.get("timings_s", {}).get("total_step1_s", 0.0)
            enc = audit_full.get("encoder_used", "")
            match_pct = audit_full.get("spotcheck", {}).get("timeline_match_pct")
            detail = f"Đã hoàn tất bản Full ({enc}, {t_s}s"
            if match_pct is not None:
                detail += f", khớp sub {match_pct}%"
            detail += ")"
            if data["steps"]["s1"]["status"] != "running":
                data["steps"]["s1"].update(
                    {
                        "status": "done",
                        "progress_pct": 100.0,
                        "detail": detail,
                        "elapsed_s": float(t_s or 0.0),
                        "updated_at": data["steps"]["s1"].get("updated_at") or now_iso(),
                    }
                )
            if data.get("status") == "idle":
                data["status"] = "step1_done"

    if test_ok:
        data["audit_test"] = read_json_safe(work_dir / "_test_5min" / "step1_audit.json")
        if not full_ok and data["steps"]["s1"]["status"] == "pending":
            t_s = (data["audit_test"] or {}).get("timings_s", {}).get("total_step1_s", 0.0)
            data["steps"]["s1"].update(
                {
                    "status": "done",
                    "progress_pct": 100.0,
                    "detail": f"Đã xong bản Test 5 phút ({t_s}s)",
                    "elapsed_s": float(t_s or 0.0),
                    "updated_at": now_iso(),
                }
            )

    # Tự đồng bộ trạng thái B2A -> B5 từ file kết quả thực tế trên ổ cứng
    check_dir = work_dir if (work_dir / "02a_audio_asr.json").exists() else (work_dir / "_test_5min")
    if check_dir.exists():
        d2a = read_json_safe(check_dir / "02a_audio_asr.json")
        if d2a and data["steps"]["s2a"]["status"] != "running":
            data["steps"]["s2a"].update({"status": "done", "progress_pct": 100.0, "detail": f"{d2a.get('engine','')}: {d2a.get('count',0)} câu thoại ({d2a.get('elapsed_s',0)}s)"})
        d2b = read_json_safe(check_dir / "02b_ocr_raw.json")
        if d2b and data["steps"]["s2b"]["status"] != "running":
            data["steps"]["s2b"].update({"status": "done", "progress_pct": 100.0, "detail": f"{d2b.get('engine','')}: {d2b.get('segments_count',0)} dòng sub từ {d2b.get('keyframes_count',0)} ảnh ({d2b.get('elapsed_s',0)}s)"})
        d3 = read_json_safe(check_dir / "03_audit_flags.json")
        if d3 and data["steps"]["s3"]["status"] != "running":
            data["steps"]["s3"].update({"status": "done", "progress_pct": 100.0, "detail": f"{d3.get('total_turns',0)} câu chuẩn (khớp {d3.get('matched_count',0)}, cứu {d3.get('ocr_rescued_count',0)}, lọc {d3.get('screen_captions_filtered',0)} chữ màn hình)"})
            if data.get("status") in ("idle", "step1_done"):
                data["status"] = "s3_done"
        d4 = read_json_safe(check_dir / "04_f0_speakers.json")
        if d4 and data["steps"]["s4"]["status"] != "running":
            data["steps"]["s4"].update({"status": "done", "progress_pct": 100.0, "detail": f"Nam {d4.get('ref_male',{}).get('duration_s',0)}s | Nữ {d4.get('ref_female',{}).get('duration_s',0)}s ({d4.get('elapsed_s',0)}s)"})
        d5 = read_json_safe(check_dir / "05_translate_meta.json")
        if d5 and data["steps"]["s5"]["status"] != "running":
            data["steps"]["s5"].update({"status": "done", "progress_pct": 100.0, "detail": f"{','.join(d5.get('langs',[])).upper()} ({d5.get('turns',0)} câu, {d5.get('elapsed_s',0)}s)"})

    data["updated_at"] = now_iso()
    atomic_write_json(mpath, data)
    return data


def save_manifest(manifest: dict[str, Any]):
    """Lưu lại manifest xuống đĩa theo cơ chế atomic."""
    work_dir = Path(manifest["work_dir"])
    manifest["updated_at"] = now_iso()
    atomic_write_json(work_dir / "manifest.json", manifest)


def scan_existing_work_manifests() -> list[dict[str, Any]]:
    """Quét toàn bộ thư mục `_engine/work/` để khôi phục danh sách job & tự nhận diện Bước 1 đã xong."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    found_dirs: list[Path] = []
    for p in sorted(WORK_ROOT.rglob("*")):
        if not p.is_dir():
            continue
        if p.name.startswith("_") or p.name in ("checkpoints", "tracks", "keyframes", "tts_cache"):
            continue
        has_marker = (
            (p / "manifest.json").exists()
            or (p / "step1_audit.json").exists()
            or (p / "_test_5min" / "step1_audit.json").exists()
        )
        if has_marker:
            found_dirs.append(p)

    results = []
    for wd in found_dirs:
        rel = wd.resolve().relative_to(WORK_ROOT.resolve()).as_posix()
        m = load_or_create_manifest(wd, rel)
        results.append(m)
    return results


def find_matching_manifest_by_source(src_path: Path) -> dict[str, Any] | None:
    """Tìm xem file video vừa kéo thả/chọn đã từng có checkpoint hợp lệ ở `_engine/work/` chưa."""
    if not src_path.exists():
        return None
    fp = compute_source_fingerprint(src_path)
    src_resolved = str(src_path.resolve()).lower()
    for m in scan_existing_work_manifests():
        if m.get("source_path", "").lower() == src_resolved:
            return m
        if fingerprints_match(m.get("source_fingerprint"), fp):
            return m
    return None

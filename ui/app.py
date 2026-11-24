"""
MT-Reel Studio UI v3 — Giao diện điều khiển sạch & khoa học cho quy trình 5 bước + Vast.ai GPU Picker.
Chạy: .venv\Scripts\python.exe -m uvicorn ui.app:app --host 127.0.0.1 --port 8787
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
PROJECTS_DIR = ROOT_DIR / "projects"
DOWNLOAD_DIR = ROOT_DIR / "Download"
VAST_EXE = ROOT_DIR / ".venv" / "Scripts" / "vastai.exe"
PY_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

GHCR_IMAGE = "ghcr.io/clevinilu-droid/mt-reel-vast:latest"

load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="MT-Reel Studio UI v3", version="3.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _vast_key() -> str:
    load_dotenv(ROOT_DIR / ".env", override=True)
    key = (os.getenv("VAST_API_KEY") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Chưa cấu hình VAST_API_KEY trong file .env")
    return key


def _run_vast(args: list[str], timeout: int = 25) -> str:
    exe = str(VAST_EXE) if VAST_EXE.exists() else "vastai"
    key = _vast_key()
    cmd = [exe, *args, "--api-key", key]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "Vast CLI error").strip()
        raise HTTPException(status_code=502, detail=msg[:400])
    return res.stdout


def _parse_vast_table(raw_text: str) -> list[dict[str, Any]]:
    """Parse vastai CLI multi-section aligned table into clean JSON rows."""
    blocks = re.split(r"\r?\n\s*\r?\n", raw_text.strip())
    rows_by_idx: dict[str, dict[str, Any]] = {}
    for blk in blocks:
        lines = [ln.rstrip() for ln in blk.splitlines() if ln.strip()]
        if len(lines) < 2 or not lines[0].lstrip().startswith("#"):
            continue
        headers = re.split(r"\s{2,}", lines[0].strip())
        for ln in lines[1:]:
            parts = re.split(r"\s{2,}", ln.strip())
            if not parts or not parts[0].isdigit():
                continue
            row_id = parts[0]
            rec = rows_by_idx.setdefault(row_id, {})
            for h, v in zip(headers, parts):
                rec[h] = v
    # Sort by integer #
    out = [rows_by_idx[k] for k in sorted(rows_by_idx.keys(), key=lambda x: int(x))]
    return out


class RentRequest(BaseModel):
    offer_id: int
    disk_gb: int = 45
    label: str = "mt-reel-step3"


class ActionRunRequest(BaseModel):
    series: str
    part: str = "P1"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/vast/status")
def vast_status():
    """Return account balance/email (no API key exposed) and active instances."""
    user_raw = _run_vast(["show", "user"])
    credit_m = re.search(r"\b\d+\.\d+\b\s+\S+@\S+", user_raw)
    credit = None
    email = None
    for line in user_raw.splitlines():
        m = re.search(r"^\s*1\s+([\d.]+)\s+(\S+@\S+)", line)
        if m:
            credit = float(m.group(1))
            email = m.group(2)
            break

    inst_raw = _run_vast(["show", "instances"])
    instances = []
    if "No instances found" not in inst_raw:
        instances = _parse_vast_table(inst_raw)

    return {
        "credit": credit,
        "email": email,
        "image": GHCR_IMAGE,
        "instances": instances,
        "instances_raw": inst_raw.strip(),
    }


@app.get("/api/vast/offers")
def vast_offers(
    preset: str = Query("balanced"),
    min_vram: int = Query(12),
    max_dph: float = Query(0.25),
    min_disk: int = Query(45),
    min_down: int = Query(200),
    exclude_cn: bool = Query(True),
    limit: int = Query(15),
):
    parts = [
        f"gpu_ram>={min_vram}",
        "num_gpus=1",
        "compute_cap>=860",
        "verified=True",
        "reliability>0.97",
        "cuda_max_good>=12.8",
        "cpu_cores_effective>=4",
        "cpu_ram>=16",
        "pcie_bw>=5",
        f"inet_down>={min_down}",
        "inet_up>=100",
        f"disk_space>={min_disk}",
        "direct_port_count>=1",
        f"dph<={max_dph}",
        "rented=False",
    ]
    if preset == "fast":
        parts.append("dlperf>=40")
    else:
        parts.append("dlperf>=22")

    if preset == "asia":
        parts.append("geolocation in [VN,TH,SG,JP,TW,KR,MY,HK]")
    elif exclude_cn:
        parts.append("geolocation notin [CN]")

    query_str = " ".join(parts)
    sort_by = "dlperf-" if preset == "fast" else "dph_total"
    raw = _run_vast(["search", "offers", query_str, "-o", sort_by, "--limit", str(limit)])
    offers = _parse_vast_table(raw)
    return {"query": query_str, "sort": sort_by, "count": len(offers), "offers": offers}


@app.post("/api/vast/rent")
def vast_rent(req: RentRequest):
    """Rent a Vast.ai GPU instance preloaded with our GHCR image & SSH direct."""
    out = _run_vast(
        [
            "create",
            "instance",
            str(req.offer_id),
            "--image",
            GHCR_IMAGE,
            "--disk",
            str(req.disk_gb),
            "--env",
            "-p 7860:7860 -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
            "--ssh",
            "--direct",
            "--label",
            req.label,
        ],
        timeout=40,
    )
    return {"ok": True, "message": out.strip()}


@app.post("/api/vast/destroy/{instance_id}")
def vast_destroy(instance_id: int):
    out = _run_vast(["destroy", "instance", str(instance_id)], timeout=30)
    return {"ok": True, "message": out.strip()}


@app.get("/api/projects")
def list_projects():
    """Inspect projects/ and Download/ to summarize exact step readiness."""
    series_list = []
    if PROJECTS_DIR.exists():
        for s_dir in sorted(PROJECTS_DIR.iterdir()):
            if not s_dir.is_dir():
                continue
            parts_info = []
            for p_dir in sorted(s_dir.iterdir()):
                if not p_dir.is_dir():
                    continue
                b2 = p_dir / "01_buoc2_nhan_vat_va_khung_blur"
                b3 = p_dir / "02_buoc3_hardsub_va_am_thanh"
                b4 = p_dir / "03_buoc4_ban_dich_da_ngon_ngu"
                b5 = p_dir / "04_buoc5_thanh_pham_youtube"
                parts_info.append({
                    "part": p_dir.name,
                    "step2_done": (b2 / "blur_config.json").exists(),
                    "has_blur_preview": (b2 / "blur_truoc_sau.jpg").exists(),
                    "blur_preview_url": f"/api/projects/{s_dir.name}/{p_dir.name}/blur-preview" if (b2 / "blur_truoc_sau.jpg").exists() else None,
                    "has_test_10m": (b3 / "test_10m_raw.mp4").exists(),
                    "has_zh_raw": (b3 / "zh_hardsub_raw.srt").exists(),
                    "has_vocals": (b3 / "vocals_48k_mono.flac").exists(),
                    "has_instr": (b3 / "instrumental_48k.flac").exists(),
                    "has_zh_synced": (b3 / "zh_master_synced.srt").exists(),
                    "has_vi_srt": (b4 / "vi.srt").exists(),
                    "outputs": [f.name for f in b5.glob("*")] if b5.exists() else [],
                })
            series_list.append({
                "series": s_dir.name,
                "has_profile": (s_dir / "HO_SO_NHAN_VAT.md").exists(),
                "has_glossary": (s_dir / "character_glossary.json").exists(),
                "parts": parts_info,
            })

    downloads = []
    if DOWNLOAD_DIR.exists():
        for f in sorted(DOWNLOAD_DIR.rglob("*.mp4")):
            downloads.append({
                "rel_path": str(f.relative_to(DOWNLOAD_DIR)),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })

    return {"projects": series_list, "downloads": downloads}


@app.get("/api/projects/{series}/{part}/blur-preview")
def get_blur_preview(series: str, part: str):
    p = PROJECTS_DIR / series / part / "01_buoc2_nhan_vat_va_khung_blur" / "blur_truoc_sau.jpg"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Chưa có ảnh xem trước làm mờ")
    return FileResponse(str(p))


@app.get("/api/projects/{series}/profile")
def get_character_profile(series: str):
    p = PROJECTS_DIR / series / "HO_SO_NHAN_VAT.md"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Chưa có HO_SO_NHAN_VAT.md")
    return {"content": p.read_text(encoding="utf-8")}


@app.get("/api/workflow-doc")
def get_workflow_doc():
    p = ROOT_DIR / "QUY_TRINH_CHUAN.md"
    return {"content": p.read_text(encoding="utf-8") if p.exists() else ""}


@app.get("/api/drive/list")
def list_drive():
    try:
        res = subprocess.run(
            ["rclone", "lsl", "gdrive:MT-Video"],
            capture_output=True, text=True, timeout=20,
        )
        items = []
        for ln in res.stdout.splitlines():
            m = re.match(r"^\s*(\d+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\.\d+\s+(.+)$", ln)
            if m:
                items.append({
                    "size_mb": round(int(m.group(1)) / (1024 * 1024), 2),
                    "modified": m.group(2),
                    "path": m.group(3),
                })
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""Backend FastAPI + WebSocket cho Web UI MT Reel v2.

Tính năng:
  - Upload chia chunk (10MB/chunk) mượt mà cho video 2–5 GB (hỗ trợ cả Kéo-Thả & nút Browse).
  - Chọn trực tiếp file sẵn có trong `Download/` (không cần copy lại).
  - Phát sóng tiến độ từng bước & thông số CPU/GPU/VRAM/Disk trực tiếp qua WebSocket mỗi 0.5s.
  - Hỗ trợ đẩy gói Cách B lên Google Drive chạy nền song song (`rclone`).
"""
from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from _engine.core.checkpoint import DOWNLOAD_ROOT
from _engine.core.drive_sync import list_drive_projects
from _engine.core.job_manager import JobManager
from _engine.core.vast_api import (
    destroy_instance,
    fetch_instance_logs,
    get_vast_api_key,
    list_instances,
    rent_instance,
    search_best_offers,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INBOX_DIR = DOWNLOAD_ROOT / "_uploaded"

app = FastAPI(title="MT Reel Studio v2", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

manager = JobManager()
_upload_sessions: dict[str, dict[str, Any]] = {}


def _forward_remote_sync(target_url: str, method: str, path_qs: str, headers_in: dict[str, str], body: bytes, timeout: int) -> tuple[int, str, bytes]:
    full_url = f"{target_url.rstrip('/')}{path_qs}"
    fwd_headers: dict[str, str] = {"User-Agent": "MT-Reel-Proxy/2.0"}
    if "content-type" in headers_in:
        fwd_headers["Content-Type"] = headers_in["content-type"]
    req = urllib.request.Request(full_url, data=body if body else None, headers=fwd_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get("Content-Type", "application/json"), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()


@app.middleware("http")
async def proxy_remote_target(request: Request, call_next):
    target = (request.headers.get("x-remote-target") or "").strip()
    path = request.url.path
    if target and path.startswith("/api/") and not path.startswith("/api/vast/"):
        parsed = urllib.parse.urlparse(target)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in ("http", "https") or not host or host in ("127.0.0.1", "localhost", "0.0.0.0"):
            raise HTTPException(status_code=400, detail="Địa chỉ máy Vast điều khiển từ xa không hợp lệ.")
        path_qs = f"{path}?{request.url.query}" if request.url.query else path
        body = await request.body()
        timeout = 300 if (path == "/api/jobs/from-drive" or path.startswith("/api/upload/") or path.endswith("/push-output")) else 25
        try:
            status, ctype, raw = await asyncio.to_thread(
                _forward_remote_sync,
                f"{parsed.scheme}://{parsed.netloc}",
                request.method,
                path_qs,
                {k.lower(): v for k, v in request.headers.items()},
                body,
                timeout,
            )
            return Response(content=raw, status_code=status, media_type=ctype)
        except Exception as exc:
            return Response(
                content=f'{{"detail":"Không kết nối được tới máy RTX thuê: {exc}"}}',
                status_code=502,
                media_type="application/json",
            )
    return await call_next(request)


class FromPathReq(BaseModel):
    source_path: str
    work_rel: str | None = None
    mode: str = "test_5min"
    langs: list[str] = ["vi", "en", "ja", "ko", "es"]
    voice_mode: str = "single_female"
    auto_sync_drive: bool = False
    auto_start: bool = True
    force_step1: bool = False


class FromDriveReq(BaseModel):
    work_rel: str
    mode: str = "full"
    langs: list[str] = ["vi", "en", "ja", "ko", "es"]
    voice_mode: str = "single_female"
    auto_start: bool = True


class VastRentReq(BaseModel):
    offer_id: int
    image: str = ""
    template_hash_id: str = ""
    disk_gb: int = 120
    include_rclone: bool = True


class StartJobReq(BaseModel):
    mode: str | None = None
    force_step1: bool = False
    target_stage: str = "auto"
    voice_mode: str | None = None
    langs: list[str] | None = None


class UploadInitReq(BaseModel):
    filename: str
    size: int


class UploadFinishReq(BaseModel):
    upload_id: str
    work_rel: str | None = None
    mode: str = "test_5min"
    langs: list[str] = ["vi", "en", "ja", "ko", "es"]
    voice_mode: str = "single_female"
    auto_sync_drive: bool = False
    auto_start: bool = True
    force_step1: bool = False


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state():
    return manager.get_snapshot()


@app.get("/api/diag")
def get_diag():
    """Chẩn đoán phần cứng tức thì (h264_nvenc + ONNX CUDA) không cần mở SSH."""
    from _engine.pipeline.s1_prepare import diagnose_gpu_encoder

    ort_info: dict[str, Any] = {}
    try:
        from _engine.core.vram_guard import ensure_onnx_cuda_libs

        loaded = ensure_onnx_cuda_libs()
        import onnxruntime as ort

        ort_info = {"providers": ort.get_available_providers(), "preloaded": loaded[:8]}
    except Exception as exc:
        ort_info = {"error": str(exc)}
    return {"nvenc": diagnose_gpu_encoder(), "onnx": ort_info}


@app.get("/api/downloads")
def get_downloads():
    return {"items": manager.list_download_videos()}


@app.post("/api/upload/init")
def upload_init(req: UploadInitReq):
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(req.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")
    uid = uuid.uuid4().hex[:10]
    tmp_path = INBOX_DIR / f"{uid}_{safe_name}.part"
    final_path = INBOX_DIR / safe_name
    if final_path.exists() and final_path.stat().st_size == req.size:
        # Đã có file giống kích thước trong _uploaded -> đánh dấu dùng lại luôn khỏi tốn dung lượng
        _upload_sessions[uid] = {
            "tmp_path": str(final_path),
            "final_path": str(final_path),
            "size": req.size,
            "skip_write": True,
            "created_at": time.time(),
        }
        return {"upload_id": uid, "already_exists": True}

    tmp_path.write_bytes(b"")
    _upload_sessions[uid] = {
        "tmp_path": str(tmp_path),
        "final_path": str(INBOX_DIR / f"{uid}_{safe_name}") if final_path.exists() else str(final_path),
        "size": req.size,
        "skip_write": False,
        "created_at": time.time(),
    }
    return {"upload_id": uid, "already_exists": False}


@app.post("/api/upload/chunk")
async def upload_chunk(upload_id: str, request: Request):
    sess = _upload_sessions.get(upload_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Phiên upload không tồn tại")
    if sess.get("skip_write"):
        return {"ok": True}
    body = await request.body()
    with open(sess["tmp_path"], "ab") as fh:
        fh.write(body)
    return {"ok": True, "received": len(body)}


@app.post("/api/upload/finish")
def upload_finish(req: UploadFinishReq):
    sess = _upload_sessions.pop(req.upload_id, None)
    if not sess:
        raise HTTPException(status_code=404, detail="Phiên upload không tồn tại")
    tmp_path = Path(sess["tmp_path"])
    final_path = Path(sess["final_path"])
    if not sess.get("skip_write"):
        os.replace(tmp_path, final_path)

    manifest = manager.add_or_attach_job(
        source_path=final_path,
        work_rel=req.work_rel or None,
        mode=req.mode,
        langs=req.langs,
        voice_mode=req.voice_mode,
        auto_sync_drive=req.auto_sync_drive,
        auto_start=req.auto_start,
        force_step1=req.force_step1,
    )
    return {"job": manifest}


@app.post("/api/jobs/from-path")
def create_job_from_path(req: FromPathReq):
    try:
        manifest = manager.add_or_attach_job(
            source_path=Path(req.source_path),
            work_rel=req.work_rel or None,
            mode=req.mode,
            langs=req.langs,
            voice_mode=req.voice_mode,
            auto_sync_drive=req.auto_sync_drive,
            auto_start=req.auto_start,
            force_step1=req.force_step1,
        )
        return {"job": manifest}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, req: StartJobReq):
    try:
        manager.enqueue_job(
            job_id,
            mode=req.mode,
            force_step1=req.force_step1,
            target_stage=req.target_stage,
            voice_mode=req.voice_mode,
            langs=req.langs,
        )
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    manager.pause_job(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/reset")
def reset_job(job_id: str, mode: str | None = None, from_step: str = "s2a"):
    try:
        return manager.reset_downstream_cache(job_id, mode=mode, from_step=from_step)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/jobs/{job_id}/push-drive")
def push_drive(job_id: str):
    try:
        manager.trigger_drive_push(job_id)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/jobs/{job_id}/open-folder")
def open_folder(job_id: str):
    m = manager.jobs.get(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    wd = Path(m["work_dir"])
    wd.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        subprocess.Popen(["explorer", str(wd)])
        return {"ok": True, "path": str(wd)}
    return {"ok": False, "path": str(wd), "detail": "Chỉ hỗ trợ mở thư mục trực tiếp trên Windows"}


@app.get("/api/drive/projects")
def drive_projects():
    try:
        return {"items": list_drive_projects()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/jobs/from-drive")
def create_job_from_drive(req: FromDriveReq):
    try:
        manifest = manager.pull_drive_job(
            work_rel=req.work_rel,
            mode=req.mode,
            langs=req.langs,
            voice_mode=req.voice_mode,
            auto_start=req.auto_start,
        )
        return {"job": manifest}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/jobs/{job_id}/push-output")
def push_output_drive(job_id: str):
    try:
        return manager.push_output_drive(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/vast/offers")
def vast_offers(gpu: str = "ANY", min_inet: int = 1000, max_dph: float = 0.65, sort_by: str = "price"):
    if not get_vast_api_key():
        raise HTTPException(status_code=400, detail="Chưa cấu hình VAST_API_KEY trong file .env trên máy.")
    try:
        offers = search_best_offers(gpu_filter=gpu, min_inet_mbps=min_inet, max_dph=max_dph, sort_by=sort_by, limit=25)
        return {"offers": offers}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn Vast.ai: {exc}")


@app.get("/api/vast/instances")
def vast_instances():
    if not get_vast_api_key():
        return {"configured": False, "instances": []}
    try:
        return {"configured": True, "instances": list_instances()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi lấy danh sách máy Vast.ai: {exc}")


@app.post("/api/vast/rent")
def vast_rent(req: VastRentReq):
    if not get_vast_api_key():
        raise HTTPException(status_code=400, detail="Chưa cấu hình VAST_API_KEY trong file .env trên máy.")
    try:
        return rent_instance(
            offer_id=req.offer_id,
            image=req.image,
            template_hash_id=req.template_hash_id,
            disk_gb=req.disk_gb,
            include_rclone=req.include_rclone,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Thuê máy Vast.ai thất bại: {exc}")


@app.get("/api/vast/instances/{instance_id}/logs")
def vast_instance_logs(instance_id: int, tail: int = 120):
    if not get_vast_api_key():
        raise HTTPException(status_code=400, detail="Chưa cấu hình VAST_API_KEY.")
    try:
        return fetch_instance_logs(instance_id, tail=tail)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi lấy log máy #{instance_id}: {exc}")


@app.delete("/api/vast/instances/{instance_id}")
def vast_destroy(instance_id: int):
    if not get_vast_api_key():
        raise HTTPException(status_code=400, detail="Chưa cấu hình VAST_API_KEY.")
    try:
        return destroy_instance(instance_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Hủy máy Vast.ai thất bại: {exc}")


@app.websocket("/ws")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    q = manager.register_ws(loop)
    try:
        await websocket.send_json(manager.get_snapshot())
        while True:
            try:
                await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            await websocket.send_json(manager.get_snapshot())
    except WebSocketDisconnect:
        pass
    finally:
        manager.unregister_ws(loop, q)

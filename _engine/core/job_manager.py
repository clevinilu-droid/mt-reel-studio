"""Bộ điều phối hàng đợi (JobManager) chạy cuốn chiếu & phát sóng tiến độ Live cho Web UI.

- Tách riêng 2 worker song song:
    1. `pipeline_worker`: xử lý hàng đợi các bước GPU/Video theo thứ tự tuần tự an toàn VRAM.
    2. `drive_worker`: đẩy gói Cách B lên Google Drive chạy nền song song, không làm nghẽn pipeline.
- Tự động nạp lại toàn bộ các bộ đã làm trong `_engine/work/` khi khởi động.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from _engine.core.checkpoint import (
    DOWNLOAD_ROOT,
    OUTPUT_ROOT,
    ROOT_DIR,
    WORK_ROOT,
    find_matching_manifest_by_source,
    infer_work_rel,
    load_or_create_manifest,
    now_iso,
    save_manifest,
    scan_existing_work_manifests,
)
from _engine.core.drive_sync import (
    check_gdrive_remote,
    pull_from_drive,
    push_method_b_to_drive,
    push_output_to_drive,
)
from _engine.core.telemetry import HardwareTelemetry
from _engine.core.vast_api import get_vast_api_key
from _engine.core.vram_guard import run_isolated_step
from _engine.pipeline.s1_prepare import run_step1

STAGE_ORDER = ["s1", "s2a", "s2b", "s3", "s4", "s5", "s6"]


class JobManager:
    """Quản lý trạng thái job, log trực tiếp, hàng đợi cuốn chiếu và đồng bộ Drive."""

    def __init__(self):
        self._lock = threading.RLock()
        self.telemetry = HardwareTelemetry(ROOT_DIR)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.logs: dict[str, deque[str]] = {}
        self._cancel_events: dict[str, threading.Event] = {}

        self._pipeline_queue: queue.Queue[tuple[str, bool, str]] = queue.Queue()
        self._drive_queue: queue.Queue[str] = queue.Queue()
        self._ws_clients: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()

        self._load_existing()

        threading.Thread(target=self._pipeline_worker_loop, name="pipeline-worker", daemon=True).start()
        threading.Thread(target=self._drive_worker_loop, name="drive-worker", daemon=True).start()

    def _load_existing(self):
        for m in scan_existing_work_manifests():
            jid = m["job_id"]
            # Nếu phiên trước bị tắt ngang lúc đang running -> chuyển về paused để bấm Resume chạy tiếp
            if m.get("status") == "running":
                m["status"] = "paused"
                for st in m.get("steps", {}).values():
                    if st.get("status") == "running":
                        st["status"] = "paused"
                save_manifest(m)
            self.jobs[jid] = m
            self.logs.setdefault(jid, deque(maxlen=400))

    def register_ws(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        with self._lock:
            self._ws_clients.add((loop, q))
        return q

    def unregister_ws(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue):
        with self._lock:
            self._ws_clients.discard((loop, q))

    def _notify_ws(self):
        with self._lock:
            clients = list(self._ws_clients)
        for loop, q in clients:
            def _put(target_q=q):
                if target_q.full():
                    try:
                        target_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    target_q.put_nowait(True)
                except asyncio.QueueFull:
                    pass
            try:
                loop.call_soon_threadsafe(_put)
            except RuntimeError:
                pass

    def append_log(self, job_id: str, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self.logs.setdefault(job_id, deque(maxlen=400)).append(line)
        self._notify_ws()

    def list_download_videos(self) -> list[dict[str, Any]]:
        """Liệt kê tất cả video gốc đang có trong `Download/` để chọn nhanh không cần upload lại."""
        if not DOWNLOAD_ROOT.exists():
            return []
        items = []
        for p in sorted(DOWNLOAD_ROOT.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".mp4", ".mkv", ".mov"):
                rel = p.resolve().relative_to(DOWNLOAD_ROOT.resolve()).as_posix()
                work_rel = infer_work_rel(p)
                items.append(
                    {
                        "path": str(p.resolve()),
                        "rel": rel,
                        "work_rel": work_rel,
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    }
                )
        return items

    def get_snapshot(self) -> dict[str, Any]:
        """Tạo snapshot tổng gửi ra Web UI qua HTTP/WebSocket."""
        drive_ok, drive_info = check_gdrive_remote()
        tunnel_file = ROOT_DIR / "_tunnel_url.txt"
        tunnel_url = tunnel_file.read_text(encoding="utf-8", errors="ignore").strip() if tunnel_file.exists() else ""
        with self._lock:
            jobs_list = sorted(self.jobs.values(), key=lambda x: x.get("updated_at", ""), reverse=True)
            logs_map = {jid: list(dq)[-80:] for jid, dq in self.logs.items()}
        return {
            "telemetry": self.telemetry.snapshot(),
            "drive_ready": drive_ok,
            "drive_info": "" if drive_ok else drive_info,
            "vast_configured": bool(get_vast_api_key()),
            "tunnel_url": tunnel_url,
            "jobs": jobs_list,
            "logs": logs_map,
        }

    def pull_drive_job(
        self,
        work_rel: str,
        mode: str = "full",
        langs: list[str] | None = None,
        voice_mode: str = "clone_2",
        auto_start: bool = True,
    ) -> dict[str, Any]:
        """Kéo gói Cách B từ `gdrive:MT-Video/<work_rel>/` về `_engine/work/` (dùng trên Vast.ai) rồi nạp vào hàng đợi."""
        rel = work_rel.replace("\\", "/").strip("/")
        work_dir = WORK_ROOT / rel
        pull_from_drive(work_rel=rel, work_dir=work_dir)

        src_mp4 = work_dir / "source.mp4"
        if not src_mp4.exists():
            raise FileNotFoundError(f"Gói Drive `{rel}` chưa có source.mp4 sau khi tải.")

        return self.add_or_attach_job(
            source_path=src_mp4,
            work_rel=rel,
            mode=mode,
            langs=langs or ["vi", "en", "ja", "ko", "es"],
            voice_mode=voice_mode,
            auto_sync_drive=False,
            auto_start=auto_start,
            force_step1=False,
        )

    def push_output_drive(self, job_id: str) -> dict[str, Any]:
        """Đẩy thư mục thành phẩm `output/<work_rel>/` lên Google Drive."""
        with self._lock:
            m = self.jobs.get(job_id)
            if not m:
                raise KeyError(f"Không tìm thấy job {job_id}")
            rel = m["work_rel"]
        out_dir = OUTPUT_ROOT / rel
        res = push_output_to_drive(rel, out_dir, log_cb=lambda msg: self.append_log(job_id, msg))
        self.append_log(job_id, f"[DRIVE OUTPUT] Đã đẩy xong thành phẩm -> {res['remote']}")
        return res

    def add_or_attach_job(
        self,
        source_path: Path,
        work_rel: str | None = None,
        mode: str = "test_5min",
        langs: list[str] | None = None,
        voice_mode: str = "clone_2",
        auto_sync_drive: bool = False,
        auto_start: bool = True,
        force_step1: bool = False,
    ) -> dict[str, Any]:
        """Thêm video vào hàng đợi — tự động bắt khớp checkpoint nếu video đã từng làm Bước 1."""
        src = source_path.resolve()
        if not src.exists():
            raise FileNotFoundError(f"Không tìm thấy video nguồn: {src}")

        with self._lock:
            matched = find_matching_manifest_by_source(src)
            if matched and not work_rel:
                rel = matched["work_rel"]
                work_dir = Path(matched["work_dir"])
                reused_msg = f"Phát hiện video đã có checkpoint tại `{rel}` — không phải làm lại Bước 1 đã xong!"
            else:
                rel = (work_rel or infer_work_rel(src)).replace("\\", "/").strip("/")
                work_dir = WORK_ROOT / rel
                reused_msg = ""

            manifest = load_or_create_manifest(
                work_dir=work_dir,
                work_rel=rel,
                source_path=src,
                mode=mode,
                langs=langs or ["vi", "en", "ja", "ko", "es"],
                voice_mode=voice_mode,
                auto_sync_drive=auto_sync_drive,
            )
            manifest["mode"] = mode
            if langs:
                manifest["langs"] = langs
            manifest["voice_mode"] = voice_mode
            manifest["auto_sync_drive"] = auto_sync_drive
            save_manifest(manifest)

            jid = manifest["job_id"]
            self.jobs[jid] = manifest
            self.logs.setdefault(jid, deque(maxlen=400))

        if reused_msg:
            self.append_log(jid, reused_msg)
        else:
            self.append_log(jid, f"Đã nạp job `{rel}` (chế độ: {mode}, ngôn ngữ: {','.join(manifest['langs'])})")

        if auto_start:
            self.enqueue_job(jid, force_step1=force_step1)
        else:
            self._notify_ws()
        return manifest

    def enqueue_job(self, job_id: str, mode: str | None = None, force_step1: bool = False, target_stage: str = "auto"):
        with self._lock:
            m = self.jobs.get(job_id)
            if not m:
                raise KeyError(f"Không tìm thấy job {job_id}")
            if m.get("status") == "running":
                return
            if mode:
                m["mode"] = mode
            if target_stage == "auto":
                has_cuda = self.telemetry._has_nvidia
                target_stage = "all" if has_cuda else "s1"
            m["status"] = "queued"
            save_manifest(m)
            self._cancel_events[job_id] = threading.Event()
        self.append_log(job_id, f"Đưa vào hàng đợi (mode={m['mode']}, target={target_stage}, force_s1={force_step1})")
        self._pipeline_queue.put((job_id, force_step1, target_stage))
        self._notify_ws()

    def pause_job(self, job_id: str):
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev:
                ev.set()
            m = self.jobs.get(job_id)
            if m and m.get("status") in ("running", "queued"):
                m["status"] = "paused"
                save_manifest(m)
        self.append_log(job_id, "Đã gửi tín hiệu tạm dừng — checkpoint đã lưu được giữ nguyên.")
        self._notify_ws()

    def trigger_drive_push(self, job_id: str):
        with self._lock:
            m = self.jobs.get(job_id)
            if not m:
                raise KeyError(f"Không tìm thấy job {job_id}")
            m["drive_sync"] = {
                "status": "queued",
                "remote": "",
                "detail": "Đang xếp hàng Upload Google Drive...",
                "updated_at": now_iso(),
            }
            save_manifest(m)
        self._drive_queue.put(job_id)
        self._notify_ws()

    def _update_step(self, job_id: str, step_key: str, **kwargs):
        with self._lock:
            m = self.jobs[job_id]
            st = m["steps"][step_key]
            st.update(kwargs)
            st["updated_at"] = now_iso()
            save_manifest(m)
        self._notify_ws()

    def _pipeline_worker_loop(self):
        while True:
            job_id, force_step1, target_stage = self._pipeline_queue.get()
            try:
                self._execute_job(job_id, force_step1, target_stage)
            except Exception as exc:
                self.append_log(job_id, f"[LỖI] {exc}")
                with self._lock:
                    m = self.jobs.get(job_id)
                    if m:
                        m["status"] = "error"
                        cur = m.get("current_step") or "s1"
                        m["steps"][cur]["status"] = "error"
                        m["steps"][cur]["detail"] = str(exc)
                        save_manifest(m)
                self._notify_ws()
            finally:
                self._pipeline_queue.task_done()

    def _run_stage_subprocess(self, job_id: str, step_key: str, mod: str, args: list[str], venv_key: str, cancel_ev: threading.Event):
        with self._lock:
            self.jobs[job_id]["current_step"] = step_key
            save_manifest(self.jobs[job_id])
        self._update_step(job_id, step_key, status="running", progress_pct=1.0, detail=f"Đang khởi chạy {step_key.upper()}...")
        t0 = time.perf_counter()
        run_isolated_step(
            module_name=mod,
            args=args,
            venv_key=venv_key,
            log_cb=lambda msg: self.append_log(job_id, msg),
            prog_cb=lambda pct, det: self._update_step(job_id, step_key, status="running", progress_pct=pct, detail=det),
            cancel_event=cancel_ev,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        self._update_step(job_id, step_key, status="done", progress_pct=100.0, elapsed_s=elapsed)

    def _execute_job(self, job_id: str, force_step1: bool, target_stage: str = "s1"):
        with self._lock:
            m = self.jobs.get(job_id)
            if not m or m.get("status") == "paused":
                return
            cancel_ev = self._cancel_events.setdefault(job_id, threading.Event())
            m["status"] = "running"
            m["current_step"] = "s1"
            save_manifest(m)
            src_mp4 = Path(m["source_path"])
            work_dir = Path(m["work_dir"])
            work_rel = m["work_rel"]
            mode = m.get("mode", "test_5min")
            langs = m.get("langs") or ["vi", "en", "ja", "ko", "es"]
            voice_mode = m.get("voice_mode", "clone_2")
            auto_sync = bool(m.get("auto_sync_drive", False))

        target_dir = (work_dir / "_test_5min") if mode == "test_5min" else work_dir
        self._update_step(job_id, "s1", status="running", progress_pct=1.0, detail=f"Đang chạy Bước 1 ({mode})...")

        try:
            # 1. Bước 1: Chuẩn hóa CFR GPU + WAV + Audit
            t0 = time.perf_counter()
            report = run_step1(
                src_mp4=src_mp4,
                work_dir=work_dir,
                mode=mode,
                force=force_step1,
                log_cb=lambda msg: self.append_log(job_id, msg),
                prog_cb=lambda pct, det: self._update_step(job_id, "s1", status="running", progress_pct=pct, detail=det),
                cancel_event=cancel_ev,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            with self._lock:
                m = self.jobs[job_id]
                if mode == "test_5min":
                    m["step1_test_ready"] = True
                    m["audit_test"] = report
                else:
                    m["step1_full_ready"] = True
                    m["audit_full"] = report
                spot = report.get("spotcheck", {})
                detail = (
                    f"Xong {mode} ({report.get('encoder_used')}, {report.get('timings_s', {}).get('total_step1_s')}s | "
                    f"khớp sub {spot.get('timeline_match_pct')}% | offset {spot.get('median_ocr_minus_asr_offset_s')}s)"
                )
                m["steps"]["s1"].update(
                    {"status": "done", "progress_pct": 100.0, "detail": detail, "elapsed_s": elapsed, "updated_at": now_iso()}
                )
                save_manifest(m)
            self._notify_ws()

            if target_stage == "s1":
                with self._lock:
                    self.jobs[job_id]["status"] = "step1_done"
                    self.jobs[job_id]["current_step"] = ""
                    save_manifest(self.jobs[job_id])
                self._notify_ws()
                if auto_sync:
                    self.trigger_drive_push(job_id)
                return

            # 2A -> 2B -> 3 (Nghe + Đọc + Đối chiếu Pinyin & Cứu hộ)
            self._run_stage_subprocess(job_id, "s2a", "_engine.pipeline.s2a_asr", [str(target_dir)], "asr_sep", cancel_ev)
            self._run_stage_subprocess(job_id, "s2b", "_engine.pipeline.s2b_ocr", [str(target_dir)], "vlm_llm", cancel_ev)
            self._run_stage_subprocess(job_id, "s3", "_engine.pipeline.s3_audit", [str(target_dir)], "asr_sep", cancel_ev)

            if target_stage == "s3":
                with self._lock:
                    self.jobs[job_id]["status"] = "s3_done"
                    self.jobs[job_id]["current_step"] = ""
                    save_manifest(self.jobs[job_id])
                self._notify_ws()
                return

            # 4 -> 5 -> 6 (Tách giọng vùng thoại & 2 mẫu Clone -> Qwen+Gemma 4 dịch -> VieNeu+Qwen3-TTS & Render)
            self._run_stage_subprocess(job_id, "s4", "_engine.pipeline.s4_separate", [str(target_dir)], "asr_sep", cancel_ev)
            self._run_stage_subprocess(
                job_id,
                "s5",
                "_engine.pipeline.s5_translate",
                [str(target_dir), "--langs", ",".join(langs)],
                "vlm_llm",
                cancel_ev,
            )
            s6_args = [str(target_dir), "--work-rel", work_rel, "--langs", ",".join(langs), "--voice-mode", voice_mode]
            if mode == "test_5min":
                s6_args.append("--test-5min")
            self._run_stage_subprocess(job_id, "s6", "_engine.pipeline.s6_tts_render", s6_args, "tts_ui", cancel_ev)

            with self._lock:
                self.jobs[job_id]["status"] = "done"
                self.jobs[job_id]["current_step"] = ""
                save_manifest(self.jobs[job_id])
            self._notify_ws()

            if auto_sync:
                self.trigger_drive_push(job_id)

        except InterruptedError as exc:
            self.append_log(job_id, str(exc))
            with self._lock:
                m = self.jobs[job_id]
                cur = m.get("current_step") or "s1"
                m["status"] = "paused"
                m["steps"][cur]["status"] = "paused"
                m["steps"][cur]["detail"] = "Đã tạm dừng — bấm Chạy tiếp để tiếp tục từ checkpoint"
                save_manifest(m)
            self._notify_ws()

    def _drive_worker_loop(self):
        while True:
            job_id = self._drive_queue.get()
            try:
                with self._lock:
                    m = self.jobs.get(job_id)
                    if not m:
                        continue
                    work_dir = Path(m["work_dir"])
                    work_rel = m["work_rel"]
                    src_mp4 = Path(m["source_path"])
                    m["drive_sync"]["status"] = "running"
                    save_manifest(m)
                self._notify_ws()

                def _prog(pct: float, det: str):
                    with self._lock:
                        self.jobs[job_id]["drive_sync"].update(
                            {"status": "running", "progress_pct": round(pct, 1), "detail": det, "updated_at": now_iso()}
                        )
                        save_manifest(self.jobs[job_id])
                    self._notify_ws()

                res = push_method_b_to_drive(
                    work_dir=work_dir,
                    work_rel=work_rel,
                    source_mp4=src_mp4,
                    log_cb=lambda msg: self.append_log(job_id, msg),
                    prog_cb=_prog,
                )
                with self._lock:
                    self.jobs[job_id]["drive_sync"] = {
                        "status": "done",
                        "progress_pct": 100.0,
                        "remote": res["remote"],
                        "detail": f"Đã đẩy {res['total_mb']} MB ({', '.join(res['files'])})",
                        "updated_at": now_iso(),
                    }
                    save_manifest(self.jobs[job_id])
                self._notify_ws()
            except Exception as exc:
                self.append_log(job_id, f"[DRIVE LỖI] {exc}")
                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["drive_sync"] = {
                            "status": "error",
                            "progress_pct": 0.0,
                            "remote": "",
                            "detail": str(exc),
                            "updated_at": now_iso(),
                        }
                        save_manifest(self.jobs[job_id])
                self._notify_ws()
            finally:
                self._drive_queue.task_done()

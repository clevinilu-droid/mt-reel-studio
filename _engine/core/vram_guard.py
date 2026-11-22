"""Quản lý cách ly VRAM 24 GB & chạy từng chặng AI trong venv độc lập.

Nguyên tắc vàng cho RTX 3090 (24 GB VRAM):
  - Mỗi model AI lớn chạy trong tiến trình con (subprocess) thuộc đúng venv của nó.
  - Khi tiến trình con kết thúc, driver CUDA trả lại 100% VRAM ngay lập tức, không rò rỉ bộ nhớ.
  - Tiến trình con in chuẩn `[PROG] <pct> | <detail>` để Web UI cập nhật thanh tiến độ live.
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from _engine.core.checkpoint import ENGINE_DIR, ROOT_DIR

CONFIG_PATH = ENGINE_DIR / "config" / "pipeline_v2.json"
PROG_RE = re.compile(r"^\[PROG\]\s*([\d.]+)\s*\|\s*(.*)$")

LogFn = Callable[[str], None]
ProgFn = Callable[[float, str], None]


def load_pipeline_config() -> dict[str, Any]:
    """Đọc cấu hình model & đường dẫn venv từ `_engine/config/pipeline_v2.json`."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"venvs": {}, "models": {}, "vllm": {}, "clone_ref": {}}


def resolve_venv_python(venv_key: str) -> str:
    """Trả về Python executable của venv mục tiêu (nếu có trên Vast/Docker), ngược lại dùng Python hiện tại."""
    cfg = load_pipeline_config()
    env_override = os.environ.get(f"MT_VENV_{venv_key.upper()}")
    if env_override and Path(env_override).exists():
        return env_override
    cand = cfg.get("venvs", {}).get(venv_key, "")
    if cand and Path(cand).exists():
        return cand
    return sys.executable


def release_vram():
    """Giải phóng bộ nhớ RAM/VRAM nếu chạy trong cùng tiến trình."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def emit_progress(pct: float, detail: str):
    """Hàm chuẩn để các script bước con đẩy tiến độ về JobManager & Web UI."""
    print(f"[PROG] {max(0.0, min(100.0, pct)):.1f} | {detail}", flush=True)


def run_isolated_step(
    module_name: str,
    args: list[str],
    venv_key: str,
    log_cb: LogFn | None = None,
    prog_cb: ProgFn | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """Chạy 1 module pipeline (`python -m _engine.pipeline.<mod> ...`) trong tiến trình tách biệt để giải phóng sạch VRAM."""
    py_exe = resolve_venv_python(venv_key)
    cmd = [py_exe, "-u", "-m", module_name, *args]
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(ROOT_DIR)
    env["PYTHONUTF8"] = "1"

    if log_cb:
        log_cb(f"[EXEC:{venv_key}] {Path(py_exe).name} -m {module_name} {' '.join(args)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    last_lines: list[str] = []
    try:
        for raw_line in proc.stdout:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise InterruptedError(f"Đã tạm dừng {module_name}")
            line = raw_line.rstrip()
            if not line:
                continue
            m = PROG_RE.match(line)
            if m:
                if prog_cb:
                    prog_cb(float(m.group(1)), m.group(2).strip())
            else:
                last_lines.append(line)
                if len(last_lines) > 25:
                    last_lines.pop(0)
                if log_cb:
                    log_cb(line)
        rc = proc.wait()
        if rc != 0:
            tail = " | ".join(last_lines[-4:])
            raise RuntimeError(f"{module_name} thất bại (exit={rc}): {tail}")
        return rc
    finally:
        if proc.poll() is None:
            proc.terminate()

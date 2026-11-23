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


def find_venv_cudnn_preload(py_exe_or_prefix: str | Path) -> str:
    """Tìm đường dẫn tuyệt đối tới `libcudnn.so.9` trong venv mục tiêu để gán LD_PRELOAD,
    ép Linux ld.so nạp cuDNN 9.5/9.8+ của venv trước `/usr/lib/x86_64-linux-gnu/libcudnn.so.9` (9.1.0 của base image).
    """
    import glob

    p = Path(py_exe_or_prefix)
    try:
        p_res = p.resolve()
    except Exception:
        p_res = p
    if p.name.startswith("python") or p_res.name.startswith("python"):
        venv_root = p.parent.parent if (p.parent.name == "bin" and p.parent.parent.exists()) else p_res.parent.parent
    else:
        venv_root = p if p.exists() else p_res

    pat = str(venv_root / "lib" / "python*" / "site-packages" / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9")
    matches = sorted(glob.glob(pat))
    return matches[0] if matches else ""


def build_venv_ld_library_path(py_exe_or_prefix: str | Path, base_ld: str | None = None) -> str:
    """Tạo LD_LIBRARY_PATH cô lập cho ĐÚNG 1 venv mục tiêu, loại bỏ mọi đường dẫn /opt/venvs/* khác để tránh xung đột cuDNN 9.1 vs 9.8."""
    import glob

    p = Path(py_exe_or_prefix)
    try:
        p_res = p.resolve()
    except Exception:
        p_res = p
    if p.name.startswith("python") or p_res.name.startswith("python"):
        venv_root = p.parent.parent if (p.parent.name == "bin" and p.parent.parent.exists()) else p_res.parent.parent
    else:
        venv_root = p if p.exists() else p_res

    lib_dirs: list[str] = []
    for pat in (
        str(venv_root / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"),
        str(venv_root / "lib" / "python*" / "site-packages" / "torch" / "lib"),
    ):
        for d in sorted(glob.glob(pat)):
            if d not in lib_dirs and Path(d).is_dir():
                lib_dirs.append(d)

    raw_ld = os.environ.get("LD_LIBRARY_PATH", "") if base_ld is None else base_ld
    kept_ld = [
        seg
        for seg in raw_ld.split(":")
        if seg and not seg.startswith("/opt/venvs/")
    ]
    return ":".join(dict.fromkeys([*lib_dirs, *kept_ld]))


def ensure_onnx_cuda_libs() -> list[str]:
    """Nạp sẵn CUDA/cuDNN shared libs từ CHỈ venv hiện tại (sys.prefix) để tránh xung đột symbol giữa các venv."""
    loaded: list[str] = []
    if not sys.platform.startswith("linux"):
        try:
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
                loaded.append("ort.preload_dlls")
        except Exception:
            pass
        return loaded

    try:
        import ctypes
        import glob

        os.environ["LD_LIBRARY_PATH"] = build_venv_ld_library_path(sys.prefix)
        base = Path(sys.prefix)
        lib_dirs: list[str] = []
        for pat in (
            str(base / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"),
            str(base / "lib" / "python*" / "site-packages" / "torch" / "lib"),
        ):
            for d in sorted(glob.glob(pat)):
                if d not in lib_dirs and Path(d).is_dir():
                    lib_dirs.append(d)

        # Thứ tự ưu tiên nạp: runtime/math trước, sau đó libcudnn.so.9 gốc rồi mới tới các mô-đun libcudnn_*.so.9 cùng thư mục
        priority_prefixes = (
            "libcudart.so",
            "libnvrtc-builtins.so",
            "libnvrtc.so",
            "libcublasLt.so",
            "libcublas.so",
            "libcufft.so",
            "libcurand.so",
            "libcusolver.so",
            "libcusparse.so",
            "libcudnn.so.9",
            "libcudnn_ops.so",
            "libcudnn_cnn.so",
            "libcudnn_adv.so",
            "libcudnn_graph.so",
            "libcudnn_heuristic.so",
            "libcudnn_engines_precompiled.so",
            "libcudnn_engines_runtime_compiled.so",
        )
        candidates: list[Path] = []
        for d in lib_dirs:
            candidates.extend(sorted(Path(d).glob("lib*.so*")))

        for pref in priority_prefixes:
            for so in candidates:
                if so.name == pref or so.name.startswith(pref):
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                        if so.name not in loaded:
                            loaded.append(so.name)
                    except Exception:
                        pass
    except Exception:
        pass

    if not any(x.startswith("libcudnn") for x in loaded):
        try:
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
                loaded.append("ort.preload_dlls")
        except Exception:
            pass
    return loaded


def patch_onnx_threads(threads: int = 4):
    """Giới hạn số luồng ONNX Runtime CPU (tránh spinlock 128 threads trên container giới hạn cgroup EPYC)."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))
    try:
        import onnxruntime as ort

        _orig = getattr(ort, "_MT_ORIG_SO", ort.SessionOptions)
        ort._MT_ORIG_SO = _orig

        class _SafeSessionOptions(_orig):
            def __init__(self):
                super().__init__()
                self.intra_op_num_threads = threads
                self.inter_op_num_threads = 1
                try:
                    self.add_session_config_entry("session.intra_op.allow_spinning", "0")
                except Exception:
                    pass

        ort.SessionOptions = _SafeSessionOptions
        try:
            import rapidocr_onnxruntime.utils as ru

            ru.SessionOptions = _SafeSessionOptions
        except Exception:
            pass
    except Exception:
        pass


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
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = build_venv_ld_library_path(py_exe, env.get("LD_LIBRARY_PATH", ""))
    env["PYTHONUTF8"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

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

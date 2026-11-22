"""Thu thập thông số phần cứng trực tiếp (GPU NVIDIA / Intel QSV, CPU, RAM, Ổ đĩa).

Không phụ thuộc thư viện nặng bên ngoài (`psutil`), dùng `nvidia-smi` khi có GPU rời
và Windows `ctypes` / Linux `/proc` để đo nhẹ dưới 2ms mỗi chu kỳ WebSocket.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


def _ft_to_int(ft: _FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


class HardwareTelemetry:
    """Bộ theo dõi CPU, RAM, Disk và GPU (NVIDIA CUDA hoặc Intel QSV)."""

    def __init__(self, watch_path: Path):
        self.watch_path = watch_path
        self._prev_cpu: tuple[int, int] | None = None
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self._static_gpu_name = self._detect_non_nvidia_gpu() if not self._has_nvidia else ""

    def _detect_non_nvidia_gpu(self) -> str:
        if platform.system() == "Windows":
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)"],
                    text=True,
                    timeout=4,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out:
                    return out
            except Exception:
                pass
        return "CPU / Integrated GPU"

    def _read_cpu_pct(self) -> float:
        if platform.system() == "Windows":
            idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                idle_i = _ft_to_int(idle)
                tot_i = _ft_to_int(kernel) + _ft_to_int(user)
                if self._prev_cpu is not None:
                    d_idle = idle_i - self._prev_cpu[0]
                    d_tot = tot_i - self._prev_cpu[1]
                    self._prev_cpu = (idle_i, tot_i)
                    if d_tot > 0:
                        return round(max(0.0, min(100.0, (1.0 - d_idle / d_tot) * 100.0)), 1)
                self._prev_cpu = (idle_i, tot_i)
            return 0.0
        # Linux /proc/stat
        try:
            parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            vals = [int(x) for x in parts[:8]]
            idle_i = vals[3] + vals[4]
            tot_i = sum(vals)
            if self._prev_cpu is not None:
                d_idle = idle_i - self._prev_cpu[0]
                d_tot = tot_i - self._prev_cpu[1]
                self._prev_cpu = (idle_i, tot_i)
                if d_tot > 0:
                    return round(max(0.0, min(100.0, (1.0 - d_idle / d_tot) * 100.0)), 1)
            self._prev_cpu = (idle_i, tot_i)
        except Exception:
            pass
        return 0.0

    def _read_ram(self) -> tuple[float, float, float]:
        if platform.system() == "Windows":
            mem = _MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                tot = mem.ullTotalPhys / (1024 ** 3)
                used = (mem.ullTotalPhys - mem.ullAvailPhys) / (1024 ** 3)
                return round(used, 2), round(tot, 2), round(float(mem.dwMemoryLoad), 1)
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = int(v.strip().split()[0])
            tot = info.get("MemTotal", 0) / (1024 ** 2)
            avail = info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 2)
            used = max(0.0, tot - avail)
            pct = (used / tot * 100.0) if tot > 0 else 0.0
            return round(used, 2), round(tot, 2), round(pct, 1)
        except Exception:
            return 0.0, 0.0, 0.0

    def _read_nvidia(self) -> dict[str, Any] | None:
        if not self._has_nvidia:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).strip()
            if not out:
                return None
            parts = [x.strip() for x in out.splitlines()[0].split(",")]
            return {
                "backend": "CUDA",
                "name": parts[0],
                "util_pct": float(parts[1]),
                "vram_used_gb": round(float(parts[2]) / 1024.0, 2),
                "vram_total_gb": round(float(parts[3]) / 1024.0, 2),
                "temp_c": int(float(parts[4])),
            }
        except Exception:
            return None

    def snapshot(self) -> dict[str, Any]:
        """Lấy toàn bộ ảnh chụp tài nguyên hiện tại."""
        ram_used, ram_total, ram_pct = self._read_ram()
        du = shutil.disk_usage(str(self.watch_path))
        gpu = self._read_nvidia()
        if gpu is None:
            gpu = {
                "backend": "QSV / Local",
                "name": self._static_gpu_name,
                "util_pct": None,
                "vram_used_gb": None,
                "vram_total_gb": None,
                "temp_c": None,
            }
        return {
            "cpu_pct": self._read_cpu_pct(),
            "cpu_cores": os.cpu_count() or 1,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "ram_pct": ram_pct,
            "disk_free_gb": round(du.free / (1024 ** 3), 1),
            "disk_total_gb": round(du.total / (1024 ** 3), 1),
            "gpu": gpu,
        }

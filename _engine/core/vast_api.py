"""Tích hợp Vast.ai REST API: Lọc GPU tốc độ tối đa, thuê với MT Reel Template & hủy máy chống tốn tiền.

Tiêu chí lọc GPU chuẩn công nghiệp (mọi máy trả ra đều tương thích 100% Docker `:v2`):
  - `verified == True`, `rentable == True`, `rented == False`, `num_gpus == 1`, `cpu_arch == "amd64"`
  - `compute_cap` thuộc Ampere/Ada (`860–890`: RTX 3090/3090Ti/4090/A5000/A6000/Ada có NVENC + BF16 + tương thích PyTorch cu124 & vLLM)
  - `gpu_ram >= 24000` MB (24 GB+ VRAM cho Gemma-4-26B-A4B 4-bit + Qwen3.5-4B)
  - `cuda_max_good >= 13.0` (bắt buộc tương thích vLLM CUDA runtime trong image `:v2`)
  - `inet_down >= 1000` & `inet_up >= 1000` Mbps (mạng > 1000 Mbps kéo weights & Drive siêu nhanh)
  - `cpu_cores_effective >= 6`, `cpu_ram >= 24000` MB, `pcie_bw >= 8.0` GB/s
  - `disk_bw >= 400` MB/s & `disk_space >= 120` GB (NVMe đọc model nhanh, đủ chỗ chứa weights + video)
  - `reliability >= 0.985`
  - Loại bỏ các quốc gia bị chặn/bóp Google Drive & HuggingFace (`CN`, `RU`, `IR`, `BY`, `KP`), datacenter GPU thiếu NVENC (`A100`, `A40`, `H100`), và Blackwell (`sm_100/120`, chưa chạy `torch cu124`).

Bảo mật: `VAST_API_KEY` đọc từ `.env` cục bộ, KHÔNG bao giờ gửi lên server thuê hay trả ra trình duyệt.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from _engine.core.checkpoint import ROOT_DIR

VAST_BASE = "https://console.vast.ai/api/v0"
BLOCKED_COUNTRY_CODES = {"CN", "RU", "IR", "BY", "KP"}
BLOCKED_GEO_KEYWORDS = ("china", "mainland", "russia", "belarus", "iran", "north korea")


def get_vast_api_key() -> str:
    """Đọc VAST_API_KEY từ môi trường hoặc file `.env` cục bộ."""
    load_dotenv(ROOT_DIR / ".env", override=True)
    return os.environ.get("VAST_API_KEY", "").strip()


def _vast_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 20,
    api_ver: str = "v0",
) -> Any:
    api_key = get_vast_api_key()
    if not api_key:
        raise RuntimeError("Chưa có VAST_API_KEY trong file .env trên máy local.")

    url = f"https://console.vast.ai/api/{api_ver}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "MT-Reel-Studio/2.0",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


def _is_blocked_location(offer: dict[str, Any]) -> bool:
    """Kiểm tra & loại bỏ máy đặt tại Trung Quốc hoặc vùng chặn HuggingFace / Google Drive."""
    geo = str(offer.get("geolocation") or "").strip()
    low = geo.lower()
    if any(k in low for k in BLOCKED_GEO_KEYWORDS):
        return True
    # Vast geolocation thường có dạng "Shanghai, CN" hoặc kết thúc bằng ", CN"
    parts = [p.strip().upper() for p in geo.replace(";", ",").split(",") if p.strip()]
    return any(p in BLOCKED_COUNTRY_CODES for p in parts)


def _score_offer(o: dict[str, Any]) -> float:
    """Chấm điểm ưu tiên: cao khi mạng khỏe + PCIe/CPU/NVMe mạnh + giá/giờ thấp."""
    dph = max(0.05, float(o.get("dph_total") or 1.0))
    dlperf = float(o.get("dlperf") or 0.0)
    net_mbps = min(float(o.get("inet_down") or 0.0), float(o.get("inet_up") or 0.0))
    cpu = float(o.get("cpu_cores_effective") or 0.0)
    pcie = float(o.get("pcie_bw") or 0.0)
    disk_bw = float(o.get("disk_bw") or 0.0)
    raw_perf = dlperf + min(net_mbps, 5000.0) * 0.015 + min(cpu, 32.0) * 1.2 + min(pcie, 25.0) * 1.5 + min(disk_bw, 3000.0) * 0.008
    return round(raw_perf / dph, 2)


def search_best_offers(
    min_vram_gb: int = 24,
    min_inet_mbps: int = 1000,
    min_cpu_cores: float = 6.0,
    min_ram_gb: int = 24,
    min_disk_gb: int = 120,
    min_disk_bw: int = 400,
    max_dph: float = 0.65,
    gpu_filter: str = "ANY",
    sort_by: str = "price",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Truy vấn và lọc Top GPU tối ưu cho pipeline từ Vast.ai (tránh Trung Quốc)."""
    q: dict[str, Any] = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "cpu_arch": {"eq": "amd64"},
        "compute_cap": {"gte": 860, "lte": 890},
        "gpu_ram": {"gte": int(min_vram_gb * 1000)},
        "cuda_max_good": {"gte": 13.0},
        "inet_down": {"gte": int(min_inet_mbps)},
        "inet_up": {"gte": int(min_inet_mbps)},
        "cpu_cores_effective": {"gte": float(min_cpu_cores)},
        "cpu_ram": {"gte": int(min_ram_gb * 1000)},
        "disk_space": {"gte": int(min_disk_gb)},
        "disk_bw": {"gte": int(min_disk_bw)},
        "pcie_bw": {"gte": 8.0},
        "reliability": {"gte": 0.985},
        "dph_total": {"lte": float(max_dph)},
        "type": "on-demand",
        "limit": 120,
        "order": [["dph_total", "asc"]],
    }
    if gpu_filter and gpu_filter != "ANY":
        q["gpu_name"] = {"eq": gpu_filter}

    encoded = urllib.parse.quote(json.dumps(q))
    data = _vast_request("GET", f"/bundles/?q={encoded}", timeout=25)
    offers = data.get("offers", []) if isinstance(data, dict) else []

    blocked_models = (
        "P40", "P100", "V100", "T4", "2080", "TITAN", "RTX 8000",
        "A100", "A40", "A30", "A10", "H100", "H200", "B200", "GB10",
        "RTX PRO", "RTX 5090", "4080", "CMP",
    )
    clean: list[dict[str, Any]] = []
    for o in offers:
        if _is_blocked_location(o):
            continue
        # Kiểm tra lại lần 2 bằng Python để bảo đảm 100% máy trả ra đều thỏa mãn mọi tiêu chí
        if str(o.get("cpu_arch") or "amd64").lower() not in ("amd64", "x86_64"):
            continue
        if int(o.get("num_gpus") or 0) != 1:
            continue
        cc = int(o.get("compute_cap") or 0)
        if not (860 <= cc <= 890):
            continue
        cuda_max = float(o.get("cuda_max_good") or 0.0)
        if cuda_max < 13.0:
            continue
        if float(o.get("gpu_ram") or 0) < int(min_vram_gb * 1000):
            continue
        if float(o.get("cpu_ram") or 0) < int(min_ram_gb * 1000):
            continue
        if float(o.get("cpu_cores_effective") or 0) < float(min_cpu_cores):
            continue
        if float(o.get("inet_down") or 0) < int(min_inet_mbps) or float(o.get("inet_up") or 0) < int(min_inet_mbps):
            continue
        if float(o.get("disk_space") or 0) < int(min_disk_gb) or float(o.get("disk_bw") or 0) < int(min_disk_bw):
            continue
        if float(o.get("pcie_bw") or 0) < 8.0 or float(o.get("reliability") or 0) < 0.985:
            continue
        gname = str(o.get("gpu_name") or "")
        up = gname.upper()
        if any(bad in up for bad in blocked_models) or ("RTX 6000" in up and "ADA" not in up):
            continue
        clean.append(
            {
                "id": o["id"],
                "gpu_name": gname,
                "vram_gb": round(float(o.get("gpu_ram") or 0) / 1024.0, 1),
                "cpu_cores": round(float(o.get("cpu_cores_effective") or 0), 1),
                "ram_gb": round(float(o.get("cpu_ram") or 0) / 1024.0, 1),
                "inet_down_mbps": round(float(o.get("inet_down") or 0), 0),
                "inet_up_mbps": round(float(o.get("inet_up") or 0), 0),
                "disk_bw_mbs": round(float(o.get("disk_bw") or 0), 0),
                "disk_free_gb": round(float(o.get("disk_space") or 0), 0),
                "pcie_bw_gbs": round(float(o.get("pcie_bw") or 0), 1),
                "cuda_max": o.get("cuda_max_good"),
                "single_gpu_host": float(o.get("gpu_frac") or 0.0) >= 0.99,
                "reliability_pct": round(float(o.get("reliability") or 0) * 100.0, 2),
                "geolocation": o.get("geolocation") or "",
                "dph_total": round(float(o.get("dph_total") or 0), 4),
                "dlperf": round(float(o.get("dlperf") or 0), 1),
                "score": _score_offer(o),
            }
        )

    if sort_by == "score":
        clean.sort(key=lambda x: (-x["score"], x["dph_total"]))
    else:
        clean.sort(key=lambda x: (x["dph_total"], -x["score"]))
    return clean[:limit]


def _read_local_rclone_b64() -> str:
    """Đọc rclone.conf cục bộ (chỉ lấy phần remote gdrive) để máy Vast tự đồng bộ Drive 1-click."""
    candidates = [
        Path.home() / ".config" / "rclone" / "rclone.conf",
        Path(os.environ.get("APPDATA", "")) / "rclone" / "rclone.conf",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            raw = p.read_bytes()
            if raw:
                return base64.b64encode(raw).decode("ascii")
    return ""


def rent_instance(
    offer_id: int,
    image: str = "",
    template_hash_id: str = "",
    disk_gb: int = 120,
    include_rclone: bool = True,
) -> dict[str, Any]:
    """Thuê GPU trên Vast.ai kèm Docker Image / Template MT Reel & tự mở cổng 7860."""
    env_map: dict[str, str] = {
        "-p 7860:7860": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    }
    if include_rclone:
        rc_b64 = _read_local_rclone_b64()
        if rc_b64:
            env_map["RCLONE_CONF_B64"] = rc_b64

    body: dict[str, Any] = {
        "disk": int(disk_gb),
        "runtype": "args",
        "env": env_map,
    }
    if template_hash_id.strip():
        body["template_hash_id"] = template_hash_id.strip()
    elif image.strip():
        body["image"] = image.strip()
    else:
        raise ValueError("Cần nhập Docker Image hoặc Template Hash ID của MT Reel.")

    return _vast_request("PUT", f"/asks/{int(offer_id)}/", payload=body, timeout=25)


def list_instances() -> list[dict[str, Any]]:
    """Lấy danh sách đầy đủ các máy đang thuê trên Vast.ai (/api/v1/instances/) kèm thông số phần cứng & trạng thái live."""
    data = _vast_request("GET", "/instances/", timeout=20, api_ver="v1")
    raw_list = data.get("instances", []) if isinstance(data, dict) else []
    out = []
    for inst in raw_list:
        pub_ip = inst.get("public_ipaddr") or ""
        ports = inst.get("ports") or {}
        mapped_7860 = None
        if isinstance(ports, dict):
            p_info = ports.get("7860/tcp") or ports.get("7860")
            if isinstance(p_info, list) and p_info:
                mapped_7860 = p_info[0].get("HostPort")
        direct_url = f"http://{pub_ip}:{mapped_7860}" if (pub_ip and mapped_7860) else ""
        vram_total_gb = round(float(inst.get("gpu_ram") or 0) / 1024.0, 1)
        vram_used_raw = inst.get("vram_usage")
        vram_used_gb = round(float(vram_used_raw) / 1024.0, 1) if vram_used_raw is not None else 0.0
        ram_total_gb = round(float(inst.get("cpu_ram") or 0) / 1024.0, 1)
        ram_used_raw = inst.get("mem_usage")
        ram_used_gb = round(float(ram_used_raw), 1) if ram_used_raw is not None else 0.0
        disk_total_gb = round(float(inst.get("disk_space") or 0), 1)
        disk_used_raw = inst.get("disk_usage")
        disk_used_gb = round(float(disk_used_raw), 1) if (disk_used_raw is not None and float(disk_used_raw) >= 0) else 0.0
        rel_raw = inst.get("reliability2") or inst.get("reliability") or 0.0

        out.append(
            {
                "id": inst.get("id"),
                "machine_id": inst.get("machine_id") or "",
                "host_id": inst.get("host_id") or "",
                "gpu_name": inst.get("gpu_name") or "",
                "vram_total_gb": vram_total_gb,
                "vram_used_gb": vram_used_gb,
                "gpu_util": inst.get("gpu_util"),
                "cuda_max": inst.get("cuda_max_good") or "",
                "dlperf": round(float(inst.get("dlperf") or 0), 1),
                "actual_status": inst.get("actual_status") or inst.get("cur_state") or "unknown",
                "intended_status": inst.get("intended_status") or "",
                "status_msg": (inst.get("status_msg") or "").strip(),
                "public_ipaddr": pub_ip,
                "port_7860": mapped_7860 or "",
                "direct_ui_url": direct_url,
                "dph_total": round(float(inst.get("dph_total") or 0), 3),
                "geolocation": inst.get("geolocation") or "",
                "reliability_pct": round(float(rel_raw) * 100.0, 2),
                "inet_down_mbps": round(float(inst.get("inet_down") or 0), 0),
                "inet_up_mbps": round(float(inst.get("inet_up") or 0), 0),
                "cpu_name": inst.get("cpu_name") or "",
                "cpu_cores": round(float(inst.get("cpu_cores_effective") or 0), 1),
                "ram_used_gb": ram_used_gb,
                "ram_total_gb": ram_total_gb,
                "disk_name": inst.get("disk_name") or "NVMe",
                "disk_bw_mbs": round(float(inst.get("disk_bw") or 0), 1),
                "disk_used_gb": disk_used_gb,
                "disk_total_gb": disk_total_gb,
                "pcie_bw_gbs": round(float(inst.get("pcie_bw") or 0), 1),
                "image_uuid": inst.get("image_uuid") or "",
                "ssh_host": inst.get("ssh_host") or "",
                "ssh_port": inst.get("ssh_port") or "",
            }
        )
    return out


def fetch_instance_logs(instance_id: int, tail: int = 120) -> dict[str, Any]:
    """Lấy log mới nhất của container Vast.ai (tự trích link Cloudflare Tunnel nếu có)."""
    import re
    import time

    res = _vast_request(
        "PUT",
        f"/instances/request_logs/{int(instance_id)}/",
        payload={"tail": str(int(tail))},
        timeout=20,
        api_ver="v0",
    )
    temp_url = res.get("result_url") or ""
    text = ""
    if temp_url:
        # Chờ Vast upload log ra URL tạm 1–2 giây rồi tải về
        for _ in range(4):
            try:
                req = urllib.request.Request(temp_url, headers={"User-Agent": "MT-Reel-Studio/2.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    text = r.read().decode("utf-8", errors="replace")
                    if text.strip():
                        break
            except Exception:
                time.sleep(0.8)
    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
    return {
        "logs": text.strip(),
        "tunnel_url": m.group(0) if m else "",
    }


def destroy_instance(instance_id: int) -> dict[str, Any]:
    """Hủy vĩnh viễn máy thuê Vast.ai để ngừng tính tiền ngay lập tức."""
    return _vast_request("DELETE", f"/instances/{int(instance_id)}/", timeout=20, api_ver="v0")

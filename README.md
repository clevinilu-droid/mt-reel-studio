# 🎬 MT-Reel Studio — Quy Trình Làm Phim AI Đa Ngôn Ngữ (Từ Con Số 0)

> Hệ thống sản xuất video phim ngắn AI Trung Quốc đa ngôn ngữ (**Việt / Anh / Nhật / Hàn**) chuẩn **YouTube Multi-Language Audio + Softsub**, kết hợp **Laptop Windows (`FFmpeg` + AI Vision)** và **GPU ngoài Vast.ai (`Docker GHCR Image`)**.

---

## 📁 Cấu Trúc Thư Mục
* **[QUY_TRINH_CHUAN.md](QUY_TRINH_CHUAN.md)** — Tài liệu chi tiết 5 bước vận hành, bộ lọc GPU Vast.ai tối ưu chi phí và 3 điểm chốt duyệt.
* **[docker/Dockerfile](docker/Dockerfile)** & **[.github/workflows/build-vast-image.yml](.github/workflows/build-vast-image.yml)** — Đóng gói tự động Image Vast.ai lên `ghcr.io/clevinilu-droid/mt-reel-vast:latest` gồm:
  * `PyTorch 2.10 + CUDA 12.8` (tương thích từ RTX 30xx/40xx đến Blackwell RTX 5060 Ti / 5070 Ti)
  * `FFmpeg 8.1` (`h264_nvenc` + `libass` + `gblur`) & `rclone`
  * `audio-separator 0.48.0` + model `Mel-Band-RoFormer` (`melband_roformer_instvox_duality_v2.ckpt`)
  * `VieNeu-TTS` (`vieneu[cuda]`) cho lồng tiếng Tiếng Việt 48 kHz
  * `pyVideoTrans v4.19` (`/opt/pyvideotrans`, Python 3.10 qua `uv`) cho lồng tiếng/clone giọng `EN / JA / KO`
* **`scripts/`** — Bộ script sạch chạy Bước 3 (`step3_extract_hardsub.py`, `step3_sync_and_gender.py`).
* **`Download/`** & **`projects/`** *(không đưa lên Git)* — Chứa phim gốc và dữ liệu xử lý từng bộ phim.

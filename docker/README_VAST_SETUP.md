# 🚀 Hướng Dẫn Setup 1 Lần Duy Nhất Cho Vast.ai (MT Reel Studio v2)

Làm **1 lần duy nhất** để tạo **Docker Image + Vast.ai Template** — từ đó về sau mỗi lần làm phim mới (`P1`, `P2`, `002`...) chỉ việc bấm nút **Thuê** ngay trong Web UI ở nhà.

---

## Bước 1: Tạo tài khoản Docker Hub miễn phí & Build Image (1 lần duy nhất)

Tại máy có Docker (hoặc ngay trên 1 con Vast.ai thuê 15 phút để build), đứng tại thư mục `MT-Video/`:

```bash
docker login
docker build -f docker/Dockerfile -t <TEN_DOCKERHUB_CUA_BAN>/mt-reel-studio:v2 .
docker push <TEN_DOCKERHUB_CUA_BAN>/mt-reel-studio:v2
```

*(Ghi chú: Để **tự động đăng nhập Google Drive** khi máy mới khởi động, có thể điền biến `RCLONE_CONF_B64` vào Vast.ai Template hoặc thuê qua tab **`☁️ Vast.ai`** trên Web UI ở nhà — hệ thống sẽ tự gửi kèm `rclone.conf` một cách an toàn mà không bao giờ gửi `VAST_API_KEY` lên server).*

---

## Bước 2: Lưu làm Template trên Vast.ai (tùy chọn, hoặc điền thẳng tên Image vào UI)

Trên trang Vast.ai → **Templates → New Template**:
* **Image Path:** `<TEN_DOCKERHUB_CUA_BAN>/mt-reel-studio:v2`
* **Expose Ports:** `7860` (`http`)
* **Disk Space:** tối thiểu `120 GB`
* **Launch Mode:** `Run interactive shell / Docker ENTRYPOINT` (giữ nguyên `/entrypoint.sh` tự bật Web UI `:7860` + Cloudflare HTTPS link).

---

## Bước 3: Sử dụng hàng ngày (0 mở Notebook, 0 gõ Terminal)

1. Ở nhà chạy [start_ui.bat](file:///d:/MT-Box/MT-Video/start_ui.bat) → mở **http://127.0.0.1:7860**.
2. Vào tab **`☁️ Vast.ai`** → bấm **Tìm Top GPU** *(tự lọc `VRAM >= 24GB`, `Mạng >= 1000 Mbps`, `RAM >= 32GB`, `NVMe >= 500 MB/s`, `tránh Trung Quốc`)* → chọn máy bấm **`⚡ Thuê ngay`**.
3. Sau ~1 phút máy bật xong, bấm link mở Web UI ngay trên con RTX 3090 → bấm **`☁️ Kéo từ Google Drive`** chọn `001_.../P1` → bấm **`🧪 Test 5 phút`** → duyệt xong bấm **`🚀 Full`** → **Hủy máy** khi hoàn tất!

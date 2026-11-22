# MT-Video — Bộ Tài Liệu Tham Khảo Toàn Diện: Làm Phim AI & Kiếm Tiền YouTube

Thư mục này lưu trữ toàn bộ kiến thức, nguồn video gốc Trung Quốc, công cụ mã nguồn mở trên GitHub, phân tích đối thủ thực tế và chiến lược kiếm tiền đã nghiên cứu để bạn tra cứu lâu dài.

---

## 🎯 Các Quyết Định Đã Chốt Cho Dự Án
* **Mục tiêu kiếm tiền:** Không bán hàng vật lý, không làm dịch vụ B2B/nhận job. Ưu tiên **YouTube AdSense đa quốc gia (trung/dài hạn)** bằng video dài 1–3 tiếng kiểu `Báo Đốm Review` (kết hợp cắt Shorts/TikTok/FB Reels kéo traffic & ăn **CPS ngắn hạn** nếu phù hợp).
* **Cấu trúc kênh (Phần 1):** **Phương án A — 1 Kênh YouTube duy nhất sử dụng tính năng Âm thanh Đa Ngôn Ngữ (`Multi-Language Audio`)** để gom chung lượt xem và nhanh đạt mốc `1.000 sub + 4.000 giờ xem`. *(Lưu ý: Nếu test reup thô kiểu công nghiệp, nên test trên 1 kênh phụ ở Gmail riêng để bảo vệ kênh chính).*
* **Bộ 4 ngôn ngữ ưu tiên (Đã chốt):**
  1. **Tiếng Việt (`vi`)** — Kéo view & gom nhanh 4.000 giờ xem giai đoạn đầu.
  2. **Tiếng Anh (`en-US`)** — Ăn trọn 5 thị trường Tier-1 (Mỹ, Úc, Canada, Anh, Singapore).
  3. **Tiếng Nhật (`ja`)** — Thị trường Tier-2 Châu Á RPM cao (\$2–\$4), cực hợp gu *Trọng sinh / Hệ thống / Vả mặt (`ざまぁ`)*.
  4. **Tiếng Hàn (`ko`)** — Thị trường Tier-2 Châu Á RPM cao (\$1.8–\$3.5), cực hợp gu *Hồi quy (`회귀`) / Tài phiệt / Báo thù (`사이다`)*.
* **Phần cứng hiện có:** **Laptop Intel Core i5-11300H (4 nhân 8 luồng) + Intel Iris Xe Graphics (không có card đồ họa rời, nhưng CÓ chip mã hóa phần cứng `Intel Quick Sync / h264_qsv`)** $\rightarrow$ Mọi khâu tạo hình AI nặng chạy qua Web/API đám mây; laptop chạy mượt `FFmpeg` (có thể bật `h264_qsv` tăng tốc render), `faster-whisper` (`small`/`int8`), `edge-tts`, và script tải Hồng Quả tự động.

---

## 🗂️ Cấu Trúc Thư Mục Chuẩn

```text
MT-Video/
├── Download/                  ← Phim gốc tải về (chia theo từng bộ phim)
├── output/                    ← Thành phẩm xuất ra — MỖI PHIM 1 THƯ MỤC CON, ĐÚNG 1 FILE DUY NHẤT
│   └── 001_BaoBiao_Ep01_05/
│       └── 001_BaoBiao_Ep01_05.mp4   ← Chỉ giữ đúng 1 file video sản phẩm sạch
├── docs/                      ← Toàn bộ tài liệu nghiên cứu & bảng quản lý phim / kênh YouTube
├── _engine/                   ← Script, model AI & thư mục nháp/track phụ (_engine/work/<TenPhim>/)
├── .venv/                     ← Môi trường Python & thư viện AI (không tự ý di chuyển)
├── data/ & hongguo-downloader-7.6-win-x64.exe  ← App tải Hồng Quả bản portable
└── README.md                  ← Mục lục dự án
```

---

## 📚 Mục Lục Tài Liệu & Bảng Quản Lý Dự Án

### 🗂️ Quản Lý Vận Hành & Tra Cứu Nhanh (Cập Nhật Thường Xuyên)
* **[DANH_SACH_PHIM.csv](DANH_SACH_PHIM.csv)** & **[DANH_SACH_PHIM.md](DANH_SACH_PHIM.md)** — **File quản lý tổng hợp toàn bộ phim (mở bằng Excel / Google Sheets hoặc xem trực tiếp): mã series, phần P1/P2/P3/END, nguồn tải, ID gốc, link YouTube, tiến độ từng phần** *(tự động cập nhật bằng `_engine/scripts/scan_series.py`)*.
* **[docs/quan_ly_phim.md](docs/quan_ly_phim.md)** — **Quy trình chuẩn đặt tên thư mục Series → Phần (`P1 / P2 / P3 → END`) & hướng dẫn quản lý kho phim.**
* **[docs/kho_san_phim_hot.md](docs/kho_san_phim_hot.md)** — **Radar săn phim hot nhiều mùa (`P2–P14`) trên Hồng Quả / Bilibili & theo dõi phần tiếp theo khi viewer hỏi.**
* **[docs/quan_ly_kenh_youtube.md](docs/quan_ly_kenh_youtube.md)** — **Bảng quản lý các kênh YouTube của bạn (Handle, trạng thái `Multi-Audio`, sub, giờ xem, YPP).**
* **[docs/bang_tham_khao_rpm.md](docs/bang_tham_khao_rpm.md)** — **Bảng tham khảo RPM 12 thị trường & công thức RPM chính thức của YouTube.**

### 📖 4 Tài Liệu Nghiên Cứu Tham Khảo

1. **[docs/quy_trinh_az_va_chinh_sach_2026.md](docs/quy_trinh_az_va_chinh_sach_2026.md)** — **Quy Trình Thực Chiến Từ A–Z, Xếp Hạng Thị Trường RPM Cao Nhất & Chính Sách 2026**
   * Bảng xếp hạng RPM 12 thị trường trả tiền view cao nhất thế giới (**Úc, Mỹ, Na Uy, Canada, Thụy Sĩ, Anh, Singapore, Đức, Nhật Bản, Hàn Quốc, Đài Loan, Việt Nam**) và bộ 4 ngôn ngữ tối ưu nhất cho phim AI Trung Quốc.
   * Đánh giá chính sách kiếm tiền cập nhật 2026 của **YouTube** (`Inauthentic Content`, `Reused Content`, nhãn AI), **Facebook** (`Facebook Content Monetization` hợp nhất, rào cản tại VN), **TikTok** (`Creator Rewards` 8 nước & luật `5-Strike Unoriginal`), cùng **Snapchat, Rumble, Dailymotion**.
   * **Quy trình 7 bước chi tiết từ A–Z**: Thiết lập kênh $\rightarrow$ Săn nguồn sạch $\rightarrow$ Dịch bản địa hóa 4 thứ tiếng $\rightarrow$ Lồng tiếng & tái chế hình ảnh $\rightarrow$ Cài đặt `Multi-Language Audio` $\rightarrow$ Kéo traffic đa kênh & CPS $\rightarrow$ Bật YPP, khai thuế `W-8BEN` và nhận tiền USD tại Việt Nam.

2. **[docs/nguon_video_trung_quoc.md](docs/nguon_video_trung_quoc.md)** — **Bản Đồ Toàn Bộ Nguồn Video, Kho Kịch Bản & Công Cụ Tải Từ Trung Quốc**
   * Danh sách đầy đủ các kênh gốc (UP主) đã kiểm chứng trên **Bilibili** (bản Full 1–7 tiếng, miễn phí).
   * Danh sách các kênh tiếng Trung gốc trên **YouTube** (nhiều kênh có sẵn phụ đề mềm `CC / MultiSub`).
   * Phân tích thực tế các nền tảng khác: **Hồng Quả Đoạn Kịch (`红果`)**, **Baidu Haokan (`好看视频`)**, **Douyin (`抖音`)**, **WeChat Channels (`微信视频号`)**, **Xiaohongshu (`小红书`)**, **Tân Phiến Trường (`新片场`)**, **Xigua (`西瓜视频`)**, **Kuaishou (`快手`)**.
   * Kho lấy kịch bản truyện gốc (**番茄小说**, **知乎盐言故事**, **WaytoAGI**) và bộ từ khóa săn nguồn hàng ngày.
   * Chi tiết toàn bộ công cụ tải chuyên dụng trên GitHub (`yt-dlp`, `hongguo-downloader`, `res-downloader`, `Douyin_TikTok_Download_API`, `TikTokDownloader`, `BiliDown`) + lưu ý kỹ thuật khi tải.

3. **[docs/chien_luoc_kiem_tien_va_doi_thu.md](docs/chien_luoc_kiem_tien_va_doi_thu.md)** — **Chiến Lược Kiếm Tiền, Phân Tích Đối Thủ & Bài Toán Chi Phí**
   * Bóc tách 4 cơ chế dòng tiền YouTube/TikTok (RPM Mỹ vs Việt Nam, Shorts vs Long-form, CPS/Affiliate).
   * Chiến lược **"Nhất ngư đa thực" (1 video ăn 2 đầu tiền)** & cách vận hành **1 Kênh Đa Ngôn Ngữ** đúng luật YouTube 2025–2026 (tránh án tử *Inauthentic Content*, *Reused Content*, và luật *Khóa chùm Brand Account*).
   * Hồ sơ phân tích trực tiếp **7 kênh thực tế** (`Báo Đốm Review`, `Huyền Nhung Review`, `Thần Thần review`, `Vân Anh Review`, `Bao Công Review`, `Búp Măng Non`, `Vạn Điều Kỳ Thú`).
   * So sánh chi tiết **Thời gian – Chi phí – Phần cứng** giữa **Hướng 1 (Dịch/Lồng tiếng từ nguồn TQ)** và **Hướng 2 (Tự tạo phim AI giống thật)**.

4. **[docs/cong_cu_github_va_ky_thuat_ai.md](docs/cong_cu_github_va_ky_thuat_ai.md)** — **Kho Công Cụ GitHub & Bí Thuật Làm Phim AI Giống Thật (`去AI味`)**
   * Danh mục đầy đủ các dự án mã nguồn mở trên GitHub chia theo 4 nhóm chức năng (Dịch & Lồng tiếng, Xóa phụ đề cứng, Tự động xuất bản nháp CapCut, All-in-One tạo phim AI, Model AI mã nguồn mở).
   * Bộ công cụ Web cho máy không card rời (**Google ImageFX/Whisk, Dreamina, SeaArt/Flux.1, Hailuo AI, Kling AI, Vidu AI, Seedance**).
   * **6 Bí thuật làm phim AI giống người thật** của các studio Trung Quốc (Prompt "khử da nhựa", Đa chủ thể tham chiếu, Quy tắc Vi biểu cảm, Khung hình Đầu-Cuối, LivePortrait, 4 lớp phủ hậu kỳ CapCut).

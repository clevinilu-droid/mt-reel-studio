# 🎬 QUY TRÌNH CHUẨN LÀM PHIM AI ĐA NGÔN NGỮ (YOUTUBE MULTI-AUDIO + SOFTSUB)

> **Tài liệu chuẩn hóa quy trình từ con số 0** — kết hợp **Laptop Windows (`FFmpeg` + AI Vision)** và **GPU ngoài Vast.ai (`pyVideoTrans v4.19`)**, vận hành từng bước có điểm chốt duyệt.

---

## 1. Cấu Trúc Thư Mục Quản Lý

```text
MT-Video/
├── Download/                                    ← [CHỈ ĐỌC] Phim gốc tải từ Hồng Quả / Bilibili
│   └── 001_<TenPhim>/
│       ├── P1/
│       └── P2/
├── pyvideotrans/                                ← Mã nguồn pyVideoTrans mới nhất (v4.19)
├── QUY_TRINH_CHUAN.md                           ← Tài liệu quy trình chuẩn (file này)
└── projects/                                    ← [KHU LÀM VIỆC & THÀNH PHẨM]
    └── 001_<TenPhim>/
        ├── HO_SO_NHAN_VAT.md                    ← Bảng nhân vật & xưng hô dùng chung P1, P2, P3...
        ├── character_glossary.json              ← Từ điển nhân vật khóa cứng cho LLM dịch
        └── P1/
            ├── 00_buoc1_chuan_hoa/                ← 🔒 Master video chuẩn (source_norm.mp4: 1080x1920 CFR 30fps + 48kHz)
            ├── 01_buoc2_nhan_vat_va_khung_blur/ ← 🔍 Chốt duyệt #1 (grids/, blur_truoc_sau.jpg, blur_config.json)
            ├── 02_buoc3_hardsub_va_am_thanh/    ← zh_hardsub.srt + vocals + instrumental
            ├── 03_buoc4_ban_dich_da_ngon_ngu/   ← 🔍 Chốt duyệt #2 (vi.srt, en.srt, ja.srt, ko.srt + role_lines.json)
            └── 04_buoc5_thanh_pham_youtube/     ← 🔍 Chốt duyệt #3 (.mp4 làm mờ sạch sub cũ, audio_tracks/, subtitles/ softsub, youtube_seo/)
```

---

## 2. Quy Trình 5 Bước Chuẩn (Từ A–Z)

### Bước 1 — Chuẩn hóa nguồn & Image GPU
* **1.1 Nguồn gốc & Image GPU:** Tải phim gốc (tách gọn theo `P1`, `P2`...) vào `Download/`, giữ nguyên 100% file gốc không sửa đè. Dùng Docker Image đóng gói sẵn trên GitHub Container Registry **`ghcr.io/clevinilu-droid/mt-reel-vast:latest`** làm Template Vast.ai (chứa sẵn `PyTorch 2.10 CUDA 12.8`, `FFmpeg 8.1 NVENC`, `rclone`, `audio-separator 0.48.0` + model `Mel-Band-RoFormer`, `VieNeu-TTS`, và `pyVideoTrans v4.19`).
* **1.2 Chuẩn hóa Video Master (`scripts/step1_normalize.py`, bắt buộc trước mọi bước):** Phim gốc ghép từ hàng chục/trăm tập thường bị lệch 1–4px giữa các tập (`1080×1924` trộn `1080×1920`) và trôi nhẹ khung hình (`29.835` vs `30 fps`). Vì vậy bắt buộc xuất ra **`00_buoc1_chuan_hoa/source_norm.mp4`** khóa cứng **`1080×1920` (SAR 1:1, cắt gọn mép thừa nếu lệch pixel), `30 fps CFR`, âm thanh `48kHz Stereo` (`aresample=async=1`)** bằng phần cứng Intel QSV (`$0` trên Laptop) hoặc NVENC — toàn bộ Bước 2 → Bước 5 đọc chung 1 file chuẩn này.

### Bước 2 — Quét Bảng Tên Dọc & Canh Khung Làm Mờ Hardsub *(Chạy trên Laptop, \$0)*
1. **Quét Hồ sơ nhân vật:** Dùng `FFmpeg` cắt lưới ảnh `6×8` nửa trên màn hình (`65%` chiều cao, `fps=1/2`, mỗi ảnh `96 giây`) $\rightarrow$ dùng Gemini Vision đọc chữ thư pháp dọc (`人物花字`) kết hợp nhìn mặt diễn viên để lập `HO_SO_NHAN_VAT.md` và `character_glossary.json` (Tên Hán-Việt, EN/JA/KO, Giới tính, Tuổi, Vai vế, Ma trận xưng hô giữa các cặp nhân vật).
2. **Đo tọa độ làm mờ hardsub đáy:** Trích khung hình thực tế có chữ dài nhất ở đáy màn hình, đo khung tọa độ chính xác `X:Y:W:H` và xuất ảnh `blur_truoc_sau.jpg` kiểm chứng bằng mắt.
3. **Chính sách phụ đề thành phẩm (Softsub):**
   * Video `.mp4` thành phẩm **CHỈ làm mờ dải chữ Trung gốc** (`gblur=sigma=22` theo `blur_config.json`), **KHÔNG đốt cứng (burn-in) phụ đề mới vào hình** vì khi đăng YouTube dùng **Softsub** đa ngôn ngữ để trình phát YouTube tự hiển thị theo ngôn ngữ người xem.
   * Cấu hình style ưu tiên cho định dạng softsub có hỗ trợ kiểu dáng (`.vtt` / `.ass` khi xem thử hoặc xuất bản hỗ trợ cue): chữ **màu vàng `#FFEB50`**, viền đen `#000000`, đặt tại vùng `Y` đè khớp lên dải vừa làm mờ. *(Lưu ý: Với file `.srt` up lên YouTube, giao diện hiển thị do cài đặt phụ đề của YouTube/người xem quyết định; nếu muốn giữ màu vàng & vị trí cố định trên YouTube cần dùng định dạng `.vtt` có cue/style được hỗ trợ hoặc `.srv3/.ytt`).*
* 🔍 **Điểm bạn duyệt #1:** Kiểm tra `HO_SO_NHAN_VAT.md` (tên, giới tính, xưng hô) và `blur_truoc_sau.jpg` (độ che kín của vết làm mờ).

### Bước 3 — Trích xuất Hardsub gốc & Tách âm thanh mới *(Độ khớp Nội dung + Timeline 1:1)*
* **Nguyên tắc cốt lõi:** Phụ đề Hardsub gốc ở đáy và giọng nói gốc trong video là cặp 1:1 cùng timeline `0.000s` (không phụ thuộc việc trên hình có thấy mặt người đang nói hay không).
* **Tách 2 luồng song song từ file gốc:**
  1. **Luồng Âm thanh (`audio-separator` + `Mel-Band-RoFormer` trên GPU Vast.ai):** Tách mới hoàn toàn từ video gốc ra **Giọng nói sạch (`vocals_48k_mono.flac`)** và **Nhạc nền + Hiệu ứng (`instrumental_48k.flac`)** chuẩn `48kHz` Lossless (vừa nhẹ hơn `.wav` gần 4 lần, vừa giữ trọn 48kHz cho clone giọng `CosyVoice`/`F5-TTS`).
  2. **Luồng Chữ (`Đọc Hardsub gốc` trên Laptop + Gemini Vision):** Tách chữ từ vùng đáy cố định đã đo ở Bước 2 theo cơ chế bắt viền chữ đổi câu (mỗi câu 1 dải ảnh ghép lưới) ra file `zh_hardsub_raw.srt` — đúng 100% mặt chữ tác giả, không sai tên nhân vật đồng âm.
* **Khớp sóng âm (`Audio-Snap` & Phân vai bằng âm thanh):**
  * Kéo mốc `start --> end` của `zh_hardsub_raw.srt` bám khít vào sóng âm thực tế của `vocals_48k_mono.flac` $\rightarrow$ xuất `zh_master_synced.srt`.
  * Gộp câu dài bị màn hình dọc cắt đôi (nếu hai dòng nối liền không có khoảng nghỉ lấy hơi trên `vocals`).
  * Phân biệt **Lời thoại** (có tiếng trên `vocals` $\rightarrow$ đo cao độ `F0 Hz` gắn nhãn Nam/Nữ) với **Chữ chú thích câm** (im lặng trên `vocals` $\rightarrow$ chỉ giữ text softsub, không đọc TTS).

---

## 3. Bộ Lọc Thuê GPU Vast.ai Tối Ưu Giá Rẻ & Hiệu Quả Cao Nhất

```text
gpu_ram>=12 num_gpus=1 compute_cap>=860 gpu_name notin [A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,H200] dlperf>=22 dph<=0.22 verified=True reliability>0.97 cuda_max_good>=12.8 cpu_cores_effective>=4 cpu_ram>=16 pcie_bw>=5 inet_down>=200 inet_up>=100 disk_space>=45 direct_port_count>=1 geolocation notin [CN] rented=False
```
*(Sắp xếp theo `-o dph_total` để lấy giá/giờ thấp nhất, ưu tiên máy có chỉ số **`DLP/$ >= 350`**).*

**Lý do tinh chỉnh từng chỉ số theo quy trình mới:**
* **`gpu_ram>=12` + `compute_cap>=860` + `dlperf>=22` *(thay vì khóa cứng `RTX 3090/4090`)*:** Quy trình mới chạy tuần tự từng bước (`Mel-Band-RoFormer` ~4–6 GB VRAM, `CosyVoice`/`F5-TTS` ~4–8 GB VRAM, `NVENC` <1.5 GB VRAM) nên **12–16 GB VRAM** là dư sức, giúp giá thuê giảm một nửa xuống **`~$0.09–$0.10/giờ`**. `compute_cap>=860` loại bỏ card đời cũ (`V100`, `T4`, `P100`) thiếu `BF16` & `NVENC` chậm.
* **`cuda_max_good>=12.8`:** Khớp với `pyVideoTrans v4.19` (`torch==2.7.1 + cu128` trong `pyproject.toml`), hỗ trợ trọn vẹn cả dòng **RTX 50xx (Blackwell `sm_120`)** lẫn RTX 30xx/40xx.
* **`disk_space>=45`, `cpu_ram>=16`, `cpu_cores_effective>=4`, `inet_down>=200`, `inet_up>=100`, `direct_port_count>=1`:** Giảm ngưỡng xuống mức vừa đủ mượt mà không bị loại oan các máy giá siêu rẻ.
* **`geolocation notin [CN]`:** Tránh node Trung Quốc đại lục (GFW chặn GitHub / HuggingFace / Google Drive).

### Bước 4 — Dịch Đa Ngôn Ngữ Khóa Xưng Hô & Chia Vai
* Kết hợp `zh_hardsub.srt` + `HO_SO_NHAN_VAT.md` / `character_glossary.json` + nhãn Nam/Nữ đo từ `vocals` (+ kiểm tra khung hình ở các câu ngắn khó đoán) để dịch sang **`vi.srt`, `en.srt`, `ja.srt`, `ko.srt`**.
* Khóa chặt xưng hô đúng giới tính, vai vế xuyên suốt các tập; co gọn độ dài câu dịch vừa thời lượng nói để lồng tiếng tự nhiên, không bị tua nhanh.
* Gán vai đọc từng dòng vào `role_lines.json`.
* 🔍 **Điểm bạn duyệt #2:** Đọc lướt 30–50 dòng đầu và 1 đoạn giữa của `vi.srt` để chốt văn phong & xưng hô trước khi chạy lồng tiếng hàng loạt.

### Bước 5 — Lồng tiếng, Trộn BGM & Xuất Thành Phẩm YouTube *(Trên Vast.ai GPU)*
1. **Lồng tiếng đa vai (`VieNeu-TTS` + `pyVideoTrans` trên Vast.ai):**
   * **Tiếng Việt:** Dùng **`VieNeu-TTS` (`vieneu[cuda]`, 48 kHz)** phân 4 vai (Nam trẻ / Nam già / Nữ trẻ / Nữ già) hoặc clone giọng cho độ tự nhiên và chuẩn dấu thanh cao nhất.
   * **Anh / Nhật / Hàn:** Dùng **`pyVideoTrans v4.19`** (`CosyVoice` / `F5-TTS`) clone trực tiếp theo giọng gốc từng câu từ `vocals_48k_mono.flac` hoặc TTS phân vai tự nhiên.
2. **Trộn âm (`Mix`):** Ghép giọng lồng tiếng mới lên `instrumental` (tự động dìm nhẹ BGM khoảng `-10dB` tại các giây có lời thoại).
3. **Render Video Master (`FFmpeg h264_nvenc`):**
   * Áp dụng duy nhất bộ lọc làm mờ dải hardsub gốc (`blur_config.json`), **KHÔNG burn phụ đề mới vào video**.
   * Xuất kèm bộ **Audio tracks** (`audio_en.m4a`, `audio_ja.m4a`, `audio_ko.m4a`) và bộ **Softsub** (`subtitles/*.srt` / `.vtt` giữ thiết lập màu vàng `#FFEB50`, vị trí đè dải mờ nếu dùng định dạng hỗ trợ style).
4. **Đóng gói YouTube:** Tạo Thumbnail + Tiêu đề/Mô tả/Chapters đa ngôn ngữ, tải về máy và tắt ngay máy Vast.ai.
* 🔍 **Điểm bạn duyệt #3:** Nghe kiểm tra Đầu – Giữa – Cuối video thành phẩm và các track âm thanh đa ngôn ngữ trước khi đăng YouTube.

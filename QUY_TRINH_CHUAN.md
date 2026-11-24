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
  1. **Luồng Âm thanh (`audio-separator` + `Mel-Band-RoFormer` trên GPU Vast.ai):** Tách mới hoàn toàn từ video gốc bằng model **`melband_roformer_instvox_duality_v2.ckpt`** ra **Giọng nói sạch (`vocals_48k_mono.flac`)** và **Nhạc nền + Hiệu ứng (`instrumental_48k.flac`)** chuẩn `48kHz` Lossless (vừa nhẹ hơn `.wav` gần 4 lần, vừa giữ trọn 48kHz cho `VieNeu-TTS` và trộn BGM).
  2. **Luồng Chữ (`Đọc Hardsub gốc` trên Laptop, `scripts/step3_extract_hardsub.py`):** Cắt vùng đáy cố định đã đo ở Bước 2 (`fps=5.0`), bắt chữ trắng/vàng có viền đen (`IoU >= 0.65`, ghép `25 dải/tờ`) $\rightarrow$ đọc ra file `zh_hardsub_raw.srt` đúng 100% mặt chữ tác giả, không sai tên nhân vật đồng âm.
* **Khớp sóng âm (`Audio-Snap` & Phân vai bằng âm thanh, `scripts/step3_sync_and_gender.py`):**
  * Kéo mốc `start --> end` của `zh_hardsub_raw.srt` bám khít vào sóng âm thực tế của `vocals_48k_mono.flac` (`+-0.25s`) $\rightarrow$ xuất `zh_master_synced.srt`.
  * Gộp câu dài bị màn hình dọc cắt đôi (`merge_gap <= 0.16s`, tổng độ dài `<= 26` ký tự, không ngắt bởi dấu kết câu).
  * Phân biệt **Lời thoại** (đo cao độ `F0 Hz`: `< 175 Hz = NAM`, `>= 175 Hz = NU`) với **Chữ chú thích câm** (`CHU_THICH_CAM`, chỉ giữ softsub) và quét lớp 2 phát hiện mọi đoạn có tiếng nói bị sót (`missing_speech_gaps.json >= 0.30s`).

---

## 3. Bộ Lọc Thuê GPU Vast.ai Tối Ưu Giá Rẻ & Hiệu Quả Cao Nhất

```text
gpu_ram>=12 num_gpus=1 gpu_frac=1 compute_cap>=860 gpu_name notin [A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,H200] dlperf>=22 dph<=0.22 verified=True reliability>0.97 cuda_max_good>=12.8 cpu_cores_effective>=4 cpu_ram>=16 pcie_bw>=5 inet_down>=200 inet_up>=100 disk_space>=45 direct_port_count>=1 geolocation notin [CN] rented=False
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

### Bước 5 — Lồng tiếng, Trộn BGM & Xuất Thành Phẩm YouTube *(Trên Vast.ai GPU, `scripts/vast_step5_dub_and_render.py`)*
1. **Lồng tiếng (Chốt cố định không clone ngoại ngữ):**
   * **Tiếng Việt:** Dùng **`VieNeu-TTS` (`mode="v3turbo"`, 48 kHz CUDA)** chỉ dùng **2 giọng Nam – Nữ tự nhiên cố định**: toàn bộ vai Nam dùng **`Hải Đăng`**, toàn bộ vai Nữ dùng **`Trúc Ly`**.
   * **Anh / Nhật / Hàn:** Không clone; dùng **`edge-tts`** mỗi quốc gia **1 giọng Nữ bản địa tự nhiên**:
     * `en`: **`en-US-AvaMultilingualNeural`**
     * `ja`: **`ja-JP-NanamiNeural`**
     * `ko`: **`ko-KR-SunHiNeural`**
     * Căn chỉnh tốc độ trước khi đọc (`rate` tối đa `+32%`) kết hợp **`FFmpeg atempo`** (`1.55x–1.70x`) mượt, tuyệt đối không cắt cụt chữ giữa câu.
2. **Trộn âm (`Mix` — Tham số đã duyệt):**
   * **Chỉ tăng +30% âm lượng riêng của giọng nói nhân vật (`VOICE_GAIN = 1.30`, soft-limiter `knee=0.78, ceil=0.95`)** cho cả 4 ngôn ngữ.
   * **Giữ nguyên mức dìm BGM (`gain = 1.0 - 0.70 * smooth`, tương đương `-10.5 dB` khi có thoại)** trên `instrumental_48k.flac`.
   * **Giữ lại 92% âm lượng `vocals_48k_mono.flac` ở mọi khoảng ngoài mốc thời gian phụ đề** (cắt êm `~90ms` về `0%` đúng trong khung `[start, end]`) để bảo toàn nguyên vẹn tiếng reo hò, tiếng cười đám đông, tiếng hét võ đài.
   * Xuất từng track `audio_tracks/audio_{vi,en,ja,ko}.m4a` (`AAC 128k 48kHz Stereo`).
3. **Render Video Master (`FFmpeg hevc_nvenc` / `h264_nvenc`):**
   * Áp dụng duy nhất bộ lọc làm mờ dải hardsub gốc (`gblur=sigma=22` theo `blur_config.json`), **KHÔNG burn phụ đề mới vào video**.
   * Tham số nén tối ưu YouTube: `hevc_nvenc -rc vbr -cq 29 -b:v 700k -maxrate 1200k` (fallback `h264_nvenc -cq 28 -b:v 950k`), kèm `-movflags +faststart`.
   * Xuất kèm bộ **Audio tracks** (`audio_en.m4a`, `audio_ja.m4a`, `audio_ko.m4a`) và bộ **Softsub** (`subtitles/{vi,en,ja,ko}.srt` + `vi.vtt` style `#FFEB50`, nền `rgba(0,0,0,0.55)`).
4. **Đóng gói YouTube (`scripts/step5_youtube_package.py`):**
   * **Thumbnail (`1280×720`, `< 2 MB`):** Dùng `poster.jpg` gốc của phần đó đặt giữa nền blur đồng bộ từ chính poster, chỉ gắn **2 huy hiệu rõ nét ở góc trên: `HOT` và `P1`/`P2`**, không chèn chữ dài gây rối hay che mặt nhân vật.
   * **Tiêu đề (KHÔNG gộp 4 ngôn ngữ vào 1 tiêu đề):** Dùng mục **Ngôn ngữ (`Languages` / `localizations`)** trong YouTube Studio để đặt riêng 4 tiêu đề (`vi`, `en`, `ja`, `ko`) dịch sát tên gốc Trung Quốc (`<= 100 ký tự`, rõ số phần `Phần 1 / Part 1` ở cuối).
   * **Mô tả (Tách riêng 4 ngôn ngữ, chuẩn chính sách chống spam / Reused Content):**
     1. Nhập riêng từng bản mô tả vào tab ngôn ngữ tương ứng (`vi`, `en`, `ja`, `ko`), **không nhồi nhiều ngôn ngữ vào một ô mô tả**.
     2. Viết tóm tắt bối cảnh riêng của đúng phần `P1` hoặc `P2` bằng văn phong điện ảnh trung tính (tránh từ nhạy cảm bạo lực/máu me trực diện, tránh copy-paste y hệt 100% giữa các video, không rải link rác).
     3. Kèm **Mốc thời gian (`Chapters`)** trích từ kịch bản thật (`00:00` + 3–5 mốc cao trào chính) và giữ **tối đa 3 hashtags** sạch ở cuối.

### Bước 6 — Đồng bộ Google Drive, Tự Hủy GPU & Sleep Máy *(Hoàn toàn tự động)*
1. **Cấu trúc lưu trữ trên Google Drive (`rclone` -> `gdrive:MT-Video/`):**
   * **Mỗi phim 1 thư mục riêng:** `gdrive:MT-Video/<Mã_TênPhim>/`
   * **Mỗi phần 1 thư mục con:** `P1/`, `P2/`... Toàn bộ file thuộc phần nào lưu gọn trong thư mục phần đó gồm: video master `.mp4`, `audio_tracks/`, `subtitles/`, và `youtube_seo/` (gồm `thumb_1280x720.jpg`, `thumb_poster.jpg`, `metadata_4langs.json`, `youtube_*.txt`).
2. **Cơ chế tự hủy GPU an toàn (`Auto-Destroy Failsafe`):**
   * Bắt buộc bọc toàn bộ phiên chạy GPU trong khối `try...finally` phía máy điều khiển (và watchdog hẹn giờ tự dừng phía máy Vast.ai) để **tự động gọi `vastai destroy instance <ID>` ngay khi hoàn tất hoặc khi gặp bất kỳ lỗi nào**, không bao giờ để rò rỉ tiền thuê GPU.
3. **Tự động đưa máy về Sleep sau khi hoàn tất:**
   * Chỉ kích hoạt Sleep (`Standby S0`) sau khi đã xác nhận: **(a)** tải đủ thành phẩm lên Google Drive và **(b)** instance Vast.ai đã `destroyed` hoàn toàn.
* 🔍 **Điểm bạn duyệt #3:** Kiểm tra file trên Google Drive (video, 4 track âm thanh, phụ đề, thumbnail & SEO text) trước khi nạp vào YouTube Studio ở chế độ `Private`.

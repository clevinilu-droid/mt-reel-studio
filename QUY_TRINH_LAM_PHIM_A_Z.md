# QUY TRÌNH SẢN XUẤT PHIM AI ĐA NGÔN NGỮ TỪ A ĐẾN Z (MT REEL STUDIO v2)

> Tài liệu chuẩn vận hành thực tế toàn bộ dây chuyền làm phim ngắn/tổng hợp Trung Quốc dài 1–3 giờ: từ **săn nguồn tải về máy Local → đồng bộ Google Drive → thuê Cloud GPU Vast.ai → chạy Pipeline 7 bước AI (`s1 → s6`) → thu thành phẩm & hủy máy tiết kiệm chi phí → đăng YouTube Đa Ngôn Ngữ (`Multi-Language Audio`)**.

---

## MỤC LỤC NHANH
1. [Kiến trúc hệ thống & Quy hoạch thư mục](#1-kiến-trúc-hệ-thống--quy-hoạch-thư-mục)
2. [Giai đoạn A — Săn nguồn, Tải phim & Đặt tên Series](#2-giai-đoạn-a--săn-nguồn-tải-phim--đặt-tên-series)
3. [Giai đoạn B — Chuẩn bị trên máy Local & Đẩy Drive (Cách B)](#3-giai-đoạn-b--chuẩn-bị-trên-máy-local--đẩy-drive-cách-b)
4. [Giai đoạn C — Thuê GPU Vast.ai chuẩn & Kiểm tra `/api/diag`](#4-giai-đoạn-c--thuê-gpu-vastai-chuẩn--kiểm-tra-apidiag)
5. [Giai đoạn D — Chi tiết Pipeline 7 bước (`s1 → s6`)](#5-giai-đoạn-d--chi-tiết-pipeline-7-bước-s1--s6)
6. [Giai đoạn E — Thu thành phẩm, Hủy GPU & Làm thêm Audio Ngoại ngữ](#6-giai-đoạn-e--thu-thành-phẩm-hủy-gpu--làm-thêm-audio-ngoại-ngữ)
7. [Giai đoạn F — Hậu kiểm, Đăng YouTube & Đóng dấu `_END`](#7-giai-đoạn-f--hậu-kiểm-đăng-youtube--đóng-dấu-_end)
8. [Dự toán Thời gian, Chi phí & Xử lý sự cố](#8-dự-toán-thời-gian-chi-phí--xử-lý-sự-cố)

---

## 1. Kiến trúc hệ thống & Quy hoạch thư mục

### 1.1. Mô hình vận hành 2 tầng
* **Máy Local (Laptop Intel i5-11300H / Intel Iris Xe `h264_qsv`, không GPU rời):**
  * Tải phim nguồn, phân loại Series/Phần, chạy Web UI quản lý (`start_ui.bat` → `http://127.0.0.1:7860`).
  * Có thể chạy trước Bước 1 nhẹ (hoặc nén gửi thẳng gói **Cách B** lên Google Drive) và dùng giao diện để tìm/giám sát GPU Vast.ai.
* **Cloud GPU Vast.ai (`1-GPU riêng` RTX 3090 / RTX 4090 / A5000 / L40S, VRAM >= 24 GB):**
  * Chạy Docker image `:v2-app` tích hợp sẵn `jellyfin-ffmpeg7` (`h264_nvenc` tương thích mọi driver NVENC v12+), `NVIDIA_DRIVER_CAPABILITIES=all`, `cloudflared` tunnel, `rclone`, và **4 môi trường Python cô lập** (`_engine/config/pipeline_v2.json`):
    1. `asr_sep`: FunASR (`paraformer-zh`, `fsmn-vad`, `ct-punc`, `cam++`), Faster-Whisper, PyAV, BS-Roformer / MDX-Net.
    2. `vlm_llm`: RapidOCR CUDA/ONNX Runtime, `vLLM` phục vụ Google **Gemma 4 26B AWQ** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`).
    3. `tts_ui`: Web UI (`FastAPI`), **VieNeu-TTS `v3turbo`**, Qwen3-TTS fallback, FFmpeg render.
    4. `voxcpm`: **OpenBMB VoxCPM2** (`openbmb/VoxCPM2`) cho giọng ngoại ngữ chất lượng cao.

### 1.2. Cấu trúc thư mục chuẩn trên máy Local & Drive
```text
d:\MT-Box\MT-Video\
├── Download\                                   # Kho phim nguồn (chia theo Series/Phần)
│   ├── _SERIES_TEMPLATE.json
│   └── 001_<TenGocTQ>_<TenVietKhongDau>\
│       ├── _SERIES.json                        # ID nguồn, số phần TQ, shared_glossary, youtube_url
│       ├── P1\                                 # Phần 1 (xong & up YouTube -> đổi thành P1_END)
│       │   ├── <TenPhim>_合集.mp4              # Video gộp full
│       │   ├── poster.jpg / tvshow.nfo / 简介.txt
│       └── P2\                                 # Phần tiếp theo nếu TQ ra tiếp
├── _engine\
│   ├── config\pipeline_v2.json                 # Cấu hình venv & danh sách model AI
│   ├── config\vieneu_stock_cast.json           # Bộ 4 giọng Việt chuẩn (Hải Đăng, Phạm Tuyên, Trúc Ly, Ngọc Huyền)
│   ├── core\                                   # checkpoint.py, drive_sync.py, job_manager.py, vast_api.py, vram_guard.py
│   ├── pipeline\                               # s1_prepare.py, s2a_asr.py, s2b_ocr.py, s3_audit.py, s4_separate.py, s5_translate.py, s6_tts_render.py, voxcpm_worker.py
│   ├── scripts\scan_series.py                  # Tự động quét kho & cập nhật bảng quản lý phim
│   ├── ui\                                     # Giao diện Web Studio (:7860)
│   └── work\<Series>\<Part>\                   # Nơi lưu toàn bộ checkpoint từng bước (resume-safe)
├── branding\                                   # Avatar, Banner, MT_Reel_Strip_Watermark.png
├── output\<Series>\<Part>\                     # Thành phẩm cuối cùng
│   ├── final_vi.mp4                            # Video hoàn chỉnh 1080x1920 (đã blur sub gốc + watermark + soft sub)
│   ├── audio\audio_<lang>.m4a                  # Track audio lồng tiếng từng ngôn ngữ (mix trên clean_bed)
│   ├── subtitles\<lang>.srt & kiem_tra_dich.html
│   └── _info\                                  # 04_ref_male.wav, 04_ref_female.wav, glossary.json
├── docker\                                     # Dockerfile, Dockerfile.app, entrypoint.sh
├── docs\                                       # Tài liệu chiến lược, nguồn TQ, RPM, quản lý kênh/phim
├── DANH_SACH_PHIM.csv & DANH_SACH_PHIM.md      # Sổ theo dõi tiến độ toàn bộ series
└── start_ui.bat                                # Mở nhanh Web UI local (:7860)
```

---

## 2. Giai đoạn A — Săn nguồn, Tải phim & Đặt tên Series

1. **Tìm & tải phim nguồn (`docs/nguon_video_trung_quoc.md`, `docs/kho_san_phim_hot.md`):**
   * **Hồng Quả Đoạn Kịch (`红果`)**: Tải bản gộp tập bằng công cụ `hongguo-downloader-7.8-win-x64.exe` trong thư mục dự án (ưu tiên phim dài 1–3 giờ dạng hợp tập `合集`).
   * **Bilibili / YouTube TQ / Douyin**: Tải bản độ phân giải tối thiểu 1080p dọc (`9:16`), tránh file có watermark to đè giữa mặt hoặc bị cắt cụt sub dưới đáy.
2. **Đặt tên thư mục theo chuẩn (`docs/quan_ly_phim.md`):**
   * Tạo thư mục Series: `Download\<Mã3Số>_<TênGốcTQ>_<TênViệtKhôngDấu>\`
     * Ví dụ: `001_退役归来这个保镖不好惹_CuuDacNhiemLamVeSi\`
   * Tạo thư mục Phần: `P1\`, `P2\`, `P3\`... (chỉ thêm `_END` vào `P1_END` sau khi đã đăng YouTube; chỉ thêm `_END` vào tên Series ngoài cùng khi phía Trung Quốc đã hết hẳn toàn bộ mùa và bạn đã up đủ các phần).
   * Sao chép `Download\_SERIES_TEMPLATE.json` thành `_SERIES.json` trong thư mục Series, điền `source_id`, `series_ended_at_source`, `shared_glossary`.
3. **Cập nhật bảng quản lý phim:**
   ```powershell
   d:\MT-Box\MT-Video\.venv\Scripts\python.exe d:\MT-Box\MT-Video\_engine\scripts\scan_series.py
   ```
   Lệnh này tự quét `Download/`, `_engine/work/`, `output/` và ghi lại `DANH_SACH_PHIM.csv` + `DANH_SACH_PHIM.md`.

---

## 3. Giai đoạn B — Chuẩn bị trên máy Local & Đẩy Drive (Cách B)

1. **Mở Web UI Local:** Nhấp đúp [`start_ui.bat`](start_ui.bat) (hoặc chạy `_engine/run_ui.py --host 127.0.0.1 --port 7860`).
2. **Chọn phim & thông số Job:**
   * Chọn file từ danh sách `Download/` (Web UI tự suy ra `work_rel = <Series>/<Part>` và tính vân tay file `compute_source_fingerprint` gồm kích thước + SHA256 64KB đầu/cuối để không bao giờ chạy lại nhầm nếu file đã làm xong từ trước).
   * **Chế độ chạy (`mode`):**
     * `test_5min`: Chỉ cắt 300 giây đầu (`work/.../_test_5min/`) để kiểm tra chất lượng khung mờ, dịch, giọng đọc trước khi chạy Full (~5–8 phút).
     * `full`: Chạy trọn bộ thời lượng phim.
   * **Chế độ giọng (`voice_mode`):**
     * Hệ thống cố định dùng **`single_female` (1 giọng Nữ chuẩn, KHÔNG clone giọng gốc, KHÔNG dùng giọng nam)**: Tiếng Việt dùng giọng **`Trúc Ly`** (VieNeu `v3turbo`), ngoại ngữ dùng 1 giọng nữ truyền cảm tạo bằng `VoxCPM2 Voice Design`, kết hợp khóa xưng hô chặt chẽ trong `relationship_bible.json` để phân biệt rõ vai vế.
   * **Ngôn ngữ (`langs`):**
     * Khuyên chạy **`["vi"]`** trước để xuất `final_vi.mp4` nhanh và rẻ nhất.
     * Muốn xuất thêm ngoại ngữ (`en`, `ja`, `ko`, `es`): chọn thêm ở lần chạy sau trên cùng `work_dir` (không cần chạy lại hình/OCR/tách nhạc).
3. **Đẩy gói sang Google Drive theo Cách B (`_engine/core/drive_sync.py`):**
   * Thay vì đẩy file `01_cfr_raw.mp4` nặng 3–4 GB vừa tốn thời gian vừa thừa, **Cách B** chỉ đóng gói 3 phần tinh gọn lên `gdrive:MT-Video/<Series>/<Part>/`:
     1. `source.mp4` (video gốc 1 đời nén).
     2. `audio_48k.flac` (audio 48kHz stereo nén lossless từ `01_orig_48k_stereo.wav`, giảm ~50% dung lượng mà không mất 1 bit chất lượng).
     3. `step1_audit.json` & `manifest.json` (hồ sơ kiểm tra + checkpoint).

---

## 4. Giai đoạn C — Thuê GPU Vast.ai chuẩn & Kiểm tra `/api/diag`

### 4.1. Bộ lọc phần cứng bắt buộc (`_engine/core/vast_api.py`)
Mọi máy thuê qua API/Web UI hoặc lọc tay trên Vast.ai **phải đạt đủ toàn bộ tiêu chí sau** để không gặp lỗi crash giữa đường:
* **`num_gpus == 1` VÀ `gpu_frac == 1.0` (1-GPU riêng biệt, CỰC KỲ QUAN TRỌNG):**
  * Các máy 2–8 GPU cho thuê lẻ 1 card (`gpu_frac < 1.0`) thường gán container vào GPU slot `!= 0`. Trên driver NVIDIA >= 565, FFmpeg NVENC gọi `OpenEncodeSessionEx` bị lỗi `unsupported device (2)`, khiến render rớt về CPU chậm gấp 5 lần!
* **Kiến trúc GPU (`compute_cap` `860–890`):** RTX 3090 / 3090 Ti / RTX 4090 / RTX A5000 / A6000 / L40S (`>= 24 GB VRAM`).
  * *Loại bỏ:* `A100 / A40 / H100` (không có chip phần cứng NVENC), `RTX 5090 / Blackwell` (chưa tương thích `torch cu124` trong container), máy đời cũ `< 24 GB VRAM`.
* **Môi trường & Phần cứng đi kèm:**
  * `cpu_arch == "amd64"`, `cuda_max_good >= 13.0` (để chạy mượt `vLLM` + `ONNX CUDA`).
  * Mạng `inet_down >= 800 Mbps` & `inet_up >= 800 Mbps` (kéo model Gemma 4 + Drive nhanh).
  * `cpu_cores_effective >= 6`, `cpu_ram >= 24 GB`, `pcie_bw >= 8.0 GB/s`.
  * Ổ NVMe: mặc định **`70 GB`** (`disk_bw >= 400 MB/s`) là đủ cho `~25 GB` model weights + `~15–20 GB` dữ liệu tạm cho phim 2–3 giờ (giảm phí storage so với 120 GB trước đây).
  * `reliability >= 98.5%`, **chặn vùng `CN, RU, IR, BY, KP`** (vì bóp/chặn HuggingFace & Google Drive).
* **Template / Image khi thuê:** Mở cổng `-p 7860:7860`, bật `NVIDIA_DRIVER_CAPABILITIES=all`, `HF_HUB_ENABLE_HF_TRANSFER=1`, đính kèm `RCLONE_CONF_B64` (chứa cấu hình `gdrive:` từ máy local).

### 4.2. Khởi động & Chẩn đoán tự động (`/api/diag`)
1. Khi container bật, `docker/entrypoint.sh`:
   * Tự kéo bản `_engine/` và `branding/` mới nhất từ GitHub `main` (`~0.5 giây`, sửa code Python không cần rebuild Docker image).
   * Tự giải mã `RCLONE_CONF_B64` ra `/root/.config/rclone/rclone.conf`.
   * Quét toàn bộ thư viện `nvidia/*/lib` & `torch/lib` trong `/opt/venvs/*` đưa vào `LD_LIBRARY_PATH` để `ONNX Runtime` nhận GPU CUDA.
   * Mở Cloudflare Quick Tunnel (`_tunnel_url.txt`) + chạy Web UI ở cổng `7860`.
2. **Kiểm tra chẩn đoán trước khi bấm chạy (`GET http://<IP>:<PORT>/api/diag`):**
   * Phải thấy `nvenc.selected_encoder == "h264_nvenc"` (`nvenc_probe_rc == 0`).
   * Phải thấy `CUDAExecutionProvider` trong `onnx.providers`.
   * Phải thấy `drive_ready == true` trên Web UI (`GET /api/snapshot`).
3. **Kéo gói phim về máy GPU (`POST /api/drive/pull`):** `rclone copy` tải `source.mp4` + `audio_48k.flac` từ `gdrive:MT-Video/<Series>/<Part>/` và giải nén ra `01_orig_48k_stereo.wav` + `01_orig_16k_ai.wav` trong vài chục giây.

---

## 5. Giai đoạn D — Chi tiết Pipeline 7 bước (`s1 → s6`)

`JobManager` điều phối tuần tự `s1 → s2a → s2b → s3 → s4 → s5 → s6`. Mỗi bước nặng chạy trong subprocess thuộc venv riêng (`run_isolated_step`), xả VRAM triệt để trước khi sang bước kế tiếp và lưu checkpoint atomic để bấm **Pause/Resume** bất kỳ lúc nào mà không mất dữ liệu đã làm.

### Bước 1 (`s1_prepare.py`) — Chuẩn hóa Video CFR & Tách Audio Gốc
* Dùng `FFmpeg` (ưu tiên `h264_nvenc` `cq=19` trên Vast, hoặc `h264_qsv` / `libx264` trên local) chuẩn hóa video về khung chuẩn **`1080x1920`, 30 fps CFR** (`01_cfr_raw.mp4`).
* Tách cùng lúc 2 file âm thanh chuẩn:
  * `01_orig_48k_stereo.wav`: âm thanh 48kHz stereo gốc dùng cho tách nhạc và làm nền sạch.
  * `01_orig_16k_ai.wav`: âm thanh 16kHz mono lọc `highpass=80Hz` + chuẩn hóa `dynaudnorm` cho AI nghe thoại.
* **Spot-check kiểm tra lệch tiếng-hình:** Lấy 3 điểm đầu/giữa/cuối để đo `timeline_match_pct` và `median_ocr_minus_asr_offset_s` → xuất `step1_audit.json`.
  * *Tốc độ thực tế với phim `2h31m` (`9085s`) trên RTX 3090 (`h264_nvenc`):* **~10.5 phút** (~14.5x real-time, nhanh gấp ~5x CPU).

### Bước 2A (`s2a_asr.py`, venv `asr_sep`) — Nghe Thoại & Gom Người Nói
* Dùng **FunASR** (`paraformer-zh` + `fsmn-vad` cắt khoảng lặng + `ct-punc` chấm câu + `cam++` nhận diện `spk_id` từng nhân vật nói), tự động fallback sang `Faster-Whisper` (`large-v3-turbo`) nếu cần.
* Lưu micro-checkpoint từng khúc **5 phút** vào `02a_chunks.jsonl` → tổng hợp thành `02a_audio_asr.json`.

### Bước 2B (`s2b_ocr.py`, venv `vlm_llm`) — Cắt Keyframe Thông Minh & Đọc Sub Cứng
* Thay vì đọc mù 30 khung/giây, thuật toán cắt keyframe dẫn hướng theo âm thanh (`speech_fps = 4.0`, đoạn lặng `idle_fps = 1.2`, ngưỡng lọc khung giống nhau `diff_thresh = 9.5`).
* Chỉ crop vùng 1/3 dưới màn hình chứa phụ đề Trung gốc, chạy **RapidOCR CUDA** (`CUDAExecutionProvider`, có self-test tự fallback CPU nếu thiếu lib) → xuất `02b_frames_ocr.json`.

### Bước 3 (`s3_audit.py`, venv `asr_sep`) — Đối Chiếu Chéo Pinyin & Cứu Hộ Câu Sót
* Dùng thư viện `pypinyin` so khớp đồng âm giữa **Chữ OCR (hình)** ∩ **Chữ ASR (tiếng)** và phân loại chuẩn 4 trường hợp:
  1. `MATCHED`: Lấy **mặt chữ chuẩn** từ OCR + **mốc giờ mở/dứt miệng chuẩn** từ Audio VAD/ASR.
  2. `OCR_RESCUED`: Nếu Audio có tiếng nhưng OCR trống hoặc cụt `< 60%`, tự dùng `PyAV` cắt dày `8 fps` đúng đoạn đó đọc lại; nếu màn hình không có chữ thì bù bằng text ASR để không bao giờ hở tiếng Trung.
  3. `SCREEN_TEXT_ONLY`: Chữ tiêu đề/chú thích trên màn hình không có tiếng người nói → tách sang `screen_captions`, không đọc đè.
  4. `CROWD_OR_NOISE`: Tiếng reo hò, va chạm không phải cấu trúc thoại → không xóa giọng nền.
* **Đầu ra:** `03_aligned_master.json`, `03_zh_master.srt`, `03_audit_flags.json`. *(Chạy chỉ ~5 giây).*

### Bước 4 (`s4_separate.py`, venv `asr_sep`) — Tách Giọng Có Chọn Lọc Tạo Nền Sạch
* Gom các câu thoại gần nhau (`pad = 0.35s`, `gap_merge = 1.8s`) thành các `dialogue blocks` và lưu cache từng block `blk_XXXX_inst.wav` trong `sep_blocks/` (xóa ngay file vocal sau khi tách để tiết kiệm 50% dung lượng).
* **Chỉ chạy mô hình tách giọng `BS-Roformer` (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`, fallback `UVR-MDX-NET-Inst_HQ_3.onnx`) trên các block có thoại**; mọi khoảng ngoài block thoại chép thẳng nguyên bản `01_orig_48k_stereo.wav`, sau đó crossfade mượt `120ms` ở 2 mép → tạo `04_clean_bed.wav` giữ trọn 100% không gian phim & hiệu ứng âm thanh gốc (bỏ hoàn toàn phần cắt mẫu clone nam/nữ).

### Bước 5 (`s5_translate.py`, venv `vlm_llm`) — Phân Vai, Khóa Xưng Hô & Dịch Đa Ngôn Ngữ
Khởi chạy `vLLM` nạp **đúng 1 lần** **`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`** chạy nối tiếp 3 pha (cache từng batch vào `.jsonl`, ngắt ngang chạy lại không mất):
1. **Pha 5A — Phân vai & Sổ Quan Hệ (`05a_roles.jsonl`, `glossary.json`, `relationship_bible.json`):**
   * Xác định bối cảnh (`MODERN` vs `ANCIENT`), tên riêng nhân vật/địa danh (`glossary.json` — có thể kế thừa từ Phần 1 sang Phần 2/3), giới tính, độ tuổi, cảm xúc (`NEUTRAL / ANGRY / HAPPY / SAD / WHISPER / URGENT`).
   * Tạo **Sổ Quan Hệ (`relationship_bible.json`)** khóa cứng cách xưng hô giữa từng cặp `speaker -> listener` (cả tiếng Việt `vi_self/vi_other` lẫn quy tắc kính ngữ cho `ko_level`, `ja_level`, `es_pronoun`).
   * *Chặn cứng lỗi:* Trong phim `MODERN` cấm `'ta/ngươi/tại hạ'`; cấm nhân vật nữ tự xưng `'anh'`.
2. **Pha 5B & 5B.2 — Dịch Tiếng Việt & Rà Soát Cảnh (`05b_sub_vi.jsonl`, `tracks/vi.srt`):**
   * Dịch khẩu ngữ Việt tự nhiên, cân đối độ dài khớp `dur_s` của từng câu (`batch_size = 30`).
   * **Pha 5B.2** đọc lại cửa sổ 40 câu liên tiếp để sửa ngay câu nào bị nhảy xưng hô giữa chừng hoặc sai chủ ngữ.
3. **Pha 5C & QA Dịch Ngược *(chỉ chạy khi có ngoại ngữ ngoài `vi`)*:**
   * Dịch lần lượt từng ngôn ngữ (`en, ja, ko, es`, `batch_size = 15`) dựa trên `Relationship Bible`, ngữ cảnh 6 câu trước đó, và `vi_ref`.
   * Cuối cùng `_build_qa_html_and_backtrans` dịch ngược toàn bộ ngoại ngữ về lại Tiếng Việt để tạo bảng đối chiếu **`tracks/kiem_tra_dich.html`**.

### Bước 6 (`s6_tts_render.py` & `voxcpm_worker.py`) — Lồng Tiếng, Mix Streaming & Render MP4
1. **6A — Tiếng Việt (`VieNeu TTS v3turbo`, venv `tts_ui`):**
   * Chế độ `single_female` dùng giọng **`Trúc Ly`** (hoặc tự map 4 giọng nếu `stock_4`), chạy `infer_batch` **32 câu/lượt** trên GPU, lưu cache từng câu vào `tts_cache/vi/turn_XXXX_<hash>.wav`.
   * `_mix_track_streaming` chèn các câu đọc (tự canh tốc độ nếu hơi dài hơn slot) lên `04_clean_bed.wav` (tự giảm âm nền vừa đủ khi có thoại) → xuất `output/.../audio/audio_vi.m4a` và `subtitles/vi.srt`.
2. **6B — Ngoại ngữ (`VoxCPM2`, venv `voxcpm`):**
   * Nạp 1 lần `openbmb/VoxCPM2` (48kHz) chạy tuần tự từng ngôn ngữ (`en, ja, ko, es`). Nếu `non-clone` sẽ tạo 1 mẫu anchor Voice Design chuẩn cho từng ngôn ngữ để khóa âm sắc đồng nhất cả phim; tự fallback `Qwen3-TTS-1.7B` nếu cần. Xuất `audio/audio_<lang>.m4a` & `subtitles/<lang>.srt`.
3. **6C — Render Video Thành Phẩm (`final_vi.mp4`):**
   * `_detect_sub_blur_rect` quét tự động chiều cao dòng phụ đề Trung gốc trên `01_cfr_raw.mp4`, lấy dải che **94% chiều ngang**, làm mờ bằng `boxblur`.
   * Đè logo [`branding/MT_Reel_Strip_Watermark.png`](branding/MT_Reel_Strip_Watermark.png) với độ mờ **`10%` (`target_opacity=0.10`)** lên giữa dải mờ.
   * Gắn luồng audio chính + soft sub `.srt` (`mov_text`) và render bằng **`h264_nvenc`** ra **`final_vi.mp4`**.
   * *(Nếu chạy chế độ `test_5min` và có nhiều ngôn ngữ: tự động nhân bản nhanh ra `preview_5min_<lang>.mp4` bằng `-c:v copy -c:a copy` trong vài giây để nghe kiểm tra ngay).*

---

## 6. Giai đoạn E — Thu thành phẩm, Hủy GPU & Làm thêm Audio Ngoại ngữ

### 6.1. Thu hoạch & Giải phóng máy ngay khi xong Bước 6
1. **Đẩy toàn bộ Output lên Google Drive:**
   * Gọi nút trên Web UI hoặc `POST http://<IP>:<PORT>/api/jobs/<job_id>/push-output` để `rclone copy` đẩy toàn bộ `output/<Series>/<Part>/` lên `gdrive:MT-Video/<Series>/<Part>/output/`.
2. **Lưu giữ checkpoint nhẹ để sau này làm thêm ngoại ngữ KHÔNG phải chạy lại `s1 → s5b`:**
   * Tải về máy hoặc đồng bộ lên Drive các file nhẹ cực kỳ giá trị trong `work/<Series>/<Part>/`:
     * `03_aligned_master.json` (kịch bản & timeline chốt)
     * `04_clean_bed.wav` (hoặc nén `.flac` cho nhẹ)
     * `glossary.json`, `relationship_bible.json`, `05a_roles.jsonl`, `05b_sub_vi.jsonl`, `tracks/vi.srt`
3. **HỦY MÁY VAST NGAY LẬP TỨC (`Destroy`):**
   * Sau khi kiểm tra Drive đã có đủ `final_vi.mp4` + `audio_vi.m4a` + `vi.srt`, bấm **Destroy** trên Web UI (hoặc gọi `destroy_instance`) và kiểm tra danh sách máy đang thuê trống hoàn toàn để **cắt 100% tiền thuê theo giờ**.

### 6.2. Khi cần làm thêm Audio ngoại ngữ (`EN, JA, KO, ES`) về sau
* **Không render lại video!** Chỉ cần khôi phục thư mục `work/` ở bước trên vào máy GPU mới, chọn giữ cache `s1–s4` và Pha `5A–5B`, chạy **Pha 5C (dịch) + Bước 6B (đọc VoxCPM2 & mix)** để lấy thêm `audio_<lang>.m4a` + `<lang>.srt`.
* Nếu chỉ cần làm nhanh **1 ngôn ngữ** (ví dụ `en` cho thị trường Tier-1 Mỹ/Úc/Anh/Canada/Singapore), thời gian rút ngắn còn **1/4** (~30–45 phút). Nếu muốn gắn sẵn thành MP4 riêng, dùng `-c:v copy -c:a copy` từ `final_vi.mp4` chỉ mất vài chục giây/file.

---

## 7. Giai đoạn F — Hậu kiểm, Đăng YouTube & Đóng dấu `_END`

1. **Hậu kiểm chất lượng (5–10 phút):**
   * Tải `final_vi.mp4` (hoặc `audio_<lang>.m4a`, `kiem_tra_dich.html`) từ Drive về máy Local.
   * Nghe lướt vài đoạn đầu/giữa/cuối: kiểm tra tiếng đọc rõ, âm nền SFX không bị méo, xưng hô đúng vai vế, dải blur che hết sub Trung mà không lẹm lên mặt diễn viên.
2. **Đăng YouTube theo Phương án A — 1 Kênh Đa Ngôn Ngữ (`docs/quy_trinh_az_va_chinh_sach_2026.md`):**
   * Đăng `final_vi.mp4` lên kênh chính.
   * **Bắt buộc tích khai báo AI** (*Altered or synthetic content*) nếu có yếu tố tạo sinh — YouTube không bóp đề xuất khi khai báo trung thực.
   * Trong **YouTube Studio → Subtitles & Audio**:
     * Tải file phụ đề `.srt` theo từng ngôn ngữ + điền Tiêu đề/Mô tả bản địa hóa (`Localizations`).
     * Thêm các track **`audio_en.m4a`**, **`audio_ja.m4a`**, **`audio_ko.m4a`** vào mục *Audio tracks* của cùng một video để gom toàn bộ lượt xem quốc tế vào một chỗ.
   * Gắn video vào Playlist riêng của Series và ghim bình luận dẫn xem tiếp phần sau (nếu có).
3. **Đóng dấu hoàn tất trong kho (`docs/quan_ly_phim.md`):**
   * Đổi tên thư mục trên máy Local từ `P1/` → **`P1_END/`**.
   * Cập nhật `youtube_url` và `status: "END"` trong `_SERIES.json`.
   * Chạy lại `scan_series.py` để cập nhật `DANH_SACH_PHIM.csv` & `DANH_SACH_PHIM.md`.
   * Xóa bớt các file trung gian hàng GB không cần giữ lâu (như `01_cfr_raw.mp4`, `sep_blocks/` nếu không cần tách lại), giữ lại `glossary.json` & `relationship_bible.json` cho Phần 2 (`P2`).

---

## 8. Dự toán Thời gian, Chi phí & Xử lý sự cố

### 8.1. Bảng ước tính cho phim dài ~2.5 giờ (`9085s`) trên Vast RTX 3090 (~`$0.30–$0.37/h`)
| Gói chạy | Khối lượng xử lý | Thời gian ước tính | Chi phí GPU ước tính |
| :--- | :--- | :---: | :---: |
| **Test 5 phút (`test_5min`, 1–5 ngôn ngữ)** | Kiểm tra tổng thể blur/giọng/dịch trên 300s đầu | **~5 – 10 phút** | **~\$0.05** |
| **Full VI-only (`vi`, `single_female`)** | Trọn bộ `s1 → s6`, xuất `final_vi.mp4` + `audio_vi.m4a` + `vi.srt` | **~1 giờ 30 phút – 1 giờ 55 phút** | **~\$0.55 – \$0.75** |
| **Làm thêm 1 audio ngoại ngữ (từ cache)** | Chạy Pha 5C + VoxCPM2 + mix 1 ngôn ngữ (ví dụ `en`) | **~30 – 45 phút** | **~\$0.20 – \$0.30** |
| **Làm thêm đủ 4 audio ngoại ngữ + QA HTML** | Dịch 4 tiếng + dịch ngược QA + VoxCPM2 tuần tự từng câu | **~2.5 – 4 giờ** | **~\$0.80 – \$1.40** |

### 8.2. Checklist xử lý sự cố thường gặp
| Hiện tượng | Nguyên nhân gốc | Cách xử lý chuẩn |
| :--- | :--- | :--- |
| `h264_nvenc` báo `OpenEncodeSessionEx failed: unsupported device (2)` | Thuê nhầm máy nhiều GPU chia sẻ (`gpu_frac < 1.0`, container map GPU != 0) | Luôn dùng bộ lọc **1-GPU riêng (`gpu_frac == 1.0`)** và kiểm tra `/api/diag` trước khi chạy |
| `s2b_ocr` chạy chậm bất thường (~100% CPU) | Thiếu `LD_LIBRARY_PATH` NVIDIA libs cho ONNX Runtime | Kiểm tra `onnx.providers` qua `/api/diag`; container `:v2-app` đã tự quét toàn bộ venv trong `entrypoint.sh` |
| Crash vLLM / CUDA OOM | Chạy nhầm card `< 24 GB VRAM` hoặc `cuda_max_good < 13.0` | Chọn đúng GPU 24GB+ (`compute_cap 860–890`, `CUDA >= 13.0`); nếu dừng ngang chỉ cần bấm **Resume** |
| Phần 2 (`P2`) dịch sai tên nhân vật so với Phần 1 | Chạy `s5_translate.py` độc lập không truyền từ điển P1 | Truyền file `glossary.json` của `P1` vào `P2` (hoặc lưu trong `shared_glossary`) |
| Quên hủy máy Vast sau khi xong | Tốn tiền giờ lưu trữ & GPU liên tục | Luôn đẩy Output lên Drive xong → **Destroy** máy → xác nhận danh sách máy thuê trống |

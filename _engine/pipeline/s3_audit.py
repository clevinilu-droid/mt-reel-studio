"""Bước 3 (s3_audit): Đối chiếu chéo Hình (OCR) ∩ Tiếng (ASR) theo Pinyin & Cứu hộ câu cụt/sót.

Logic chuẩn 4 trường hợp:
  1. MATCHED (Audio + OCR khớp Pinyin): Giữ mặt chữ chuẩn của OCR, chốt mốc giờ mở/dứt miệng theo Audio VAD/ASR.
  2. OCR_RESCUED (Audio có thoại nhưng OCR trống hoặc cụt <60%): Tự dùng PyAV cắt dày 8 fps đúng đoạn đó OCR lại;
     nếu màn hình thật sự không có chữ thì bù bằng text ASR để không bao giờ thiếu câu / hở tiếng Trung.
  3. SCREEN_TEXT_ONLY (OCR có chữ nhưng Audio im lặng): Tách sang `screen_captions`, không lồng tiếng đè.
  4. CROWD_OR_NOISE (Tiếng reo hò không cấu trúc thoại): Không đưa vào danh sách xóa giọng.

Đầu ra:
  - `03_aligned_master.json` (kịch bản chuẩn tiếng Trung + timeline chốt + spk_id)
  - `03_zh_master.srt`
  - `03_audit_flags.json` (báo cáo số câu khớp, đã cứu hộ, chữ màn hình loại bỏ)
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np

from _engine.core.checkpoint import atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, patch_onnx_threads

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

try:
    from pypinyin import Style, lazy_pinyin

    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False


def clean_cjk(text: str) -> str:
    return "".join(CJK_RE.findall(text or ""))


def to_pinyin_key(text: str) -> str:
    """Chuyển chữ Hán sang chuỗi Pinyin không dấu để so khớp chính xác kể cả khi ASR nghe sai đồng âm."""
    ck = clean_cjk(text)
    if not ck:
        return ""
    if _HAS_PYPINYIN:
        return " ".join(lazy_pinyin(ck, style=Style.NORMAL))
    return ck


def pinyin_similarity(a: str, b: str) -> float:
    pa, pb = to_pinyin_key(a), to_pinyin_key(b)
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0
    return difflib.SequenceMatcher(None, pa, pb).ratio()


def format_srt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    if ms == 1000:
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_ass_ts(sec: float) -> str:
    sec = max(0.0, sec)
    cs = int(round((sec - int(sec)) * 100))
    total = int(sec)
    if cs == 100:
        total += 1
        cs = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass_text(turns: list[dict[str, Any]], width: int = 1080, height: int = 1920) -> str:
    """Tạo file phụ đề .ass chuẩn kích thước bản gốc, căn giữa và tự xuống dòng không tràn khung."""
    font_sz = max(38, int(round(height * 0.025)))  # ~48px trên 1080x1920 (cỡ chữ chuẩn bản gốc)
    margin_v = max(120, int(round(height * 0.145)))  # ~278px từ mép dưới, nằm ngay trên vùng sub gốc
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Microsoft YaHei,{font_sz},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,2,60,60,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{format_ass_ts(t['start'])},{format_ass_ts(t['end'])},Default,,0,0,0,,{t['text']}"
        for t in turns
    ]
    return header + "\n".join(events) + "\n"


def _tighten_words_span(a: dict[str, Any]) -> tuple[float, float]:
    """Cắt bỏ khoảng lặng >1.2s bị Whisper kéo dãn theo danh sách words."""
    words = a.get("words") or []
    if len(words) < 2:
        return float(a["start"]), float(a["end"])
    clusters: list[list[dict[str, Any]]] = [[words[0]]]
    for w in words[1:]:
        if float(w["s"]) - float(clusters[-1][-1]["e"]) >= 1.2:
            clusters.append([w])
        else:
            clusters[-1].append(w)
    main_cl = max(clusters, key=lambda cl: (len("".join(x["w"] for x in cl)), float(cl[-1]["e"]) - float(cl[0]["s"])))
    s = float(main_cl[0]["s"])
    e = max(s + 0.15, float(main_cl[-1]["e"]))
    return round(s, 3), round(e, 3)


def _measure_voice_in_wav(wf: wave.Wave_read | None, start_s: float, end_s: float) -> tuple[float, int, float | None]:
    """Đo năng lượng RMS và số frame có dao động thanh quản F0 để xác nhận đoạn OCR có tiếng người thật."""
    if wf is None:
        return 0.0, 0, None
    sr = wf.getframerate()
    n_tot = wf.getnframes()
    i0 = max(0, int(round((start_s - 0.15) * sr)))
    i1 = min(n_tot, int(round((end_s + 0.15) * sr)))
    if i1 <= i0:
        return 0.0, 0, None
    wf.setpos(i0)
    arr = np.frombuffer(wf.readframes(i1 - i0), dtype=np.int16).astype(np.float32) / 32768.0
    if wf.getnchannels() > 1:
        arr = arr.reshape(-1, wf.getnchannels()).mean(axis=1)
    rms = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0

    win, hop = int(0.04 * sr), int(0.015 * sr)
    min_l, max_l = int(sr / 380), int(sr / 75)
    f0s: list[float] = []
    for k in range(0, max(0, len(arr) - win), hop):
        seg = arr[k : k + win]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.03:
            continue
        seg = seg - np.mean(seg)
        ac = np.correlate(seg, seg, mode="full")[win - 1 :]
        if ac[0] <= 0:
            continue
        pk = int(np.argmax(ac[min_l:max_l]) + min_l)
        if ac[pk] > 0.42 * ac[0]:
            f0s.append(sr / pk)
    med_f0 = round(float(np.median(f0s)), 1) if f0s else None
    return round(rms, 4), len(f0s), med_f0


def rescue_ocr_window(cfr_mp4: Path, ocr_engine, start_s: float, end_s: float, fps: float = 8.0) -> str:
    """Quét OCR dày 8 fps cục bộ riêng cho 1 đoạn câu thoại bị nghi cụt/sót chữ."""
    container = av.open(str(cfr_mp4))
    stream = container.streams.video[0]
    time_base = float(stream.time_base)
    win_s = max(0.0, start_s - 0.35)
    win_e = end_s + 0.35
    best_txt = ""
    best_len = 0

    try:
        seek_pts = int(win_s / time_base)
        container.seek(max(0, seek_pts), stream=stream, backward=True, any_frame=False)
        last_t = -999.0
        min_dt = 1.0 / fps
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = frame.pts * time_base
            if t < win_s:
                continue
            if t > win_e:
                break
            if t - last_t < min_dt:
                continue
            last_t = t

            bgr = frame.to_ndarray(format="bgr24")
            h = bgr.shape[0]
            strip = bgr[int(h * 0.54) : int(h * 0.95), :, :]
            res, _ = ocr_engine(strip)
            if not res:
                continue
            parts = [r[1].strip() for r in res if float(r[2]) >= 0.50 and clean_cjk(r[1])]
            cand = " ".join(parts)
            clen = len(clean_cjk(cand))
            if clen > best_len:
                best_len = clen
                best_txt = cand
    except Exception:
        pass
    finally:
        container.close()
    return best_txt


def run_step3(target_dir: Path, force: bool = False) -> dict[str, Any]:
    out_master = target_dir / "03_aligned_master.json"
    out_srt = target_dir / "03_zh_master.srt"
    out_flags = target_dir / "03_audit_flags.json"

    asr_path = target_dir / "02a_audio_asr.json"
    ocr_path = target_dir / "02b_ocr_raw.json"
    cfr_mp4 = target_dir / "01_cfr_raw.mp4"

    if not force:
        existing = read_json_safe(out_master)
        if existing and existing.get("turns"):
            emit_progress(100.0, f"Tái sử dụng checkpoint Bước 3 ({len(existing['turns'])} câu chuẩn)")
            return existing

    asr_data = read_json_safe(asr_path)
    ocr_data = read_json_safe(ocr_path)
    if not asr_data or not ocr_data:
        raise FileNotFoundError("Thiếu kết quả 02a_audio_asr.json hoặc 02b_ocr_raw.json để chạy Bước 3.")

    t0 = time.perf_counter()
    asr_segs = asr_data.get("segments", [])
    ocr_segs = ocr_data.get("segments", [])
    used_ocr_ids: set[int] = set()

    emit_progress(10.0, f"Đang đối chiếu {len(asr_segs)} câu ASR ↔ {len(ocr_segs)} dòng Hard-sub (Pinyin={_HAS_PYPINYIN})...")

    candidates: list[dict[str, Any]] = []
    screen_captions: list[dict[str, Any]] = []
    rescued_ocr = 0
    fallback_asr = 0
    ocr_promoted = 0

    if len(ocr_segs) >= 3:
        # CHẾ ĐỘ CHUẨN HARDSUB-FIRST:
        # Hard-sub hiển thị từng câu trên màn hình là chuẩn gốc cả về nội dung lẫn timeline [start, end].
        # ASR chỉ dùng để gắn `spk_id` (nếu có) và sửa câu OCR bị nhìn nhầm khi trùng độ dài & Pinyin cao.
        wav_16k = target_dir / "01_orig_16k_ai.wav"
        wf_16k = wave.open(str(wav_16k), "rb") if wav_16k.exists() else None
        try:
            for o in ocr_segs:
                os_t = float(o["start"])
                oe_t = max(os_t + 0.18, float(o["end"]))
                o_txt = str(o["text"]).strip()
                o_ck = clean_cjk(o_txt)
                o_conf = float(o.get("conf", 0.8) or 0.8)
                if not o_ck:
                    continue

                # Tìm đoạn ASR gần nhất (không dùng ASR gom nhiều câu để thay thế)
                spk_id = "SPK_UNK"
                best_asr_txt = ""
                best_sim = 0.0
                for a in asr_segs:
                    a_s, a_e = _tighten_words_span(a)
                    if a_e < os_t - 1.5 or a_s > oe_t + 1.5:
                        continue
                    spk_cand = str(a.get("spk_id") or "SPK_UNK")
                    if spk_cand != "SPK_UNK":
                        spk_id = spk_cand
                    a_txt = str(a.get("text") or "").strip()
                    a_ck = clean_cjk(a_txt)
                    sim = pinyin_similarity(a_txt, o_txt)
                    if sim > best_sim and abs(len(a_ck) - len(o_ck)) <= max(2, int(len(o_ck) * 0.25)):
                        best_sim = sim
                        best_asr_txt = a_txt

                rms, f0_hits, med_f0 = _measure_voice_in_wav(wf_16k, os_t, oe_t)
                # Nếu câu chỉ có 1-2 chữ, tự tin thấp và hoàn toàn im lặng (logo áo/chữ cảnh) -> lọc bỏ
                if len(o_ck) <= 2 and o_conf < 0.85 and rms < 0.015 and f0_hits < 2 and best_sim < 0.4:
                    screen_captions.append(o)
                    continue

                candidates.append(
                    {
                        "start": round(os_t, 3),
                        "end": round(oe_t, 3),
                        "text": o_txt,
                        "asr_text": best_asr_txt,
                        "ocr_text": o_txt,
                        "spk_id": spk_id,
                        "audit_status": "MATCHED" if best_sim >= 0.45 else "OCR_HARDSUB",
                        "pinyin_sim": round(best_sim, 3),
                        "f0_hint": med_f0,
                    }
                )
        finally:
            if wf_16k is not None:
                wf_16k.close()
    else:
        # FALLBACK CHO PHIM THÔ KHÔNG CÓ HARDSUB: đối chiếu ASR + cứu hộ OCR 8fps
        need_rescue_indices: list[int] = []
        for idx, a in enumerate(asr_segs):
            a_s, a_e = _tighten_words_span(a)
            a_txt = str(a["text"]).strip()
            a_ck = clean_cjk(a_txt)
            nearby = [o for o in ocr_segs if float(o["end"]) >= a_s - 1.3 and float(o["start"]) <= a_e + 1.3]
            best_ocr = None
            best_score = -1.0
            for o in nearby:
                sim = pinyin_similarity(a_txt, o["text"])
                overlap = max(0.0, min(a_e, float(o["end"])) - max(a_s, float(o["start"])))
                dist = max(0.0, max(a_s - float(o["end"]), float(o["start"]) - a_e))
                score = sim * 0.72 + (0.28 if overlap > 0 else max(0.0, 0.22 - dist * 0.15))
                if score > best_score:
                    best_score = score
                    best_ocr = o
            ocr_txt = best_ocr["text"] if best_ocr else ""
            ocr_ck = clean_cjk(ocr_txt)
            sim_best = pinyin_similarity(a_txt, ocr_txt) if best_ocr else 0.0
            matched = bool(best_ocr) and (sim_best >= 0.45 or (len(a_ck) <= 2 and len(ocr_ck) >= 1))
            if matched and best_ocr:
                used_ocr_ids.add(int(best_ocr["id"]))
                candidates.append(
                    {
                        "start": round(a_s, 3),
                        "end": round(a_e, 3),
                        "text": ocr_txt,
                        "asr_text": a_txt,
                        "ocr_text": ocr_txt,
                        "spk_id": a.get("spk_id", "SPK_UNK"),
                        "audit_status": "MATCHED",
                        "pinyin_sim": round(sim_best, 3),
                    }
                )
            else:
                candidates.append(
                    {
                        "start": round(a_s, 3),
                        "end": round(a_e, 3),
                        "text": ocr_txt if (len(ocr_ck) >= len(a_ck) and sim_best >= 0.35) else a_txt,
                        "asr_text": a_txt,
                        "ocr_text": ocr_txt,
                        "spk_id": a.get("spk_id", "SPK_UNK"),
                        "audit_status": "NEEDS_RESCUE",
                        "pinyin_sim": round(sim_best, 3),
                    }
                )
                need_rescue_indices.append(idx)

        if need_rescue_indices and cfr_mp4.exists():
            patch_onnx_threads(4)
            from rapidocr_onnxruntime import RapidOCR

            ocr_engine = RapidOCR(use_angle_cls=False, det_use_cuda=False, rec_use_cuda=False)
            for ri, ci in enumerate(need_rescue_indices, start=1):
                c = candidates[ci]
                r_txt = rescue_ocr_window(cfr_mp4, ocr_engine, c["start"], c["end"], fps=8.0)
                r_ck = clean_cjk(r_txt)
                r_sim = pinyin_similarity(c["asr_text"], r_txt) if r_ck else 0.0
                if r_ck and r_sim >= 0.45:
                    c["text"] = r_txt
                    c["ocr_text"] = r_txt
                    c["audit_status"] = "OCR_RESCUED"
                    c["pinyin_sim"] = round(r_sim, 3)
                    rescued_ocr += 1
                else:
                    c["text"] = c["asr_text"]
                    c["audit_status"] = "ASR_FALLBACK"
                    fallback_asr += 1

    candidates.sort(key=lambda c: c["start"])
    # Khóa chống chồng chéo timeline tuyệt đối (không bao giờ hiện 2 dòng sub đè nhau cùng lúc)
    for i in range(len(candidates) - 1):
        if candidates[i]["end"] > candidates[i + 1]["start"]:
            candidates[i]["end"] = round(max(candidates[i]["start"] + 0.15, candidates[i + 1]["start"] - 0.02), 3)

    for i, turn in enumerate(candidates, start=1):
        turn["id"] = i

    # Ghi SRT + ASS chuẩn kích cỡ & khung 1080x1920
    srt_text = "\n".join(
        f"{t['id']}\n{format_srt_ts(t['start'])} --> {format_srt_ts(t['end'])}\n{t['text']}\n"
        for t in candidates
    )
    ass_text = build_ass_text(candidates, width=1080, height=1920)
    out_srt.write_text(srt_text, encoding="utf-8")
    out_srt.with_suffix(".ass").write_text(ass_text, encoding="utf-8")
    if cfr_mp4.exists():
        cfr_mp4.with_suffix(".srt").write_text(srt_text, encoding="utf-8")
        cfr_mp4.with_suffix(".ass").write_text(ass_text, encoding="utf-8")

    elapsed = round(time.perf_counter() - t0, 2)
    matched_cnt = sum(1 for c in candidates if c["audit_status"] == "MATCHED")
    flags_report = {
        "elapsed_s": elapsed,
        "pypinyin_enabled": _HAS_PYPINYIN,
        "total_turns": len(candidates),
        "matched_count": matched_cnt,
        "ocr_rescued_count": rescued_ocr,
        "ocr_audio_confirmed_count": ocr_promoted,
        "asr_fallback_count": fallback_asr,
        "screen_captions_filtered": len(screen_captions),
        "flagged_samples": [c for c in candidates if c["audit_status"] != "MATCHED"][:25],
    }
    master_payload = {
        "elapsed_s": elapsed,
        "count": len(candidates),
        "turns": candidates,
        "screen_captions": screen_captions,
    }
    atomic_write_json(out_flags, flags_report)
    atomic_write_json(out_master, master_payload)

    emit_progress(
        100.0,
        f"Xong Bước 3: {len(candidates)} câu chuẩn (khớp {matched_cnt}, cứu OCR {rescued_ocr}, kéo thêm từ audio {ocr_promoted}, bù ASR {fallback_asr}, lọc {len(screen_captions)} chữ màn hình)",
    )
    print(f"[S3 DONE] {len(candidates)} câu chuẩn trong {elapsed}s -> {out_master.name}", flush=True)
    return master_payload


def main():
    parser = argparse.ArgumentParser(description="Bước 3: Đối chiếu chéo Audio ∩ OCR theo Pinyin & Cứu hộ câu cụt/sót.")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu")
    parser.add_argument("--force", action="store_true", help="Chạy lại từ đầu")
    args = parser.parse_args()
    run_step3(args.target_dir.resolve(), force=args.force)


if __name__ == "__main__":
    main()

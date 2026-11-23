"""Bước 4 (s4_separate): Tách giọng CÓ CHỌN LỌC chỉ tại các block có thoại + Cắt 2 mẫu Clone vàng (1 Nam + 1 Nữ, 6–10s).

Đột phá so với cách cũ:
  - KHÔNG tách mù toàn bộ 2.5 tiếng phim! Chỉ chạy BS-RoFormer / MelBand-RoFormer (fallback MDX-Net)
    trên các block đã xác nhận có thoại ở Bước 3 -> lưu cache từng block (`sep_blocks/`).
  - Ngoài block thoại: chép thẳng 100% `01_orig_48k_stereo.wav` nguyên bản (giữ nguyên tiếng reo hò,
    va chạm, không gian phim không bị méo kiểu MDX!).
  - Trong block thoại: đắp `instrumental` vừa tách với crossfade mượt 120ms ở 2 mép.
  - Đo F0 (cao độ giọng nam/nữ) + SNR từng câu, rồi tự chọn ghép 2–4 câu sạch nhất thành đúng 2 mẫu:
      * `04_ref_male.wav`   (6–10 giây giọng Nam chuẩn nhất)
      * `04_ref_female.wav` (6–10 giây giọng Nữ chuẩn nhất)
"""
from __future__ import annotations

import argparse
import shutil
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from _engine.core.checkpoint import ENGINE_DIR, atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, load_pipeline_config, release_vram

SR = 48000
CHANNELS = 2
CROSSFADE_S = 0.12


def _read_stereo_f32(wf: wave.Wave_read, start_f: int, count_f: int) -> np.ndarray:
    n_tot = wf.getnframes()
    i0 = max(0, min(n_tot, start_f))
    cnt = max(0, min(n_tot - i0, count_f))
    wf.setpos(i0)
    raw = wf.readframes(cnt)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n_ch = wf.getnchannels()
    if n_ch == 1:
        arr = np.repeat(arr[:, None], 2, axis=1)
    else:
        arr = arr.reshape(-1, n_ch)[:, :2]
    return arr


def _write_stereo_wav(path: Path, samples: np.ndarray, sr: int = SR):
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _write_mono_wav(path: Path, samples: np.ndarray, sr: int = SR):
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _group_dialogue_blocks(turns: list[dict[str, Any]], total_dur: float, pad: float = 0.35, gap_merge: float = 1.8) -> list[dict[str, Any]]:
    """Gom các câu thoại gần nhau thành block tách giọng để giảm overhead & crossfade mượt."""
    blocks: list[dict[str, Any]] = []
    for t in turns:
        s = max(0.0, float(t["start"]) - pad)
        e = min(total_dur, float(t["end"]) + pad)
        if blocks and s <= blocks[-1]["end"] + gap_merge:
            blocks[-1]["end"] = max(blocks[-1]["end"], e)
            blocks[-1]["turn_ids"].append(t["id"])
        else:
            blocks.append({"id": len(blocks) + 1, "start": s, "end": e, "turn_ids": [t["id"]]})
    return blocks


def _match_sep_outputs(output_files: list[str], search_dir: Path) -> tuple[Path, Path]:
    inst = voc = None
    for f in output_files:
        p = Path(f)
        if not p.is_absolute():
            p = search_dir / p.name
        low = p.name.lower()
        if "instrumental" in low or "no_vocals" in low or "other" in low:
            inst = p
        elif "vocal" in low:
            voc = p
    if inst is None or voc is None:
        raise RuntimeError(f"Không nhận diện được cặp instrumental/vocals từ: {output_files}")
    return inst, voc


def _measure_pitch_and_snr(voc_mono: np.ndarray, inst_mono: np.ndarray, sr: int = SR) -> dict[str, Any]:
    """Đo cao độ trung vị F0 (Hz) và tỷ lệ sạch giọng/nhạc nền (SNR dB)."""
    if len(voc_mono) < int(0.1 * sr):
        return {"f0_hz": None, "rms": 0.0, "snr_db": -99.0}
    v_rms = float(np.sqrt(np.mean(voc_mono ** 2)) + 1e-8)
    i_rms = float(np.sqrt(np.mean(inst_mono ** 2)) + 1e-8)
    snr_db = round(float(20.0 * np.log10(v_rms / i_rms)), 2)

    win = int(0.045 * sr)
    hop = int(0.02 * sr)
    min_lag, max_lag = int(sr / 360), int(sr / 75)
    f0s = []
    for k in range(0, len(voc_mono) - win, hop):
        seg = voc_mono[k : k + win]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.025:
            continue
        seg = seg - np.mean(seg)
        ac = np.correlate(seg, seg, mode="full")[win - 1 :]
        if ac[0] <= 0:
            continue
        pk = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
        if ac[pk] > 0.42 * ac[0]:
            f0s.append(sr / pk)
    med_f0 = round(float(np.median(f0s)), 1) if f0s else None
    return {"f0_hz": med_f0, "rms": round(v_rms, 4), "snr_db": snr_db}


def _build_golden_clone_ref(
    selected_clips: list[np.ndarray],
    out_wav: Path,
    min_sec: float = 5.0,
    max_sec: float = 10.0,
    pause_sec: float = 0.2,
) -> float:
    """Ghép các câu sạch nhất thành clip mẫu clone chuẩn 6–10 giây."""
    if not selected_clips:
        return 0.0
    pause = np.zeros(int(pause_sec * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    acc_len = 0
    max_samples = int(max_sec * SR)
    for clip in selected_clips:
        if acc_len + len(clip) > max_samples and acc_len >= int(min_sec * SR):
            break
        if parts:
            parts.append(pause)
            acc_len += len(pause)
        take = clip[: max(0, max_samples - acc_len)]
        if len(take) > 0:
            parts.append(take)
            acc_len += len(take)
        if acc_len >= int(8.0 * SR):
            break
    if not parts:
        return 0.0
    merged = np.concatenate(parts)
    pk = float(np.max(np.abs(merged))) + 1e-8
    if pk > 0.92:
        merged = merged * (0.92 / pk)
    _write_mono_wav(out_wav, merged, sr=SR)
    return round(len(merged) / float(SR), 2)


def run_step4(target_dir: Path, force: bool = False) -> dict[str, Any]:
    clean_bed_wav = target_dir / "04_clean_bed.wav"
    meta_json = target_dir / "04_f0_speakers.json"
    ref_male_wav = target_dir / "04_ref_male.wav"
    ref_female_wav = target_dir / "04_ref_female.wav"
    sep_dir = target_dir / "sep_blocks"

    orig_wav = target_dir / "01_orig_48k_stereo.wav"
    master_json = target_dir / "03_aligned_master.json"
    if not orig_wav.exists() or not master_json.exists():
        raise FileNotFoundError("Thiếu 01_orig_48k_stereo.wav hoặc 03_aligned_master.json cho Bước 4.")

    if not force and clean_bed_wav.exists() and meta_json.exists() and clean_bed_wav.stat().st_size > 100_000:
        existing = read_json_safe(meta_json)
        if existing:
            emit_progress(100.0, "Tái sử dụng checkpoint Bước 4 (clean_bed + 2 mẫu Clone vàng)")
            return existing

    t0 = time.perf_counter()
    sep_dir.mkdir(parents=True, exist_ok=True)
    turns = (read_json_safe(master_json) or {}).get("turns", [])

    with wave.open(str(orig_wav), "rb") as wf_orig:
        n_total = wf_orig.getnframes()
        sr = wf_orig.getframerate()
        total_dur = n_total / float(sr)

    blocks = _group_dialogue_blocks(turns, total_dur)
    dialogue_sec = sum(b["end"] - b["start"] for b in blocks)
    kept_orig_sec = max(0.0, total_dur - dialogue_sec)
    print(f"[S4] Gom {len(turns)} câu -> {len(blocks)} block thoại ({dialogue_sec:.1f}s tách, giữ nguyên 100% gốc {kept_orig_sec:.1f}s).", flush=True)

    # Nạp audio-separator (ưu tiên BS-RoFormer, fallback UVR-MDX-NET-Inst_HQ_3)
    cfg = load_pipeline_config()
    model_primary = cfg.get("models", {}).get("separator_model", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    model_fallback = cfg.get("models", {}).get("separator_fallback", "UVR-MDX-NET-Inst_HQ_3.onnx")
    model_dir = ENGINE_DIR / "models" / "separator"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra xem có block nào chưa tách xong trong cache không
    pending_blocks = [
        b for b in blocks if not (sep_dir / f"blk_{b['id']:04d}_inst.wav").exists() or not (sep_dir / f"blk_{b['id']:04d}_voc.wav").exists() or force
    ]

    model_used = model_primary
    if pending_blocks:
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

        mdx_dir = ENGINE_DIR / "models" / "mdx"
        if not use_cuda and (mdx_dir / model_fallback).exists():
            model_dir = mdx_dir
            model_primary = model_fallback
            model_used = model_fallback

        emit_progress(5.0, f"Đang nạp model tách giọng ({model_primary})...")
        from audio_separator.separator import Separator

        tmp_sep = sep_dir / "_tmp"
        tmp_sep.mkdir(parents=True, exist_ok=True)
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir=str(tmp_sep),
            output_format="WAV",
            sample_rate=sr,
            use_autocast=use_cuda,
        )
        try:
            separator.load_model(model_filename=model_primary)
        except Exception as exc:
            print(f"[S4 WARN] Không tải được {model_primary} ({exc}), chuyển sang {model_fallback}.", flush=True)
            model_used = model_fallback
            separator.load_model(model_filename=model_fallback)

        with wave.open(str(orig_wav), "rb") as wf_orig:
            for idx, b in enumerate(pending_blocks, start=1):
                pct = 8.0 + (idx / len(pending_blocks)) * 65.0
                emit_progress(pct, f"Tách giọng block {idx}/{len(pending_blocks)} [{b['start']:.1f}s–{b['end']:.1f}s]...")
                i0 = int(round(b["start"] * sr))
                i1 = int(round(b["end"] * sr))
                clip = _read_stereo_f32(wf_orig, i0, i1 - i0)
                in_wav = tmp_sep / f"in_{b['id']:04d}.wav"
                _write_stereo_wav(in_wav, clip, sr=sr)

                outs = separator.separate(str(in_wav))
                inst_src, voc_src = _match_sep_outputs(outs, tmp_sep)
                shutil.move(str(inst_src), str(sep_dir / f"blk_{b['id']:04d}_inst.wav"))
                shutil.move(str(voc_src), str(sep_dir / f"blk_{b['id']:04d}_voc.wav"))
                in_wav.unlink(missing_ok=True)
                if idx % 20 == 0:
                    release_vram()

        shutil.rmtree(tmp_sep, ignore_errors=True)
        del separator
        release_vram()

    # Pha 2: Ghép 04_clean_bed.wav (100% gốc ngoài block thoại + instrumental crossfade 120ms trong block thoại)
    emit_progress(76.0, "Đang ghép 04_clean_bed.wav (giữ 100% SFX/reo hò gốc ngoài thoại)...")
    xfade = int(CROSSFADE_S * sr)
    blk_by_turn: dict[int, dict[str, Any]] = {}
    for b in blocks:
        for tid in b["turn_ids"]:
            blk_by_turn[tid] = b

    with wave.open(str(orig_wav), "rb") as wf_orig, wave.open(str(clean_bed_wav), "wb") as wf_out:
        wf_out.setnchannels(2)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sr)

        cursor = 0
        for b in blocks:
            b0 = int(round(b["start"] * sr))
            b1 = min(n_total, int(round(b["end"] * sr)))
            # Ghi nguyên bản 100% đoạn không có thoại trước block này (stream theo khúc 30s để nhẹ RAM)
            while cursor < b0:
                step = min(b0 - cursor, 30 * sr)
                raw = _read_stereo_f32(wf_orig, cursor, step)
                wf_out.writeframes(np.clip(raw * 32767.0, -32768, 32767).astype(np.int16).tobytes())
                cursor += step

            # Ghi block thoại với crossfade 120ms từ gốc -> instrumental -> gốc
            orig_blk = _read_stereo_f32(wf_orig, b0, b1 - b0)
            with wave.open(str(sep_dir / f"blk_{b['id']:04d}_inst.wav"), "rb") as wfi:
                inst_blk = _read_stereo_f32(wfi, 0, len(orig_blk))
            if len(inst_blk) < len(orig_blk):
                pad_arr = np.zeros((len(orig_blk) - len(inst_blk), 2), dtype=np.float32)
                inst_blk = np.vstack([inst_blk, pad_arr])
            else:
                inst_blk = inst_blk[: len(orig_blk)]

            xf = min(xfade, len(orig_blk) // 3)
            mixed_blk = inst_blk.copy()
            if xf > 0:
                ramp_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
                ramp_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)[:, None]
                mixed_blk[:xf] = orig_blk[:xf] * (1.0 - ramp_in) + inst_blk[:xf] * ramp_in
                mixed_blk[-xf:] = inst_blk[-xf:] * ramp_out + orig_blk[-xf:] * (1.0 - ramp_out)

            wf_out.writeframes(np.clip(mixed_blk * 32767.0, -32768, 32767).astype(np.int16).tobytes())
            cursor = b1

        while cursor < n_total:
            step = min(n_total - cursor, 30 * sr)
            raw = _read_stereo_f32(wf_orig, cursor, step)
            wf_out.writeframes(np.clip(raw * 32767.0, -32768, 32767).astype(np.int16).tobytes())
            cursor += step

    # Pha 3: Đo F0 + SNR từng câu & cắt ghép 2 mẫu Clone vàng (1 Nam + 1 Nữ, 6–10s)
    emit_progress(88.0, "Đang đo F0 nam/nữ & lọc ghép 2 mẫu Clone vàng (6–10s)...")
    turn_acoustics: list[dict[str, Any]] = []
    male_cands: list[tuple[float, np.ndarray]] = []
    female_cands: list[tuple[float, np.ndarray]] = []

    for t in turns:
        b = blk_by_turn[t["id"]]
        rel_s = max(0.0, float(t["start"]) - b["start"])
        rel_e = max(rel_s + 0.1, float(t["end"]) - b["start"])
        s0, cnt = int(round(rel_s * sr)), int(round((rel_e - rel_s) * sr))

        with wave.open(str(sep_dir / f"blk_{b['id']:04d}_voc.wav"), "rb") as wfv:
            v_arr = _read_stereo_f32(wfv, s0, cnt).mean(axis=1)
        with wave.open(str(sep_dir / f"blk_{b['id']:04d}_inst.wav"), "rb") as wfi:
            i_arr = _read_stereo_f32(wfi, s0, cnt).mean(axis=1)

        m = _measure_pitch_and_snr(v_arr, i_arr, sr=sr)
        f0 = m["f0_hz"]
        gender_hint = "FEMALE" if (f0 is not None and f0 >= 175.0) else ("MALE" if f0 is not None else "UNKNOWN")
        dur_t = rel_e - rel_s
        turn_acoustics.append(
            {
                "id": t["id"],
                "spk_id": t.get("spk_id", "SPK_UNK"),
                "f0_hz": f0,
                "rms": m["rms"],
                "snr_db": m["snr_db"],
                "gender_hint": gender_hint,
            }
        )
        # Tuyển câu sạch cho mẫu Clone: dài >=0.75s, không rè/clipping, SNR tốt (ưu tiên câu dài & SNR cao)
        peak = float(np.max(np.abs(v_arr))) if len(v_arr) else 0.0
        if 0.75 <= dur_t <= 5.5 and 0.03 <= m["rms"] <= 0.45 and peak < 0.98 and m["snr_db"] >= 2.0:
            score = m["snr_db"] + min(4.0, dur_t * 1.5)
            if gender_hint == "MALE":
                male_cands.append((score, v_arr))
            elif gender_hint == "FEMALE":
                female_cands.append((score, v_arr))

    ref_cfg = cfg.get("clone_ref", {})
    male_cands.sort(key=lambda x: x[0], reverse=True)
    female_cands.sort(key=lambda x: x[0], reverse=True)
    male_dur = _build_golden_clone_ref(
        [c for _, c in male_cands],
        ref_male_wav,
        min_sec=float(ref_cfg.get("min_sec", 5.0)),
        max_sec=float(ref_cfg.get("max_sec", 10.0)),
        pause_sec=float(ref_cfg.get("pause_sec", 0.2)),
    )
    female_dur = _build_golden_clone_ref(
        [c for _, c in female_cands],
        ref_female_wav,
        min_sec=float(ref_cfg.get("min_sec", 5.0)),
        max_sec=float(ref_cfg.get("max_sec", 10.0)),
        pause_sec=float(ref_cfg.get("pause_sec", 0.2)),
    )

    elapsed = round(time.perf_counter() - t0, 2)
    payload = {
        "model_used": model_used,
        "elapsed_s": elapsed,
        "blocks_count": len(blocks),
        "dialogue_sec": round(dialogue_sec, 1),
        "kept_orig_sec": round(kept_orig_sec, 1),
        "ref_male": {"path": str(ref_male_wav), "duration_s": male_dur, "valid": male_dur >= 5.0},
        "ref_female": {"path": str(ref_female_wav), "duration_s": female_dur, "valid": female_dur >= 5.0},
        "acoustics": turn_acoustics,
    }
    atomic_write_json(meta_json, payload)
    emit_progress(
        100.0,
        f"Xong Bước 4 ({elapsed}s): giữ nguyên {kept_orig_sec:.0f}s gốc | Mẫu Clone Nam {male_dur}s, Nữ {female_dur}s",
    )
    print(f"[S4 DONE] {elapsed}s | Ref Nam={male_dur}s, Ref Nữ={female_dur}s -> {clean_bed_wav.name}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Bước 4: Tách giọng chọn lọc vùng thoại + Cắt 2 mẫu Clone vàng (6–10s).")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu")
    parser.add_argument("--force", action="store_true", help="Chạy lại toàn bộ các block")
    args = parser.parse_args()
    run_step4(args.target_dir.resolve(), force=args.force)


if __name__ == "__main__":
    main()

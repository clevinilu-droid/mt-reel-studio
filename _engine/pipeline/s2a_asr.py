"""Bước 2A (s2a_asr): Nghe thoại & Gom người nói bằng FunASR (ưu tiên) hoặc Faster-Whisper Batched.

Đầu vào:
  - `<target_dir>/01_orig_16k_ai.wav`
Đầu ra:
  - `<target_dir>/02a_chunks.jsonl` (micro-checkpoint từng khúc 5 phút, resume-safe 100%)
  - `<target_dir>/02a_audio_asr.json` (danh sách câu thoại chuẩn có mốc giờ + spk_id + words)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from _engine.core.checkpoint import atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, load_pipeline_config, release_vram

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def read_wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def read_wav_chunk_f32(wav_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_total = wf.getnframes()
        i0 = max(0, min(n_total, int(round(start_s * sr))))
        cnt = max(0, min(n_total - i0, int(round(dur_s * sr))))
        wf.setpos(i0)
        raw = wf.readframes(cnt)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_chunk_checkpoints(jsonl_path: Path) -> dict[int, list[dict[str, Any]]]:
    done: dict[int, list[dict[str, Any]]] = {}
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done[int(rec["chunk_idx"])] = rec["segments"]
        except Exception:
            continue
    return done


def append_chunk_checkpoint(jsonl_path: Path, chunk_idx: int, segments: list[dict[str, Any]]):
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"chunk_idx": chunk_idx, "segments": segments}, ensure_ascii=False) + "\n")


def _try_run_funasr(wav_16k: Path, jsonl_path: Path, total_dur: float, chunk_sec: float, cfg: dict) -> list[dict[str, Any]] | None:
    """Chạy FunASR (Paraformer-zh + FSMN-VAD + CAM++ tách người nói) theo chunk 5 phút; tự fallback sang Faster-Whisper nếu lỗi."""
    try:
        from funasr import AutoModel

        models_cfg = cfg.get("models", {})
        emit_progress(5.0, "Đang nạp FunASR (Paraformer-zh + VAD + CAM++ tách người nói)...")
        print("[S2A] Khởi tạo FunASR AutoModel (Paraformer-zh + CAM++)...", flush=True)
        model = None
        for punc_cand in (models_cfg.get("funasr_punc", "ct-punc"), "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", None):
            try:
                kw: dict[str, Any] = {
                    "model": models_cfg.get("funasr_model", "paraformer-zh"),
                    "vad_model": models_cfg.get("funasr_vad", "fsmn-vad"),
                    "spk_model": models_cfg.get("funasr_spk", "cam++"),
                    "disable_update": True,
                }
                if punc_cand:
                    kw["punc_model"] = punc_cand
                model = AutoModel(**kw)
                break
            except Exception as exc:
                print(f"[S2A WARN] FunASR punc={punc_cand} lỗi ({exc}), thử cấu hình dự phòng...", flush=True)
        if model is None:
            return None

        done_chunks = load_chunk_checkpoints(jsonl_path)
        n_chunks = max(1, int(np.ceil(total_dur / chunk_sec)))
        all_segs: list[dict[str, Any]] = []

        for ci in range(n_chunks):
            c_start = ci * chunk_sec
            c_dur = min(chunk_sec, total_dur - c_start)
            if ci in done_chunks:
                all_segs.extend(done_chunks[ci])
                emit_progress(10.0 + ((ci + 1) / n_chunks) * 85.0, f"Bỏ qua chunk {ci+1}/{n_chunks} (đã lưu checkpoint)")
                continue

            emit_progress(10.0 + (ci / n_chunks) * 85.0, f"FunASR nghe & tách người nói chunk {ci+1}/{n_chunks} ({c_start:.0f}s)...")
            audio = read_wav_chunk_f32(wav_16k, c_start, c_dur)
            res = model.generate(input=audio, batch_size_s=180, sentence_timestamp=True)
            chunk_segs: list[dict[str, Any]] = []
            if res and isinstance(res, list):
                for item in res:
                    sent_info = item.get("sentence_info") or []
                    for s in sent_info:
                        txt = str(s.get("text", "")).strip()
                        if not has_cjk(txt):
                            continue
                        st = round(c_start + float(s.get("start", 0)) / 1000.0, 3)
                        en = round(c_start + float(s.get("end", 0)) / 1000.0, 3)
                        spk_raw = s.get("spk", 0)
                        chunk_segs.append(
                            {
                                "start": st,
                                "end": max(st + 0.15, en),
                                "text": txt,
                                "spk_id": f"SPK_{int(spk_raw):02d}" if str(spk_raw).isdigit() else str(spk_raw),
                                "no_speech_prob": 0.05,
                            }
                        )
            append_chunk_checkpoint(jsonl_path, ci, chunk_segs)
            all_segs.extend(chunk_segs)
            print(f"[S2A] FunASR chunk {ci+1}/{n_chunks}: {len(chunk_segs)} câu thoại.", flush=True)

        del model
        release_vram()
        return all_segs
    except Exception as exc:
        print(f"[S2A WARN] FunASR không khả dụng ({exc}) -> tự chuyển sang Faster-Whisper Batched GPU!", flush=True)
        return None


def _run_faster_whisper(wav_16k: Path, jsonl_path: Path, total_dur: float, chunk_sec: float, cfg: dict) -> list[dict[str, Any]]:
    """Chạy Faster-Whisper (BatchedInferencePipeline trên CUDA, int8 trên CPU) có lọc Silero VAD."""
    from faster_whisper import WhisperModel

    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    models_cfg = cfg.get("models", {})
    model_name = models_cfg.get("whisper_cuda_model", "large-v3-turbo") if use_cuda else models_cfg.get("whisper_cpu_model", "small")
    emit_progress(5.0, f"Đang nạp Faster-Whisper ({model_name} | {'CUDA' if use_cuda else 'CPU'})...")
    print(f"[S2A] Khởi tạo Faster-Whisper model={model_name}, cuda={use_cuda}", flush=True)

    base_model = WhisperModel(
        model_name,
        device="cuda" if use_cuda else "cpu",
        compute_type="float16" if use_cuda else "int8",
    )

    batched = None
    if use_cuda:
        try:
            from faster_whisper import BatchedInferencePipeline

            batched = BatchedInferencePipeline(model=base_model)
            print("[S2A] Đã bật BatchedInferencePipeline (batch_size=16) trên CUDA.", flush=True)
        except Exception:
            batched = None

    done_chunks = load_chunk_checkpoints(jsonl_path)
    n_chunks = max(1, int(np.ceil(total_dur / chunk_sec)))
    all_segs: list[dict[str, Any]] = []

    for ci in range(n_chunks):
        c_start = ci * chunk_sec
        c_dur = min(chunk_sec, total_dur - c_start)
        if ci in done_chunks:
            all_segs.extend(done_chunks[ci])
            emit_progress(10.0 + ((ci + 1) / n_chunks) * 85.0, f"Bỏ qua chunk {ci+1}/{n_chunks} (đã lưu checkpoint)")
            continue

        emit_progress(10.0 + (ci / n_chunks) * 85.0, f"Whisper VAD+ASR chunk {ci+1}/{n_chunks} ({c_start:.0f}s)...")
        audio = read_wav_chunk_f32(wav_16k, c_start, c_dur)
        if batched is not None:
            segs_iter, _ = batched.transcribe(audio, language="zh", batch_size=16, vad_filter=True, word_timestamps=True)
        else:
            segs_iter, _ = base_model.transcribe(audio, language="zh", beam_size=2, vad_filter=True, word_timestamps=True)

        chunk_segs: list[dict[str, Any]] = []
        for s in segs_iter:
            txt = (s.text or "").strip()
            nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
            if not has_cjk(txt) or nsp >= 0.78:
                continue
            words = []
            if getattr(s, "words", None):
                for w in s.words:
                    words.append({"s": round(c_start + w.start, 3), "e": round(c_start + w.end, 3), "w": w.word})

            seg_start = round(c_start + s.start, 3)
            seg_end = round(c_start + s.end, 3)
            # Cắt bỏ từ đơn bị Whisper kéo dãn vượt qua khoảng lặng >1.2s ở đầu/cuối câu
            if len(words) >= 2:
                best_cluster = [words[0]]
                clusters: list[list[dict[str, Any]]] = [best_cluster]
                for w in words[1:]:
                    if w["s"] - clusters[-1][-1]["e"] >= 1.2:
                        clusters.append([w])
                    else:
                        clusters[-1].append(w)
                # Chọn cụm có nhiều chữ nhất (ưu tiên cụm chính thực sự nói)
                main_cl = max(clusters, key=lambda cl: (len("".join(x["w"] for x in cl)), cl[-1]["e"] - cl[0]["s"]))
                seg_start = main_cl[0]["s"]
                seg_end = max(seg_start + 0.15, main_cl[-1]["e"])
                words = main_cl

            chunk_segs.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": txt,
                    "spk_id": "SPK_UNK",
                    "no_speech_prob": round(nsp, 3),
                    "words": words,
                }
            )
        append_chunk_checkpoint(jsonl_path, ci, chunk_segs)
        all_segs.extend(chunk_segs)
        print(f"[S2A] Whisper chunk {ci+1}/{n_chunks}: {len(chunk_segs)} câu thoại.", flush=True)

    del batched
    del base_model
    release_vram()
    return all_segs


def run_step2a(target_dir: Path, force: bool = False, chunk_sec: float = 300.0) -> dict[str, Any]:
    out_json = target_dir / "02a_audio_asr.json"
    jsonl_path = target_dir / "02a_chunks.jsonl"
    wav_16k = target_dir / "01_orig_16k_ai.wav"
    if not wav_16k.exists():
        raise FileNotFoundError(f"Thiếu file audio Bước 1: {wav_16k}")

    if not force:
        existing = read_json_safe(out_json)
        if existing and existing.get("segments"):
            emit_progress(100.0, f"Tái sử dụng checkpoint 2A ({len(existing['segments'])} câu)")
            print(f"[S2A] Tái sử dụng {out_json.name} ({len(existing['segments'])} câu).", flush=True)
            return existing
    else:
        jsonl_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    cfg = load_pipeline_config()
    total_dur = read_wav_duration(wav_16k)
    engine_pref = cfg.get("models", {}).get("asr_engine", "auto")

    segs = None
    engine_used = "faster-whisper"
    if engine_pref in ("auto", "funasr"):
        segs = _try_run_funasr(wav_16k, jsonl_path, total_dur, chunk_sec, cfg)
        if segs is not None:
            engine_used = "funasr"
    if segs is None:
        segs = _run_faster_whisper(wav_16k, jsonl_path, total_dur, chunk_sec, cfg)

    segs.sort(key=lambda x: x["start"])
    for idx, s in enumerate(segs, start=1):
        s["id"] = idx

    elapsed = round(time.perf_counter() - t0, 2)
    payload = {
        "engine": engine_used,
        "duration_s": round(total_dur, 3),
        "elapsed_s": elapsed,
        "count": len(segs),
        "segments": segs,
    }
    atomic_write_json(out_json, payload)
    emit_progress(100.0, f"Xong 2A ({engine_used}): {len(segs)} câu thoại ({elapsed}s)")
    print(f"[S2A DONE] {engine_used}: {len(segs)} câu thoại trong {elapsed}s -> {out_json.name}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Bước 2A: Nghe thoại & Gom người nói (FunASR / Faster-Whisper).")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu (work_dir hoặc work_dir/_test_5min)")
    parser.add_argument("--force", action="store_true", help="Chạy lại từ đầu bỏ qua checkpoint")
    parser.add_argument("--chunk-sec", type=float, default=300.0, help="Độ dài mỗi chunk checkpoint (giây)")
    args = parser.parse_args()
    run_step2a(args.target_dir.resolve(), force=args.force, chunk_sec=args.chunk_sec)


if __name__ == "__main__":
    main()

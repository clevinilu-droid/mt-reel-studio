"""Worker cô lập chạy trong venv `/opt/venvs/voxcpm` để tổng hợp giọng đa ngôn ngữ bằng OpenBMB/VoxCPM2.

Tối ưu cho chế độ `single_female` (1 giọng nữ đồng nhất giàu cảm xúc, KHÔNG clone giọng gốc):
1. Nạp model `openbmb/VoxCPM2` ĐÚNG 1 LẦN duy nhất cho cả loạt ngôn ngữ (EN/JA/KO/ES).
2. Với mỗi ngôn ngữ, dùng VoxCPM2 Voice Design tạo 1 mẫu giọng nữ chuẩn nhân tạo (~5s, 48kHz).
3. Dùng chính mẫu giọng nhân tạo đó làm `reference_wav_path` (Controllable Voice Cloning) để khóa
   cùng một chất giọng đồng nhất cho toàn bộ các câu thoại trong ngôn ngữ đó, kèm chỉ dẫn cảm xúc.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import re
import sys
from pathlib import Path

ANCHOR_TEXTS = {
    "en": "Good evening everyone, today we are continuing the story right where we left off, smoothly and clearly.",
    "ja": "みなさんこんばんは、今夜も物語の続きをわかりやすく、落ち着いた声でお届けします。",
    "ko": "안녕하세요 여러분, 오늘도 이야기의 다음 장면을 자연스럽고 또렷한 목소리로 들려드리겠습니다.",
    "es": "Buenas noches a todos, hoy continuamos con nuestra historia de una manera clara, cálida y muy natural.",
}

VOICE_DESIGN_PERSONAS = {
    "FEMALE": "(A young adult woman in her mid-20s with an expressive, warm, clear, natural storytelling voice, lively conversational cadence, crisp native diction, studio recording)",
}

EMOTION_CTRL = {
    "NEUTRAL": "(natural expressive conversational tone)",
    "ANGRY": "(sharp, firm, angry dramatic tone)",
    "HAPPY": "(warm, bright, cheerful smiling tone)",
    "SAD": "(soft, emotional, slightly sorrowful tone)",
    "WHISPER": "(low, hushed, urgent whisper tone)",
    "URGENT": "(fast, tense, urgent dramatic tone)",
}


def _clean_dialogue_text(text: str) -> str:
    """Làm sạch ngoặc đơn trong lời thoại để VoxCPM2 không nhầm với thẻ mô tả giọng."""
    cleaned = re.sub(r"\(([^)]*)\)", r"\1", text or "")
    cleaned = re.sub(r"[()\[\]]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM2 Worker (Voice Design / Controllable Non-Clone).")
    parser.add_argument("--manifest", type=Path, required=True, help="File JSON cấu hình lượt thoại cần tổng hợp.")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_id = payload.get("model_id", "openbmb/VoxCPM2")
    use_clone = bool(payload.get("use_clone", False))

    lang_jobs = payload.get("languages")
    if not isinstance(lang_jobs, list):
        lang_jobs = [
            {
                "lang": str(payload.get("lang", "en")).lower(),
                "anchor_paths": dict(payload.get("anchor_paths", {})),
                "ref_paths": dict(payload.get("ref_paths", {})),
                "tasks": list(payload.get("tasks", [])),
            }
        ]

    total_tasks = sum(len(j.get("tasks", [])) for j in lang_jobs)
    if total_tasks == 0:
        print("[VOXCPM] Không có câu thoại mới cần tổng hợp.", flush=True)
        return 0

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lang_list_str = ",".join(str(j.get("lang", "")).upper() for j in lang_jobs)
    print(f"[VOXCPM] Đang nạp 1 lần {model_id} trên {device} ({lang_list_str}, tổng {total_tasks} câu)...", flush=True)
    model = VoxCPM.from_pretrained(
        hf_model_id=model_id,
        load_denoiser=False,
        optimize=False,
        device=device,
    )
    sr = int(getattr(model.tts_model, "sample_rate", 48000))
    gen_params = set(inspect.signature(model._generate).parameters.keys())

    def _call_generate(**kwargs):
        seed_val = kwargs.pop("seed", None)
        if seed_val is not None:
            if "seed" in gen_params:
                kwargs["seed"] = seed_val
            else:
                torch.manual_seed(int(seed_val))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed_val))
        filtered = {k: v for k, v in kwargs.items() if k in gen_params}
        return model.generate(**filtered)

    done_global = 0
    for job in lang_jobs:
        lang = str(job.get("lang", "en")).lower()
        tasks = list(job.get("tasks", []))
        if not tasks:
            continue
        anchor_paths = dict(job.get("anchor_paths", {}))
        active_refs: dict[str, str] = {}

        # Chỉ dùng 1 giọng Nữ không clone: tạo 1 anchor nhân tạo bằng Voice Design rồi khóa timbre qua reference_wav_path
        anchor_text = ANCHOR_TEXTS.get(lang, ANCHOR_TEXTS["en"])
        for g in ("FEMALE",):
            ap_str = anchor_paths.get(g)
            if not ap_str:
                continue
            ap = Path(ap_str)
            if not ap.exists() or ap.stat().st_size <= 4000:
                ap.parent.mkdir(parents=True, exist_ok=True)
                persona = VOICE_DESIGN_PERSONAS["FEMALE"]
                print(f"[VOXCPM] Tạo giọng Nữ mẫu nhân tạo {lang.upper()} bằng Voice Design...", flush=True)
                wav_anchor = _call_generate(
                    text=f"{persona}{anchor_text}",
                    cfg_value=2.0,
                    inference_timesteps=12,
                    min_len=8,
                    max_len=1024,
                    normalize=True,
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    seed=2026,
                )
                wav_anchor = np.asarray(wav_anchor, dtype=np.float32)
                sf.write(str(ap), wav_anchor, sr)
            active_refs[g] = str(ap)

        for t in tasks:
            done_global += 1
            out_wav = Path(t["out_wav"])
            if out_wav.exists() and out_wav.stat().st_size > 2000:
                print(f"[VOXCPM_PROGRESS] {done_global}/{total_tasks} {lang.upper()}", flush=True)
                continue

            out_wav.parent.mkdir(parents=True, exist_ok=True)
            g = "FEMALE"
            emo = str(t.get("emotion", "NEUTRAL")).upper()
            ctrl = EMOTION_CTRL.get(emo, EMOTION_CTRL["NEUTRAL"])
            clean_txt = _clean_dialogue_text(str(t.get("text", "")))
            if not clean_txt:
                print(f"[VOXCPM_PROGRESS] {done_global}/{total_tasks} {lang.upper()}", flush=True)
                continue

            ref_wav = active_refs.get(g)
            if ref_wav:
                wav = _call_generate(
                    text=f"{ctrl}{clean_txt}",
                    reference_wav_path=ref_wav,
                    cfg_value=2.0,
                    inference_timesteps=10,
                    min_len=2,
                    max_len=2048,
                    normalize=True,
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=8.0,
                )
            else:
                persona = VOICE_DESIGN_PERSONAS.get(g, VOICE_DESIGN_PERSONAS["FEMALE"])
                wav = _call_generate(
                    text=f"{persona}{clean_txt}",
                    cfg_value=2.0,
                    inference_timesteps=10,
                    min_len=2,
                    max_len=2048,
                    normalize=True,
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=8.0,
                    seed=2026,
                )

            wav = np.asarray(wav, dtype=np.float32)
            sf.write(str(out_wav), wav, sr)
            print(f"[VOXCPM_PROGRESS] {done_global}/{total_tasks} {lang.upper()}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[VOXCPM DONE] {lang_list_str} ({total_tasks} câu)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

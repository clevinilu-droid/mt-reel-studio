"""
Worker chạy bằng /opt/pyvideotrans/.venv/bin/python (pyVideoTrans v4.19 · qwen-tts-pvt):
Clone trực tiếp giọng diễn viên gốc từ vocals_48k_mono.flac sang đa ngôn ngữ EN / JA / KO
bằng Qwen/Qwen3-TTS-12Hz-0.6B-Base trên GPU CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


LANG_MAP = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}


def parse_ts(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def build_ref_clips(lines: list[dict], vocals: np.ndarray, sr: int, tmp_dir: Path) -> tuple[dict[str, tuple[Path, str]], dict[int, tuple[Path, str]]]:
    """Extract clean reference clips from vocals_48k_mono.flac: both per-role best clip and per-line (>=1.8s) clip."""
    role_best: dict[str, tuple[float, dict]] = {}
    for x in lines:
        dur = parse_ts(x["end"]) - parse_ts(x["start"])
        role = x["voice_role"]
        # Prefer 2.0s - 4.5s clean segments
        score = dur if 1.8 <= dur <= 4.5 else (4.5 - abs(dur - 3.0) * 0.5)
        if role not in role_best or score > role_best[role][0]:
            role_best[role] = (score, x)

    role_refs: dict[str, tuple[Path, str]] = {}
    for role, (_, x) in role_best.items():
        s0 = max(0, int(parse_ts(x["start"]) * sr))
        s1 = min(len(vocals), int(parse_ts(x["end"]) * sr))
        p = tmp_dir / f"ref_role_{role}.wav"
        sf.write(str(p), vocals[s0:s1], sr, subtype="PCM_16")
        role_refs[role] = (p, x["zh"])
        print(f"[Ref] Role {role} <- line #{x['id']} ({(s1-s0)/sr:.2f}s): {x['zh']}", flush=True)

    line_refs: dict[int, tuple[Path, str]] = {}
    for x in lines:
        s0 = max(0, int(parse_ts(x["start"]) * sr))
        s1 = min(len(vocals), int(parse_ts(x["end"]) * sr))
        dur = (s1 - s0) / sr
        if dur >= 1.8:
            p = tmp_dir / f"ref_line_{x['id']:04d}.wav"
            sf.write(str(p), vocals[s0:s1], sr, subtype="PCM_16")
            line_refs[x["id"]] = (p, x["zh"])
        else:
            line_refs[x["id"]] = role_refs[x["voice_role"]]
    return role_refs, line_refs


def main():
    ap = argparse.ArgumentParser(description="Clone voices from vocals_48k_mono.flac into EN/JA/KO using pyVideoTrans Qwen3-TTS.")
    ap.add_argument("vocals", type=Path)
    ap.add_argument("role_lines", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--langs", default="en,ja,ko")
    ap.add_argument("--model-dir", default="/opt/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref_dir = args.out_dir / "_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    lines = json.loads(args.role_lines.read_text(encoding="utf-8"))
    vocals, sr = sf.read(str(args.vocals), dtype="float32")
    if vocals.ndim > 1:
        vocals = np.mean(vocals, axis=1)

    role_refs, line_refs = build_ref_clips(lines, vocals, sr, ref_dir)

    t0 = time.time()
    local_dir = Path(args.model_dir)
    if not (local_dir / "config.json").exists():
        from huggingface_hub import snapshot_download
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pyVideoTrans Qwen3-TTS] Downloading Qwen/Qwen3-TTS-12Hz-0.6B-Base -> {local_dir}...", flush=True)
        snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", local_dir=str(local_dir))
    model_src = str(local_dir)
    print(f"[pyVideoTrans Qwen3-TTS] Loading {model_src} on CUDA...", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained(model_src, device_map="auto", dtype=dtype)
    print(f"[pyVideoTrans Qwen3-TTS] Loaded in {time.time() - t0:.1f}s on {model.device}", flush=True)

    # Precompute reusable voice clone prompt for each role (super fast for short cues!)
    role_prompts = {}
    has_prompt_api = hasattr(model, "create_voice_clone_prompt")
    if has_prompt_api:
        for role, (r_wav, r_txt) in role_refs.items():
            role_prompts[role] = model.create_voice_clone_prompt(
                ref_audio=str(r_wav),
                ref_text=r_txt,
            )
        print(f"[pyVideoTrans Qwen3-TTS] Cached {len(role_prompts)} role prompts.", flush=True)

    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        lang_name = LANG_MAP[lang]
        lang_dir = args.out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        tl = time.time()
        print(f"[Qwen3-TTS] Cloning {len(lines)} lines -> {lang_name}...", flush=True)

        # If prompt API exists, batch per role sorted by duration with bounded max_new_tokens
        if has_prompt_api:
            for role in ("NAM_TRE", "NAM_GIA", "NU_TRE", "NU_GIA"):
                grp = [x for x in lines if x["voice_role"] == role and x.get(lang, "").strip()]
                if not grp:
                    continue
                grp.sort(key=lambda x: len(x[lang]))
                for i in range(0, len(grp), 8):
                    chunk = grp[i : i + 8]
                    texts = [x[lang].strip() for x in chunk]
                    max_slot = max(parse_ts(x["end"]) - parse_ts(x["start"]) for x in chunk)
                    max_toks = min(135, max(36, int(max_slot * 18) + 28))
                    with torch.inference_mode():
                        try:
                            wavs, out_sr = model.generate_voice_clone(
                                text=texts,
                                language=[lang_name] * len(texts),
                                voice_clone_prompt=role_prompts[role],
                                non_streaming_mode=True,
                                max_new_tokens=max_toks,
                            )
                        except Exception:
                            wavs, out_sr = [], None
                            for x_one, t in zip(chunk, texts):
                                slot_one = parse_ts(x_one["end"]) - parse_ts(x_one["start"])
                                tok_one = min(135, max(36, int(slot_one * 18) + 28))
                                w_one, sr_one = model.generate_voice_clone(
                                    text=t,
                                    language=lang_name,
                                    voice_clone_prompt=role_prompts[role],
                                    non_streaming_mode=True,
                                    max_new_tokens=tok_one,
                                )
                                wavs.append(w_one[0])
                                out_sr = sr_one
                    for x, w in zip(chunk, wavs):
                        sf.write(str(lang_dir / f"{x['id']:04d}.wav"), np.asarray(w, dtype=np.float32), out_sr)
                    del wavs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                for x in lines:
                    txt = x.get(lang, "").strip()
                    if not txt:
                        continue
                    r_wav, r_txt = line_refs[x["id"]]
                    slot_one = parse_ts(x["end"]) - parse_ts(x["start"])
                    tok_one = min(135, max(36, int(slot_one * 18) + 28))
                    wavs, out_sr = model.generate_voice_clone(
                        text=txt,
                        language=lang_name,
                        ref_audio=str(r_wav),
                        ref_text=r_txt,
                        non_streaming_mode=True,
                        max_new_tokens=tok_one,
                    )
                    sf.write(str(lang_dir / f"{x['id']:04d}.wav"), np.asarray(wavs[0], dtype=np.float32), out_sr)

        print(f"[OK] Cloned {lang_name} in {time.time() - tl:.1f}s -> {lang_dir}", flush=True)


if __name__ == "__main__":
    main()

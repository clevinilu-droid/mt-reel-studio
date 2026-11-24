"""
Worker chạy bằng /opt/pyvideotrans/.venv/bin/python (pyVideoTrans v4.19 · qwen-tts-pvt):
Dùng Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice sinh giọng NỮ bản địa tự nhiên theo từng quốc gia
cho EN / JA / KO (không clone giọng gốc) trên GPU CUDA:
  - English:  Serena
  - Japanese: Ono_anna
  - Korean:   Sohee
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


LANG_SPEC = {
    "en": ("English", "Serena"),
    "ja": ("Japanese", "Ono_anna"),
    "ko": ("Korean", "Sohee"),
}


def parse_ts(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def main():
    ap = argparse.ArgumentParser(description="Generate native female voices per country (EN/JA/KO) via pyVideoTrans Qwen3-TTS CustomVoice.")
    ap.add_argument("role_lines", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--langs", default="en,ja,ko")
    ap.add_argument("--model-dir", default="/opt/pyvideotrans/models/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lines = json.loads(args.role_lines.read_text(encoding="utf-8"))

    t0 = time.time()
    local_dir = Path(args.model_dir)
    if not (local_dir / "config.json").exists():
        from huggingface_hub import snapshot_download
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pyVideoTrans Qwen3-TTS] Downloading CustomVoice -> {local_dir}...", flush=True)
        snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", local_dir=str(local_dir))

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained(str(local_dir), device_map="auto", dtype=dtype)
    print(f"[pyVideoTrans Qwen3-TTS] CustomVoice loaded in {time.time() - t0:.1f}s on {model.device}", flush=True)

    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        if lang not in LANG_SPEC:
            continue
        lang_name, speaker = LANG_SPEC[lang]
        lang_dir = args.out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        tl = time.time()

        items = [x for x in lines if x.get(lang, "").strip()]
        items.sort(key=lambda x: len(x[lang]))
        print(f"[Qwen3-TTS CustomVoice] {lang_name} ({speaker}): {len(items)} lines...", flush=True)

        for i in range(0, len(items), 8):
            chunk = items[i : i + 8]
            texts = [x[lang].strip() for x in chunk]
            max_slot = max(parse_ts(x["end"]) - parse_ts(x["start"]) for x in chunk)
            max_toks = min(135, max(36, int(max_slot * 18) + 28))
            with torch.inference_mode():
                try:
                    wavs, out_sr = model.generate_custom_voice(
                        text=texts,
                        speaker=[speaker] * len(texts),
                        language=[lang_name] * len(texts),
                        non_streaming_mode=True,
                        max_new_tokens=max_toks,
                    )
                except Exception:
                    wavs, out_sr = [], None
                    for x_one, t in zip(chunk, texts):
                        slot_one = parse_ts(x_one["end"]) - parse_ts(x_one["start"])
                        tok_one = min(135, max(36, int(slot_one * 18) + 28))
                        w_one, sr_one = model.generate_custom_voice(
                            text=t,
                            speaker=speaker,
                            language=lang_name,
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

        print(f"[OK] {lang_name} ({speaker}) finished in {time.time() - tl:.1f}s -> {lang_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
Bước 3.4: Khớp sóng âm (Audio-Snap), gộp câu dài bị cắt đôi và đo giới tính (F0 Hz) trên file vocals sạch.
Đầu vào: zh_hardsub_raw.srt + vocals (.flac/.wav)
Đầu ra: zh_master_synced.srt + voice_profile.json
"""
import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n([\s\S]*?)(?=\n{2,}|\Z)"
)


def parse_ts(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_srt(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    out = []
    for m in SRT_BLOCK_RE.finditer(raw):
        out.append({
            "id": int(m.group(1)),
            "start": parse_ts(m.group(2)),
            "end": parse_ts(m.group(3)),
            "text": " ".join(line.strip() for line in m.group(4).splitlines() if line.strip()),
        })
    return out


def snap_bounds(audio: np.ndarray, sr: int, start: float, end: float, pad: float = 0.25) -> tuple[float, float, float]:
    """Snap [start, end] to actual speech energy onset/offset within +-pad seconds and compute RMS."""
    n = len(audio)
    w_s = max(0, int((start - pad) * sr))
    w_e = min(n, int((end + pad) * sr))
    if w_e <= w_s + int(0.05 * sr):
        return start, end, 0.0

    win = audio[w_s:w_e]
    hop = int(0.01 * sr)  # 10ms frames
    frame_len = int(0.025 * sr)
    rms = librosa.feature.rms(y=win, frame_length=frame_len, hop_length=hop)[0]
    peak = float(np.max(rms)) if len(rms) else 0.0
    if peak < 0.008:
        return start, end, peak

    thresh = max(0.008, peak * 0.12)
    active = np.where(rms >= thresh)[0]
    if len(active) == 0:
        return start, end, peak

    new_s = max(0.0, (w_s + active[0] * hop) / sr - 0.03)
    new_e = min(n / sr, (w_s + active[-1] * hop + frame_len) / sr + 0.04)
    if new_e - new_s < 0.18:
        return start, end, peak
    return round(new_s, 3), round(new_e, 3), round(peak, 4)


def estimate_voice_profile(seg: np.ndarray, sr: int) -> tuple[float, str]:
    """Estimate median F0 (Hz) and classify NAM / NU / CHU_THICH_CAM."""
    if len(seg) < int(0.08 * sr):
        return 0.0, "CHU_THICH_CAM"
    rms = float(np.sqrt(np.mean(seg ** 2)))
    if rms < 0.004:
        return 0.0, "CHU_THICH_CAM"

    f0 = librosa.yin(seg, fmin=65.0, fmax=420.0, sr=sr, frame_length=1024, hop_length=256)
    valid = f0[(f0 >= 70.0) & (f0 <= 400.0)]
    if len(valid) < 3:
        return 0.0, "UNKNOWN"
    med_f0 = float(np.median(valid))
    gender = "NAM" if med_f0 < 175.0 else "NU"
    return round(med_f0, 1), gender


def main():
    ap = argparse.ArgumentParser(description="Audio-Snap hardsub SRT against clean vocals and tag speaker gender/F0.")
    ap.add_argument("raw_srt", type=Path, help="Input zh_hardsub_raw.srt")
    ap.add_argument("vocals", type=Path, help="Clean vocals audio (.flac or .wav)")
    ap.add_argument("out_dir", type=Path, help="Output directory")
    ap.add_argument("--merge-gap", type=float, default=0.16, help="Max gap (s) to merge split continuous lines")
    args = ap.parse_args()

    audio, sr = sf.read(str(args.vocals), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    items = load_srt(args.raw_srt)
    snapped = []
    for it in items:
        ns, ne, peak = snap_bounds(audio, sr, it["start"], it["end"])
        seg = audio[int(ns * sr) : int(ne * sr)]
        f0, tag = estimate_voice_profile(seg, sr)
        snapped.append({
            "start": ns,
            "end": ne,
            "text": it["text"],
            "f0_hz": f0,
            "tag": tag,
            "rms_peak": peak,
        })

    # Merge split lines if gap <= merge_gap, same gender tag, continuous energy, and no terminal punctuation
    merged = []
    for cur in snapped:
        if (
            merged
            and cur["tag"] in ("NAM", "NU")
            and merged[-1]["tag"] == cur["tag"]
            and (cur["start"] - merged[-1]["end"]) <= args.merge_gap
            and abs(cur["f0_hz"] - merged[-1]["f0_hz"]) <= 45.0
            and not merged[-1]["text"].endswith(("。", "？", "！", ".", "?", "!"))
            and len(merged[-1]["text"]) + len(cur["text"]) <= 26
        ):
            merged[-1]["end"] = max(merged[-1]["end"], cur["end"])
            merged[-1]["text"] = merged[-1]["text"] + "，" + cur["text"]
            seg = audio[int(merged[-1]["start"] * sr) : int(merged[-1]["end"] * sr)]
            f0, tag = estimate_voice_profile(seg, sr)
            merged[-1]["f0_hz"] = f0
            merged[-1]["tag"] = tag
        else:
            merged.append(cur)

    # Prevent overlap with next line
    for i in range(len(merged) - 1):
        if merged[i]["end"] > merged[i + 1]["start"]:
            mid = round((merged[i]["end"] + merged[i + 1]["start"]) / 2, 3)
            merged[i]["end"] = mid
            merged[i + 1]["start"] = mid

    args.out_dir.mkdir(parents=True, exist_ok=True)
    srt_lines = []
    for idx, m in enumerate(merged, 1):
        m["id"] = idx
        srt_lines.append(f"{idx}\n{fmt_ts(m['start'])} --> {fmt_ts(m['end'])}\n{m['text']}\n")

    out_srt = args.out_dir / "zh_master_synced.srt"
    out_prof = args.out_dir / "voice_profile.json"
    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    out_prof.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {k: sum(1 for x in merged if x["tag"] == k) for k in ("NAM", "NU", "UNKNOWN", "CHU_THICH_CAM")}
    print(f"[OK] {len(items)} raw -> {len(merged)} synced lines -> {out_srt.name} | Tags: {counts}")


if __name__ == "__main__":
    main()

"""
Step 2 (Offline Zero-Shot Voice Cloning): Clone each character's actual voice from
10-13s reference clips extracted from _engine/work/<movie>/02_vocals_mdx.wav using
VieNeu-TTS (v3turbo, 48kHz ONNX CPU), apply Strict Frame Sync (never drift across cuts),
mix in full 48kHz stereo over 02_instrumental_mdx.wav, and output strictly 1 MP4 file:
  output/<movie>/<movie>.mp4

Usage:
  python _engine/scripts/step2_vieneu_clone_dub.py <movie_out_dir> [--force]
"""
import argparse
import hashlib
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step2_dub_and_render import parse_srt, probe_duration, render_clean_16x9_video
from step2_gemini_tts_dub import (
    LANG_CONFIG,
    group_entries_into_turns,
    load_cast_and_emotions,
    load_stereo_bgm,
)

SAMPLE_RATE = 48000


def prepare_strict_pcm(wav_path: Path, slot_dur: float) -> np.ndarray:
    """Trim silence and fit strictly inside slot_dur (up to 1.48x) so dialogue never drifts into next shot."""
    trimmed = wav_path.with_suffix(".trim.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav_path),
            "-af", "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB,areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB,areverse",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(trimmed),
        ],
        check=True,
    )
    with wave.open(str(trimmed), "rb") as wf:
        actual_dur = wf.getnframes() / SAMPLE_RATE

    src_wav = trimmed
    if slot_dur > 0.35 and actual_dur > slot_dur:
        tempo = min(actual_dur / slot_dur, 1.48)
        if tempo > 1.02:
            fitted = wav_path.with_suffix(".fit.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(trimmed),
                    "-af", f"atempo={tempo:.4f}",
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                    str(fitted),
                ],
                check=True,
            )
            src_wav = fitted

    with wave.open(str(src_wav), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    trimmed.unlink(missing_ok=True)
    wav_path.with_suffix(".fit.wav").unlink(missing_ok=True)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def mix_strict_stereo(
    turns: list[dict],
    cache_dir: Path,
    turn_files: list[Path],
    bgm_stereo: np.ndarray,
    total_dur: float,
    out_m4a: Path,
):
    total_frames = len(bgm_stereo)
    action_gain = 0.92
    speech_gain = 0.45
    ramp_frames = int(0.16 * SAMPLE_RATE)

    gain = np.full(total_frames, action_gain, dtype=np.float32)
    speech_mono = np.zeros(total_frames, dtype=np.float32)

    cursor_end = 0
    placed = 0
    max_drift = 0.0

    for idx, turn in enumerate(turns):
        wav_path = turn_files[idx]
        if not wav_path.exists() or wav_path.stat().st_size <= 2000:
            continue

        start_s = turn["start"]
        next_start = turns[idx + 1]["start"] if idx + 1 < len(turns) else total_dur
        # Strict window: never spill past next turn's start
        hard_end = min(next_start - 0.05, max(turn["end"] + 0.35, turn["start"] + 0.6))
        slot_dur = max(0.35, hard_end - start_s)

        samples = prepare_strict_pcm(wav_path, slot_dur)
        if len(samples) == 0:
            continue

        start_idx = int(round(start_s * SAMPLE_RATE))
        min_idx = cursor_end + int(0.03 * SAMPLE_RATE) if cursor_end > 0 else 0
        if start_idx < min_idx:
            start_idx = min_idx
        if start_idx >= total_frames:
            break

        drift = (start_idx / SAMPLE_RATE) - start_s
        if drift > max_drift:
            max_drift = drift

        end_idx = min(total_frames, start_idx + len(samples))
        copy_len = end_idx - start_idx
        speech_mono[start_idx:end_idx] += samples[:copy_len]
        cursor_end = end_idx
        placed += 1

        d0 = max(0, start_idx - ramp_frames)
        d1 = min(total_frames, end_idx + ramp_frames)
        r_in = start_idx - d0
        r_out = d1 - end_idx
        if r_in > 0:
            gain[d0:start_idx] = np.minimum(gain[d0:start_idx], np.linspace(action_gain, speech_gain, r_in, dtype=np.float32))
        gain[start_idx:end_idx] = np.minimum(gain[start_idx:end_idx], speech_gain)
        if r_out > 0:
            gain[end_idx:d1] = np.minimum(gain[end_idx:d1], np.linspace(speech_gain, action_gain, r_out, dtype=np.float32))

    mixed = bgm_stereo * gain[:, None] + speech_mono[:, None]
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.96:
        mixed = mixed * (0.96 / peak)

    pcm_out = np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16)
    raw_wav = cache_dir / f"mix_{out_m4a.stem}.wav"
    with wave.open(str(raw_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_out.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_wav),
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={total_dur:.3f}",
            "-t", f"{total_dur:.3f}",
            "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-c:a", "aac", "-b:a", "192k",
            str(out_m4a),
        ],
        check=True,
    )
    raw_wav.unlink(missing_ok=True)
    out_dur = probe_duration(out_m4a)
    print(f"[OK] Strict-mixed {out_m4a.name} ({placed}/{len(turns)} turns, {out_dur:.2f}s stereo, max_drift=+{max_drift:.2f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Offline VieNeu-TTS zero-shot character voice cloning + MDX-Net dubbing.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--max-gap", type=float, default=0.55, help="Max gap (seconds) to group same-speaker lines")
    parser.add_argument("--max-chars", type=int, default=150, help="Max characters per turn for tight lip/cut sync")
    parser.add_argument("--max-span", type=float, default=6.5, help="Max turn duration (seconds) before splitting")
    parser.add_argument("--force", action="store_true", help="Re-synthesize all turns ignoring cache")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = SCRIPTS_DIR.parent / "work" / out_dir.name
    tracks_dir = work_dir / "tracks"
    refs_dir = work_dir / "refs"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    merged_mp4 = work_dir / "01_merged_raw.mp4"
    inst_wav = work_dir / "02_instrumental_mdx.wav"
    vi_srt = tracks_dir / "vi.srt"
    if not merged_mp4.exists() or not inst_wav.exists() or not vi_srt.exists():
        raise FileNotFoundError(f"Missing required files in {work_dir}")

    _, line_meta = load_cast_and_emotions(out_dir, work_dir)
    entries = parse_srt(vi_srt)
    turns = group_entries_into_turns(entries, line_meta, "vi", args.max_gap, args.max_chars, args.max_span)
    print(f"[VIENEU CLONE] {len(entries)} lines -> {len(turns)} tight scene-synced turns", flush=True)

    cache_dir = work_dir / "_vieneu_tts_cache_vi"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from vieneu import Vieneu
    tts = None
    turn_files = []
    synth_count = 0

    for t in turns:
        role = t["role"]
        ref_wav = refs_dir / f"ref_{role}.wav"
        if not ref_wav.exists():
            ref_wav = refs_dir / "ref_M_LEAD.wav"
        key_hash = hashlib.sha256(f"v3turbo|{ref_wav.name}|{t['text']}".encode("utf-8")).hexdigest()[:8]
        out_wav = cache_dir / f"turn_{t['turn_idx']:04d}_{key_hash}.wav"
        turn_files.append(out_wav)

        if not args.force and out_wav.exists() and out_wav.stat().st_size > 2000:
            continue

        if tts is None:
            t0 = time.time()
            print("[INFO] Loading VieNeu-TTS (v3turbo, CPU ONNX)...", flush=True)
            tts = Vieneu(mode="v3turbo")
            print(f"[OK] Loaded in {time.time() - t0:.1f}s", flush=True)

        ts = time.time()
        wav = tts.infer(t["text"], ref_audio=str(ref_wav))
        tts.save(wav, str(out_wav))
        synth_count += 1
        print(f"  [CLONE {t['turn_idx']+1}/{len(turns)}] {role} ({ref_wav.name}) in {time.time()-ts:.1f}s -> {out_wav.name}", flush=True)

    if tts is not None:
        tts.close()

    print(f"[INFO] Synthesis done ({synth_count} new, {len(turns)-synth_count} cached).", flush=True)

    total_dur = probe_duration(merged_mp4)
    total_frames = int(round(total_dur * SAMPLE_RATE))
    bgm_stereo = load_stereo_bgm(inst_wav, total_frames)

    vi_m4a = tracks_dir / "audio_vi.m4a"
    mix_strict_stereo(turns, cache_dir, turn_files, bgm_stereo, total_dur, vi_m4a)

    base_video = work_dir / "03_clean_16x9_video.mp4"
    if not base_video.exists():
        render_clean_16x9_video(merged_mp4, vi_m4a, base_video, (40, 1410, 1000, 190))

    final_mp4 = out_dir / f"{out_dir.name}.mp4"
    tmp_mp4 = work_dir / f"{out_dir.name}.tmp.mp4"
    iso = LANG_CONFIG["vi"]["iso3"]
    label = LANG_CONFIG["vi"]["label"]
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(base_video), "-i", str(vi_m4a),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            f"-metadata:s:a:0", f"language={iso}",
            f"-metadata:s:a:0", f"handler_name={label}",
            f"-metadata:s:a:0", f"title={label}",
            "-disposition:a:0", "default",
            "-movflags", "+faststart", "-shortest",
            str(tmp_mp4),
        ],
        check=True,
    )
    tmp_mp4.replace(final_mp4)
    print(f"[ALL DONE] Single cloned-voice output ready: {final_mp4} ({final_mp4.stat().st_size/1024/1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()

"""
Step 1: Merge downloaded episodes (1..N) into a normalized MP4 safely in chunks
(prevents OOM when merging 50-120 episodes) and optionally run faster-whisper ASR.

Usage:
  python scripts/step1_merge_and_transcribe.py <episodes_dir> <start_ep> <end_ep> <project_out_dir> [whisper_model] [--merge-only] [--encoder libx264|h264_nvenc]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def format_ts(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def find_episodes(ep_dir: Path, start_ep: int, end_ep: int):
    found = []
    for f in sorted(ep_dir.glob("*.mp4")):
        m = re.search(r"(\d+)\.mp4$", f.name)
        if not m:
            continue
        num = int(m.group(1))
        if start_ep <= num <= end_ep:
            found.append((num, f))
    found.sort(key=lambda x: x[0])
    return [f for _, f in found]


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def merge_episode_batch(batch: list[Path], out_mp4: Path, encoder: str):
    """Normalize and concatenate up to ~12 episodes safely without opening 100 decoders."""
    cmd = ["ffmpeg", "-y", "-v", "warning"]
    for e in batch:
        cmd += ["-i", str(e.resolve())]

    filter_parts = []
    concat_inputs = []
    for idx in range(len(batch)):
        filter_parts.append(
            f"[{idx}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,fps=30,setsar=1,format=yuv420p[v{idx}];"
            f"[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]"
        )
        concat_inputs.append(f"[v{idx}][a{idx}]")

    filter_complex = ";".join(filter_parts) + f";{''.join(concat_inputs)}concat=n={len(batch)}:v=1:a=1[vout][aout]"
    if encoder == "h264_nvenc":
        v_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19"]
    else:
        v_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        *v_args,
        "-video_track_timescale", "15360",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Merge episodes safely in chunks and optionally run faster-whisper.")
    parser.add_argument("ep_dir", type=Path)
    parser.add_argument("start_ep", type=int)
    parser.add_argument("end_ep", type=int)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("model_name", nargs="?", default="small")
    parser.add_argument("--merge-only", action="store_true", help="Stop after 01_merged_raw.mp4 (when using OCR subtitles)")
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    args = parser.parse_args()

    ep_dir = args.ep_dir
    out_dir = args.out_dir
    model_name = args.model_name

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(__file__).resolve().parent.parent / "work" / out_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    merged_mp4 = work_dir / "01_merged_raw.mp4"

    if merged_mp4.exists() and merged_mp4.stat().st_size > 1_000_000:
        dur = probe_duration(merged_mp4)
        print(f"[SKIP] Merged video already exists ({dur:.2f}s): {merged_mp4}", flush=True)
    elif ep_dir.is_file():
        print(f"[INFO] Using pre-merged video directly: {ep_dir.name} -> {merged_mp4.name}", flush=True)
        shutil.copy2(ep_dir, merged_mp4)
        dur = probe_duration(merged_mp4)
        print(f"[OK] Pre-merged video ready: {dur:.2f}s ({dur/60:.2f} min)", flush=True)
    else:
        episodes = find_episodes(ep_dir, args.start_ep, args.end_ep)
        if not episodes:
            raise RuntimeError(f"No episodes found in {ep_dir} for range {args.start_ep}-{args.end_ep}")
        print(f"[INFO] Found {len(episodes)} episodes ({episodes[0].name} .. {episodes[-1].name}).", flush=True)

        batch_size = 12
        if len(episodes) <= batch_size:
            print(f"[INFO] Merging {len(episodes)} episodes (encoder={args.encoder}) -> {merged_mp4.name}...", flush=True)
            merge_episode_batch(episodes, merged_mp4, args.encoder)
        else:
            chunk_files = []
            tmp_dir = work_dir / "_merge_chunks"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for b_idx in range(0, len(episodes), batch_size):
                batch = episodes[b_idx : b_idx + batch_size]
                chunk_mp4 = tmp_dir / f"chunk_{b_idx // batch_size:03d}.mp4"
                if not chunk_mp4.exists() or chunk_mp4.stat().st_size < 100_000:
                    print(f"[CHUNK] Merging episodes {b_idx+1}..{b_idx+len(batch)} ({args.encoder}) -> {chunk_mp4.name}...", flush=True)
                    merge_episode_batch(batch, chunk_mp4, args.encoder)
                chunk_files.append(chunk_mp4)
            concat_txt = tmp_dir / "chunks.txt"
            concat_txt.write_text("\n".join(f"file '{c.resolve().as_posix()}'" for c in chunk_files), encoding="utf-8")
            print(f"[CONCAT] Joining {len(chunk_files)} chunks losslessly -> {merged_mp4.name}...", flush=True)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "warning", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(merged_mp4)],
                check=True,
            )
            for c in chunk_files:
                c.unlink(missing_ok=True)
            concat_txt.unlink(missing_ok=True)
            tmp_dir.rmdir()
        dur = probe_duration(merged_mp4)
        print(f"[OK] Merged video duration: {dur:.2f}s ({dur/60:.2f} min)", flush=True)

    if args.merge_only:
        return

    # 2. Extract 16kHz mono WAV for fast Whisper ASR
    wav_path = work_dir / "01_audio_16k.wav"
    if not wav_path.exists() or wav_path.stat().st_size < 100_000:
        print(f"[INFO] Extracting 16kHz mono audio -> {wav_path} ...", flush=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(merged_mp4),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )

    # 3. Load WAV directly as float32 numpy array (bypasses PyAV version incompatibilities)
    import wave
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())
    audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # 4. Transcribe with faster-whisper (CPU int8)
    print(f"[INFO] Loading faster-whisper model '{model_name}' (CPU int8)...", flush=True)
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    segments, info = model.transcribe(
        audio_np,
        language="zh",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        beam_size=5,
        initial_prompt="以下是中文短剧对白，请使用简体中文标点。",
    )

    srt_path = work_dir / "zh.srt"
    count = 0
    with srt_path.open("w", encoding="utf-8") as fh:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            count += 1
            fh.write(f"{count}\n")
            fh.write(f"{format_ts(seg.start)} --> {format_ts(seg.end)}\n")
            fh.write(f"{text}\n\n")
            if count % 25 == 0:
                print(f"  [ASR] {count} lines transcribed (at {seg.end:.1f}s / {dur:.1f}s)...", flush=True)

    print(f"[DONE] Transcribed {count} Chinese lines -> {srt_path}")


if __name__ == "__main__":
    main()

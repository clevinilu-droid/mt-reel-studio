"""
Step 1b: Studio-quality vocal/instrumental separation using MDX-Net (python-audio-separator).
Uses streamed chunked separation (default 480s chunks + 2s overlap) for long movies (>10 min)
so 2-3 hour dramas run safely on 12 GB RAM without OOM while keeping 48kHz sample-exact sync.

Outputs:
  - _work/02_instrumental_mdx.wav (100% original stereo BGM + SFX, vocals removed)
  - _work/02_vocals_mdx.wav (isolated Chinese dialogue)

Usage:
  python _engine/scripts/step1b_separate_mdx.py <movie_out_dir> [--force] [--chunk-sec 480]
"""
import argparse
import shutil
import subprocess
import time
import wave
from pathlib import Path

MODEL_NAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_BYTES = CHANNELS * SAMPLE_WIDTH


def extract_orig_stereo_wav(merged_mp4: Path, out_wav: Path):
    """Extract 48kHz stereo PCM16 WAV from merged video."""
    print(f"[INFO] Extracting 48kHz stereo WAV from {merged_mp4.name} ...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(merged_mp4),
            "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True,
    )


def _match_outputs(output_files: list[str], search_dir: Path) -> tuple[Path, Path]:
    """Locate instrumental and vocals stems produced by audio-separator."""
    inst_src = None
    voc_src = None
    for f in output_files:
        p = Path(f)
        if not p.is_absolute():
            p = search_dir / p
        low = p.name.lower()
        if "(instrumental)" in low:
            inst_src = p
        elif "(vocals)" in low:
            voc_src = p
    if not inst_src or not inst_src.exists() or not voc_src or not voc_src.exists():
        raise RuntimeError(f"Unexpected output filenames from audio-separator: {output_files}")
    return inst_src, voc_src


def _append_trimmed_pcm(src_wav: Path, dst_wf: wave.Wave_write, left_trim_frames: int, keep_frames: int):
    """Read exact non-overlapping [left_trim_frames : left_trim_frames + keep_frames] and append to dst_wf."""
    with wave.open(str(src_wav), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != CHANNELS or wf.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(f"Unexpected WAV format in {src_wav}: {wf.getparams()}")
        total = wf.getnframes()
        start = min(left_trim_frames, total)
        avail = max(0, total - start)
        to_read = min(keep_frames, avail)
        wf.setpos(start)
        raw = wf.readframes(to_read)
    if to_read < keep_frames:
        raw += b"\x00" * ((keep_frames - to_read) * FRAME_BYTES)
    dst_wf.writeframes(raw)


def _separate_chunked(separator, orig_wav: Path, inst_out: Path, voc_out: Path, chunk_sec: float, overlap_sec: float):
    """Stream long WAV in bounded chunks into wave outputs to keep RAM under ~1 GB."""
    tmp_dir = orig_wav.parent / "_mdx_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    separator.output_dir = str(tmp_dir)

    with wave.open(str(orig_wav), "rb") as src_wf:
        total_frames = src_wf.getnframes()
        chunk_frames = int(round(chunk_sec * SAMPLE_RATE))
        overlap_frames = int(round(overlap_sec * SAMPLE_RATE))
        total_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)
        print(f"[INFO] Streaming MDX across {total_chunks} chunk(s) ({chunk_sec:.0f}s + {overlap_sec:.0f}s overlap)...", flush=True)

        inst_tmp = inst_out.with_suffix(".part.wav")
        voc_tmp = voc_out.with_suffix(".part.wav")
        with wave.open(str(inst_tmp), "wb") as iw, wave.open(str(voc_tmp), "wb") as vw:
            for w in (iw, vw):
                w.setnchannels(CHANNELS)
                w.setsampwidth(SAMPLE_WIDTH)
                w.setframerate(SAMPLE_RATE)

            for idx in range(total_chunks):
                core_start = idx * chunk_frames
                core_end = min(total_frames, core_start + chunk_frames)
                keep_frames = core_end - core_start

                pad_start = max(0, core_start - overlap_frames)
                pad_end = min(total_frames, core_end + overlap_frames)
                left_trim = core_start - pad_start

                chunk_in = tmp_dir / f"chunk_{idx:03d}.wav"
                src_wf.setpos(pad_start)
                with wave.open(str(chunk_in), "wb") as cw:
                    cw.setnchannels(CHANNELS)
                    cw.setsampwidth(SAMPLE_WIDTH)
                    cw.setframerate(SAMPLE_RATE)
                    cw.writeframes(src_wf.readframes(pad_end - pad_start))

                t_c = time.time()
                out_files = separator.separate(str(chunk_in))
                inst_stem, voc_stem = _match_outputs(out_files, tmp_dir)
                _append_trimmed_pcm(inst_stem, iw, left_trim, keep_frames)
                _append_trimmed_pcm(voc_stem, vw, left_trim, keep_frames)

                for p in (chunk_in, inst_stem, voc_stem):
                    p.unlink(missing_ok=True)
                print(f"  [MDX] Chunk {idx+1}/{total_chunks} done in {time.time()-t_c:.1f}s ({core_end/SAMPLE_RATE:.0f}s / {total_frames/SAMPLE_RATE:.0f}s)", flush=True)

    shutil.move(str(inst_tmp), str(inst_out))
    shutil.move(str(voc_tmp), str(voc_out))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Separate BGM/SFX and Chinese vocals using MDX-Net.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--chunk-sec", type=float, default=480.0, help="Chunk duration in seconds for streaming separation (default: 480)")
    parser.add_argument("--overlap-sec", type=float, default=2.0, help="Overlap padding per chunk edge in seconds (default: 2.0)")
    parser.add_argument("--force", action="store_true", help="Re-run even if MDX outputs already exist")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    work_dir = Path(__file__).resolve().parent.parent / "work" / out_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)

    merged_mp4 = work_dir / "01_merged_raw.mp4"
    if not merged_mp4.exists():
        merged_mp4 = out_dir / "01_merged_raw.mp4"
    if not merged_mp4.exists():
        raise FileNotFoundError(f"Missing merged video: {merged_mp4}")

    inst_out = work_dir / "02_instrumental_mdx.wav"
    voc_out = work_dir / "02_vocals_mdx.wav"
    if not args.force and inst_out.exists() and voc_out.exists() and inst_out.stat().st_size > 1_000_000 and voc_out.stat().st_size > 1_000_000:
        print(f"[SKIP] MDX outputs already exist:\n  - {inst_out}\n  - {voc_out}", flush=True)
        return

    orig_wav = work_dir / "01_orig_stereo.wav"
    extract_orig_stereo_wav(merged_mp4, orig_wav)

    model_dir = Path(__file__).resolve().parent.parent / "models" / "mdx"
    model_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(orig_wav), "rb") as wf:
        dur_s = wf.getnframes() / float(wf.getframerate())

    t0 = time.time()
    print(f"[INFO] Initializing Separator (model={MODEL_NAME}, dur={dur_s:.1f}s)...", flush=True)
    from audio_separator.separator import Separator
    separator = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(work_dir),
        output_format="WAV",
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": False,
        },
    )
    separator.load_model(model_filename=MODEL_NAME)

    if dur_s <= args.chunk_sec + 60.0:
        print(f"[INFO] Separating {orig_wav.name} in single pass...", flush=True)
        output_files = separator.separate(str(orig_wav))
        inst_src, voc_src = _match_outputs(output_files, work_dir)
        inst_out.unlink(missing_ok=True)
        voc_out.unlink(missing_ok=True)
        shutil.move(str(inst_src), str(inst_out))
        shutil.move(str(voc_src), str(voc_out))
    else:
        _separate_chunked(separator, orig_wav, inst_out, voc_out, args.chunk_sec, args.overlap_sec)

    orig_wav.unlink(missing_ok=True)
    print(f"[OK] Separation finished in {time.time() - t0:.1f}s -> {inst_out.name}, {voc_out.name}", flush=True)


if __name__ == "__main__":
    main()

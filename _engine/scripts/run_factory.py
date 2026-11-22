"""
MT Reel Industrial Multi-Language Video Factory (1-Command End-to-End Orchestrator).

Input:
  - A single merged .mp4 file (--input video.mp4 --name 001_BaoBiao_Full)
  - OR a batch directory containing multiple merged .mp4 files (--batch-dir path/to/inputs)

Output per movie in output/<MOVIE_NAME>/:
  - <MOVIE_NAME>.mp4               (1080p HQ video + Vietnamese dub + watermark + Golden Hook + soft sub)
  - <MOVIE_NAME>.srt               (Hook-synced Vietnamese SRT)
  - subtitles/<lang>.srt           (Hook-synced subtitles for all requested languages: vi, en, ja, ko, es...)
  - audio/audio_<lang>.m4a         (48kHz stereo crowd-preserving audio tracks, 0.000s synced to <MOVIE_NAME>.mp4)

Auto-detects best hardware on Vast.ai / RunPod / Kaggle / Local Windows:
  - Encoder: h264_nvenc (CUDA NVENC) -> h264_qsv (Intel QuickSync) -> libx264 (fast fallback)
  - OCR: CUDAExecutionProvider (--cuda) -> 4-worker CPU

Usage:
  python _engine/scripts/run_factory.py --input path/to/movie.mp4 --name 001_BaoBiao_Full --langs vi,en,ja,ko,es
  python _engine/scripts/run_factory.py --batch-dir input_queue/ --langs vi,en,ja,ko,es --clean-work-wavs
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
ROOT_DIR = ENGINE_DIR.parent


def detect_best_encoder() -> str:
    """Probe actual FFmpeg hardware encoding capability instead of only checking encoder names."""
    for enc, extra in (
        ("h264_nvenc", ["-preset", "p5", "-cq", "18"]),
        ("h264_qsv", ["-preset", "medium", "-global_quality", "18", "-look_ahead", "0"]),
    ):
        res = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=128x128:r=30",
             "-t", "0.5", "-pix_fmt", "yuv420p", "-c:v", enc, *extra, "-f", "null", "-"],
            text=True, capture_output=True,
        )
        if res.returncode == 0:
            return enc
    return "libx264"


def has_ort_cuda() -> bool:
    """Return True only if RapidOCR actually activates CUDAExecutionProvider (not silent CPU fallback)."""
    try:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from step1c_ocr_hardsub import probe_rapidocr_cuda
        return probe_rapidocr_cuda()
    except Exception:
        return False


def safe_movie_slug(stem: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return slug or f"movie_{int(time.time())}"


def run_step(script_name: str, *args):
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *[str(a) for a in args]]
    print(f"\n>>> [{script_name}] {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run(cmd, check=True)


def process_one_movie(
    input_source: Path,
    movie_name: str,
    langs: list[str],
    encoder: str,
    use_cuda: bool,
    hook_mode: str = "AUTO",
    no_watermark: bool = False,
    crf: int = 18,
    fps_ocr: float = 2.5,
    glossary: Path | None = None,
    clean_work_wavs: bool = False,
    force: bool = False,
):
    """Run complete pipeline from single MP4/episode dir to output/<movie>/ + subtitles/ + audio/."""
    t_start = time.time()
    out_dir = ROOT_DIR / "output" / movie_name
    work_dir = ENGINE_DIR / "work" / movie_name
    tracks_dir = work_dir / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"[FACTORY] Movie: {movie_name} | Encoder: {encoder} | CUDA OCR: {use_cuda} | Langs: {','.join(langs)}", flush=True)
    print("=" * 72, flush=True)

    # 1. Standardize raw input -> 01_merged_raw.mp4
    run_step("step1_merge_and_transcribe.py", input_source, 1, 999, out_dir, "--merge-only", "--encoder", encoder if encoder != "h264_qsv" else "libx264")

    # 2. Streaming MDX-Net separation (480s chunks, OOM-safe)
    mdx_args = [out_dir] + (["--force"] if force else [])
    run_step("step1b_separate_mdx.py", *mdx_args)

    # 3. Hardsub OCR (GPU CUDA on cloud / 4-worker CPU otherwise)
    zh_ocr = work_dir / "zh_ocr.srt"
    if force or not zh_ocr.exists() or zh_ocr.stat().st_size < 50:
        ocr_args = [out_dir, "--fps", str(fps_ocr)]
        if use_cuda:
            cuda_workers = max(2, min(4, os.cpu_count() or 4))
            ocr_args.extend(["--cuda", "--workers", str(cuda_workers)])
        else:
            cpu_workers = max(4, min(8, os.cpu_count() or 4))
            ocr_args.extend(["--workers", str(cpu_workers)])
        run_step("step1c_ocr_hardsub.py", *ocr_args)
    else:
        print(f"[SKIP] Existing zh_ocr.srt: {zh_ocr.name}", flush=True)

    # 4. 3-Layer Vietnamese translation + persistent Glossary
    vi_srt = tracks_dir / "vi.srt"
    if force or not vi_srt.exists() or vi_srt.stat().st_size < 50:
        t_args = [out_dir]
        if glossary:
            t_args.extend(["--glossary", glossary])
        run_step("step1d_translate_ocr_srt.py", *t_args)
    else:
        print(f"[SKIP] Existing tracks/vi.srt", flush=True)

    # 5. Translate target foreign languages (en, ja, ko, es...)
    foreign_langs = [x for x in langs if x != "vi"]
    if foreign_langs:
        ml_args = [out_dir, "--langs", ",".join(foreign_langs)] + (["--force"] if force else [])
        run_step("step1e_translate_multilang.py", *ml_args)

    # 6. Master Vietnamese VieNeu 4-voice dub + Render MP4
    final_mp4 = out_dir / f"{movie_name}.mp4"
    if force or not final_mp4.exists() or final_mp4.stat().st_size < 1_000_000:
        s2_args = [out_dir, "--encoder", encoder, "--crf", str(crf)]
        if encoder == "libx264":
            s2_args.extend(["--preset", "fast"])
        if hook_mode == "AUTO":
            s2_args.append("--hook-auto")
        elif hook_mode != "NONE" and ":" in hook_mode:
            s2_args.extend(["--hook-range", hook_mode])
        if no_watermark:
            s2_args.append("--no-watermark")
        if force:
            s2_args.append("--force")
        run_step("step2_vieneu_stock_dub.py", *s2_args)
    else:
        print(f"[SKIP] Existing master MP4: {final_mp4.name}", flush=True)

    # 7. Parallel Edge-TTS foreign dubs + hook-synced packaging into output/<movie>/audio & subtitles
    s2b_args = [out_dir, "--langs", ",".join(langs)] + (["--force"] if force else [])
    run_step("step2b_multilang_tracks.py", *s2b_args)

    # Save light metadata alongside output for archival/re-dubbing
    for extra in ("glossary.json", "relationship_bible.json", "render_meta.json"):
        src_meta = work_dir / extra
        if src_meta.exists():
            dst_meta = out_dir / extra
            dst_meta.write_text(src_meta.read_text(encoding="utf-8"), encoding="utf-8")

    # Optionally purge heavy intermediate WAVs (>3 GB) once verified
    if clean_work_wavs and final_mp4.exists():
        for wname in ("02_instrumental_mdx.wav", "02_vocals_mdx.wav", "01_merged_raw.mp4"):
            wp = work_dir / wname
            if wp.exists():
                wp.unlink()
        print("[CLEAN] Removed heavy intermediate WAV/raw copy while preserving glossary/meta.", flush=True)

    elapsed = (time.time() - t_start) / 60.0
    print("\n" + "=" * 72, flush=True)
    print(f"[DONE] {movie_name} finished in {elapsed:.1f} min -> {out_dir}", flush=True)
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(out_dir)} ({f.stat().st_size/1024/1024:.2f} MB)", flush=True)
    print("=" * 72, flush=True)


def main():
    parser = argparse.ArgumentParser(description="MT Reel Industrial Multi-Language Video Factory.")
    parser.add_argument("--input", type=Path, default=None, help="Single merged .mp4 file (or episode folder)")
    parser.add_argument("--name", type=str, default=None, help="Clean ASCII output folder name (e.g., 001_BaoBiao_Full)")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Directory of merged .mp4 files to process sequentially")
    parser.add_argument("--langs", type=str, default="vi,en,ja,ko,es", help="Comma-separated languages (default: vi,en,ja,ko,es)")
    parser.add_argument("--encoder", choices=("auto", "h264_nvenc", "h264_qsv", "libx264"), default="auto")
    parser.add_argument("--hook-mode", type=str, default="AUTO", help="AUTO | NONE | START:END")
    parser.add_argument("--no-watermark", action="store_true")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ocr-fps", type=float, default=2.5)
    parser.add_argument("--glossary", type=Path, default=None, help="Optional shared glossary.json from P1 for sequels P2/P3")
    parser.add_argument("--clean-work-wavs", action="store_true", help="Delete heavy work WAV/MP4 copies after verifying output")
    parser.add_argument("--force", action="store_true", help="Force regenerate all stages")
    args = parser.parse_args()

    if not args.input and not args.batch_dir:
        parser.error("Provide either --input <video.mp4> or --batch-dir <folder>")

    encoder = detect_best_encoder() if args.encoder == "auto" else args.encoder
    use_cuda = has_ort_cuda()
    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    if "vi" not in langs:
        langs = ["vi", *langs]

    jobs: list[tuple[Path, str]] = []
    if args.input:
        name = args.name or safe_movie_slug(args.input.stem)
        jobs.append((args.input.resolve(), name))
    else:
        for mp4 in sorted(args.batch_dir.resolve().glob("*.mp4")):
            jobs.append((mp4, safe_movie_slug(mp4.stem)))

    if not jobs:
        raise RuntimeError("No input videos found to process.")

    for src, name in jobs:
        process_one_movie(
            src, name, langs, encoder, use_cuda,
            hook_mode=args.hook_mode, no_watermark=args.no_watermark,
            crf=args.crf, fps_ocr=args.ocr_fps, glossary=args.glossary,
            clean_work_wavs=args.clean_work_wavs, force=args.force,
        )


if __name__ == "__main__":
    main()

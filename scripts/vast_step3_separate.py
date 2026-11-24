"""
Bước 3.1 (Chạy trên GPU Vast.ai với Image ghcr.io/clevinilu-droid/mt-reel-vast:latest):
Tách giọng nói & nhạc nền bằng audio-separator (Mel-Band-RoFormer InstVoc Duality V2)
rồi đóng gói Lossless 48kHz (.flac) bằng FFmpeg để giữ trọn chất lượng cho clone giọng & mix BGM.
"""
import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

from audio_separator.separator import Separator


MODEL_NAME = "melband_roformer_instvox_duality_v2.ckpt"


def main():
    ap = argparse.ArgumentParser(description="Separate clean vocals & instrumental at 48kHz FLAC using Mel-Band-RoFormer on GPU.")
    ap.add_argument("input_media", type=Path, help="Input .mp4 or .wav/.flac file")
    ap.add_argument("out_dir", type=Path, help="Output directory for vocals_48k_mono.flac and instrumental_48k.flac")
    ap.add_argument("--model-dir", default=os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "/opt/audio-separator-models"))
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_wav = tmp_path / "input_48k.wav"
        print("[3.1] Extracting 48kHz stereo WAV from input...", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(args.input_media),
            "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(raw_wav),
        ], check=True)

        print(f"[3.1] Loading {args.model} via audio-separator...", flush=True)
        sep = Separator(
            model_file_dir=args.model_dir,
            output_dir=str(tmp_path),
            output_format="WAV",
            sample_rate=48000,
            use_autocast=True,
        )
        sep.load_model(model_filename=args.model)
        out_files = sep.separate(str(raw_wav))
        print("[3.1] Separated raw stems:", out_files, flush=True)

        vocal_src = None
        instr_src = None
        for f in out_files:
            p = tmp_path / f if not Path(f).is_absolute() else Path(f)
            low = p.name.lower()
            if "(vocals)" in low or "vocal" in low:
                vocal_src = p
            elif "(instrumental)" in low or "instr" in low or "no_vocals" in low:
                instr_src = p

        if not vocal_src or not instr_src:
            raise RuntimeError(f"Could not identify vocal/instrumental stems from: {out_files}")

        out_vocals = args.out_dir / "vocals_48k_mono.flac"
        out_instr = args.out_dir / "instrumental_48k.flac"
        print("[3.1] Encoding lossless 48kHz FLAC outputs...", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(vocal_src),
            "-ac", "1", "-ar", "48000", "-c:a", "flac", "-compression_level", "5", str(out_vocals),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(instr_src),
            "-ac", "2", "-ar", "48000", "-c:a", "flac", "-compression_level", "5", str(out_instr),
        ], check=True)

    dt = time.time() - t0
    print(f"[OK] Step 3.1 done in {dt:.1f}s -> {out_vocals.name} ({out_vocals.stat().st_size/1e6:.1f} MB), {out_instr.name} ({out_instr.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

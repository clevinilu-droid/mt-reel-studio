"""
Bước 5 (Chạy trên GPU Vast.ai với Image ghcr.io/clevinilu-droid/mt-reel-vast:latest):
  1. Tiếng Việt: VieNeu-TTS v3 Turbo (48 kHz CUDA infer_batch) theo 4 vai (NAM_TRE, NAM_GIA, NU_TRE, NU_GIA).
  2. Anh / Nhật / Hàn (EN / JA / KO): pyVideoTrans v4.19 (Qwen3-TTS-12Hz-0.6B-Base trên /opt/pyvideotrans/.venv)
     clone trực tiếp giọng diễn viên gốc từ vocals_48k_mono.flac!
  3. Căn chỉnh thời lượng từng câu (cắt khoảng lặng thừa, tự co giãn vừa slot, chống đè lên câu kế),
     rồi trộn lên instrumental_48k.flac với ducking (-10.5 dB khi có lời thoại).
  4. Render video Master làm mờ dải hardsub gốc theo blur_config.json bằng NVENC (hevc_nvenc / h264_nvenc),
     KHÔNG burn sub mới vào hình; xuất kèm audio_en/ja/ko.m4a và bộ softsub (.srt / .vtt).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 48000
PYVT_PYTHON = Path("/opt/pyvideotrans/.venv/bin/python")
PYVT_CLONE_SCRIPT = Path("/workspace/pyvt_qwen_clone.py")


def parse_ts(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def trim_silence(wav: np.ndarray, top_db: float = 38.0) -> np.ndarray:
    if len(wav) < 400:
        return wav
    yt, _ = librosa.effects.trim(wav, top_db=top_db)
    return yt if len(yt) >= 400 else wav


def fit_to_slot(wav: np.ndarray, sr: int, max_dur: float, max_speed: float = 1.30) -> np.ndarray:
    dur = len(wav) / sr
    if dur <= max_dur or max_dur <= 0.15:
        return wav
    rate = min(max_speed, dur / max_dur)
    if rate <= 1.03:
        return wav
    stretched = librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)
    max_samp = int(max_dur * sr)
    if len(stretched) > max_samp:
        fade = min(480, max_samp // 4)
        stretched = stretched[:max_samp].copy()
        if fade > 0:
            stretched[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return stretched


def pick_vieneu_roles(tts) -> dict[str, tuple[str, dict]]:
    """Chuẩn quy trình: chỉ dùng đúng 2 giọng Nam–Nữ tự nhiên (Hải Đăng cho Nam, Trúc Ly cho Nữ)."""
    presets = tts.list_preset_voices()
    print(f"[5.1] VieNeu preset voices ({len(presets)}): {presets[:8]}", flush=True)
    males = [p for p in presets if not any(k in f"{p[0]} {p[1]}".lower() for k in ("nữ", "nu_", "female", "chị", "cô", "bé gái", "linh", "ngọc", "thảo", "mai", "hà", "trang", "trúc ly", "huyền", "dung", "duyên", "thanh"))]
    females = [p for p in presets if p not in males]

    if not males:
        males = presets[:1]
    if not females:
        females = presets[-1:]

    male_voice = next((m for m in males if "hải đăng" in m[0].lower()), males[0])
    female_voice = next((f for f in females if "trúc ly" in f[0].lower()), females[0])

    chosen = {
        "NAM_TRE": male_voice,
        "NAM_GIA": male_voice,
        "NU_TRE": female_voice,
        "NU_GIA": female_voice,
    }
    out = {}
    for role, (lbl, vid) in chosen.items():
        print(f"  -> Role {role}: {lbl} ({vid})", flush=True)
        out[role] = (lbl, tts.get_preset_voice(vid))
    return out


def synthesize_vieneu(lines: list[dict]) -> dict[int, np.ndarray]:
    from vieneu import Vieneu

    t0 = time.time()
    tts = Vieneu(mode="v3turbo", device="auto")
    role_voices = pick_vieneu_roles(tts)
    clips: dict[int, np.ndarray] = {}

    for role in ("NAM_TRE", "NAM_GIA", "NU_TRE", "NU_GIA"):
        group = [x for x in lines if x["voice_role"] == role and x["vi"].strip()]
        if not group:
            continue
        texts = [x["vi"].strip() for x in group]
        _, voice_dict = role_voices[role]
        print(f"[5.1] VieNeu batch {role}: {len(texts)} lines...", flush=True)
        wavs = tts.infer_batch(texts, voice=voice_dict, batch_size=32, apply_watermark=False)
        for item, wav in zip(group, wavs):
            w = np.asarray(wav, dtype=np.float32)
            if w.ndim > 1:
                w = np.mean(w, axis=1)
            clips[item["id"]] = trim_silence(w)

    tts.close()
    print(f"[OK] VieNeu synthesized {len(clips)} lines in {time.time() - t0:.1f}s", flush=True)
    return clips


def synthesize_pyvt_clones(vocals: Path, role_lines: Path, clone_root: Path, langs: str = "en,ja,ko"):
    script = PYVT_CLONE_SCRIPT if PYVT_CLONE_SCRIPT.exists() else (Path(__file__).resolve().parent / "pyvt_qwen_clone.py")
    py = str(PYVT_PYTHON) if PYVT_PYTHON.exists() else "python3"
    print(f"[5.2] Running pyVideoTrans voice clone ({py} {script})...", flush=True)
    subprocess.run([
        py, str(script), str(vocals), str(role_lines), str(clone_root), "--langs", langs
    ], check=True)


def load_cloned_lang_clips(lang_dir: Path, lines: list[dict]) -> dict[int, np.ndarray]:
    clips: dict[int, np.ndarray] = {}
    for x in lines:
        p = lang_dir / f"{x['id']:04d}.wav"
        if not p.is_file():
            continue
        w, sr = sf.read(str(p), dtype="float32")
        if w.ndim > 1:
            w = np.mean(w, axis=1)
        if sr != TARGET_SR:
            w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
        clips[x["id"]] = trim_silence(w.astype(np.float32))
    return clips


def build_timeline_and_mix(
    lines: list[dict],
    clips: dict[int, np.ndarray],
    orig_vocals: np.ndarray,
    instr_stereo: np.ndarray,
    total_samples: int,
    out_m4a: Path,
    tmp_dir: Path,
):
    hop = 480  # 10ms frame resolution
    n_frames = (total_samples + hop - 1) // hop
    pad_len = n_frames * hop - total_samples

    voice_track = np.zeros(total_samples, dtype=np.float32)
    # Keep non-dialogue vocals (crowd cheers, laughter, fight grunts) outside subtitle spans
    orig_keep_frames = np.ones(n_frames, dtype=np.float32) * 0.92

    for idx, item in enumerate(lines):
        st = parse_ts(item["start"])
        en = parse_ts(item["end"])
        mute_end_sec = en + 0.10

        w = clips.get(item["id"])
        if w is not None and len(w) > 0:
            next_st = parse_ts(lines[idx + 1]["start"]) if idx + 1 < len(lines) else total_samples / TARGET_SR
            allowed_end = max(en, min(next_st - 0.04, en + 0.25))
            max_dur = max(0.25, allowed_end - st)
            w_fit = fit_to_slot(w, TARGET_SR, max_dur)
            peak = float(np.max(np.abs(w_fit)))
            if peak > 0.92:
                w_fit = w_fit * (0.88 / peak)
            s0 = int(st * TARGET_SR)
            s1 = min(total_samples, s0 + len(w_fit))
            if s1 > s0:
                voice_track[s0:s1] += w_fit[: s1 - s0]
                mute_end_sec = max(mute_end_sec, s1 / TARGET_SR + 0.06)

        f0 = max(0, int((st - 0.08) * TARGET_SR // hop))
        f1 = min(n_frames, int(mute_end_sec * TARGET_SR // hop) + 1)
        orig_keep_frames[f0:f1] = 0.0

    # Smooth fade transitions (~80ms) on orig_keep so cheers/laughter fade cleanly without clicks
    fade_win = np.ones(9, dtype=np.float32) / 9.0
    orig_keep_smooth = np.convolve(orig_keep_frames, fade_win, mode="same")
    # Force exact 0 inside core subtitle dialogue centers so zero Chinese dialogue leaks
    for item in lines:
        f0 = max(0, int(parse_ts(item["start"]) * TARGET_SR // hop))
        f1 = min(n_frames, int(parse_ts(item["end"]) * TARGET_SR // hop) + 1)
        orig_keep_smooth[f0:f1] = 0.0

    orig_keep_samples = np.interp(
        np.arange(total_samples, dtype=np.float32),
        np.arange(n_frames, dtype=np.float32) * hop,
        orig_keep_smooth,
    ).astype(np.float32)

    v_pad = np.pad(voice_track, (0, pad_len))
    env = np.max(np.abs(v_pad.reshape(n_frames, hop)), axis=1)
    active = (env > 0.015).astype(np.float32)
    win = np.ones(25, dtype=np.float32) / 25.0
    smooth = np.clip(np.convolve(active, win, mode="same") * 1.8, 0.0, 1.0)
    gain_frames = 1.0 - 0.70 * smooth
    gain_samples = np.interp(
        np.arange(total_samples, dtype=np.float32),
        np.arange(n_frames, dtype=np.float32) * hop,
        gain_frames,
    ).astype(np.float32)

    ambient_vocals = orig_vocals[:total_samples] * orig_keep_samples
    mixed = instr_stereo[:total_samples].copy()
    mixed[:, 0] += ambient_vocals
    mixed[:, 1] += ambient_vocals
    mixed *= gain_samples[:, None]
    mixed[:, 0] += voice_track
    mixed[:, 1] += voice_track
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.95 / peak

    tmp_wav = tmp_dir / f"{out_m4a.stem}.wav"
    sf.write(str(tmp_wav), mixed, TARGET_SR, subtype="PCM_16")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(tmp_wav),
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(out_m4a),
    ], check=True)
    tmp_wav.unlink(missing_ok=True)


def pick_video_encoder() -> tuple[str, list[str]]:
    for enc, args in (
        ("hevc_nvenc", ["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "29", "-b:v", "700k", "-maxrate", "1200k", "-bufsize", "2400k", "-preset", "p4", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]),
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "28", "-b:v", "950k", "-maxrate", "1500k", "-bufsize", "3000k", "-preset", "p4", "-pix_fmt", "yuv420p"]),
    ):
        r = subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=640x640:d=0.2",
            "-c:v", enc, "-f", "null", "-",
        ], capture_output=True)
        if r.returncode == 0:
            return enc, args
    return "libx264", ["-c:v", "libx264", "-crf", "26", "-b:v", "800k", "-maxrate", "1300k", "-bufsize", "2600k", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def srt_to_vtt(srt_path: Path, vtt_path: Path):
    txt = srt_path.read_text(encoding="utf-8")
    vtt_body = txt.replace(",", ".")
    header = (
        "WEBVTT\n\n"
        "STYLE\n"
        "::cue {\n"
        "  color: #FFEB50;\n"
        "  background-color: rgba(0, 0, 0, 0.55);\n"
        "  text-shadow: 0 2px 3px #000000;\n"
        "}\n\n"
    )
    vtt_path.write_text(header + vtt_body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Step 5 GPU: VieNeu-TTS + pyVideoTrans Qwen3-TTS Voice Clone + Ducking + NVENC Blur Render.")
    ap.add_argument("video", type=Path, help="Normalized video (.mp4)")
    ap.add_argument("vocals", type=Path, help="vocals_48k_mono.flac (used as voice clone ref)")
    ap.add_argument("instrumental", type=Path, help="instrumental_48k.flac")
    ap.add_argument("role_lines", type=Path, help="role_lines.json")
    ap.add_argument("blur_config", type=Path, help="blur_config.json")
    ap.add_argument("out_dir", type=Path, help="Output directory")
    ap.add_argument("--name", default="P1_10m_vi.mp4", help="Output master MP4 filename")
    ap.add_argument("--langs", default="vi,en,ja,ko", help="Comma-separated tracks to synthesize (e.g. vi or vi,en,ja,ko)")
    args = ap.parse_args()

    t_all = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out_dir / "audio_tracks"
    sub_dir = args.out_dir / "subtitles"
    audio_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)

    lines = json.loads(args.role_lines.read_text(encoding="utf-8"))
    blur_cfg = json.loads(args.blur_config.read_text(encoding="utf-8"))
    vf_blur = blur_cfg["ffmpeg_filter"]

    print("[5.0] Loading vocals & instrumental 48kHz...", flush=True)
    orig_vocals, vsr = sf.read(str(args.vocals), dtype="float32")
    if orig_vocals.ndim > 1:
        orig_vocals = np.mean(orig_vocals, axis=1)
    if vsr != TARGET_SR:
        orig_vocals = librosa.resample(orig_vocals, orig_sr=vsr, target_sr=TARGET_SR)
    instr, sr = sf.read(str(args.instrumental), dtype="float32")
    if sr != TARGET_SR:
        raise RuntimeError(f"Expected 48kHz instrumental, got {sr}")
    if instr.ndim == 1:
        instr = np.column_stack([instr, instr])
    total_samples = min(len(instr), len(orig_vocals)) if len(orig_vocals) else len(instr)

    for lang in ("vi", "en", "ja", "ko"):
        blocks = [f"{x['id']}\n{x['start']} --> {x['end']}\n{x[lang]}\n" for x in lines]
        srt_p = sub_dir / f"{lang}.srt"
        srt_p.write_text("\n".join(blocks), encoding="utf-8")
        if lang == "vi":
            srt_to_vtt(srt_p, sub_dir / "vi.vtt")

    cache_dir = args.out_dir / "_tts_cache"
    cache_dir.mkdir(exist_ok=True)

    wanted_langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    clone_langs = [x for x in wanted_langs if x in ("en", "ja", "ko")]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        vi_m4a = audio_dir / "audio_vi.m4a"
        if "vi" in wanted_langs:
            # 1. VieNeu-TTS v3 Turbo (Vietnamese 2 voices: Hải Đăng + Trúc Ly)
            vi_clips = synthesize_vieneu(lines)
            build_timeline_and_mix(lines, vi_clips, orig_vocals, instr, total_samples, vi_m4a, tmp_dir)

        # 2. pyVideoTrans Qwen3-TTS voice cloning from vocals_48k_mono.flac (EN / JA / KO)
        if clone_langs:
            clone_root = cache_dir / "qwen_clones"
            synthesize_pyvt_clones(args.vocals, args.role_lines, clone_root, langs=",".join(clone_langs))
            for lang in clone_langs:
                lang_clips = load_cloned_lang_clips(clone_root / lang, lines)
                out_m4a = audio_dir / f"audio_{lang}.m4a"
                build_timeline_and_mix(lines, lang_clips, orig_vocals, instr, total_samples, out_m4a, tmp_dir)
                print(f"[OK] {lang.upper()} cloned track mixed -> {out_m4a.name}", flush=True)

        # 3. Render final blurred MP4 with Vietnamese dubbed audio
        enc_name, enc_flags = pick_video_encoder()
        out_mp4 = args.out_dir / args.name
        t_v = time.time()
        print(f"[5.3] Rendering video with {enc_name} + blur ({vf_blur})...", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "warning", "-stats",
            "-i", str(args.video),
            "-i", str(vi_m4a),
            "-vf", vf_blur,
            "-map", "0:v:0", "-map", "1:a:0",
            *enc_flags,
            "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            str(out_mp4),
        ], check=True)
        print(f"[OK] Video rendered with {enc_name} in {time.time() - t_v:.1f}s -> {out_mp4.name} ({out_mp4.stat().st_size/1e6:.1f} MB)", flush=True)

    print(f"[DONE] Step 5 completed in {time.time() - t_all:.1f}s!", flush=True)


if __name__ == "__main__":
    main()

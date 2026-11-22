"""
Step 2: Generate 4-language multi-voice time-synced AI dubbing tracks (vi, en, ja, ko) via edge-tts,
smartly keep non-dialogue action SFX/music while muting 100% of Chinese speech,
render clean 16:9 YouTube master with dynamic blurred sidebars and heavy Chinese subtitle blur,
and package a 4-language switchable preview MKV.

Usage:
  python scripts/step2_dub_and_render.py <project_out_dir>
"""
import array
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# Per-language voices: (voice_name, edge_rate, ffmpeg_pitch_factor)
VOICE_PROFILES = {
    "vi": {
        "label": "Vietnamese",
        "iso3": "vie",
        "M": ("vi-VN-NamMinhNeural", "+6%", 1.00),
        "F": ("vi-VN-HoaiMyNeural", "+6%", 1.00),
        "M2": ("vi-VN-NamMinhNeural", "+4%", 0.94),
    },
    "en": {
        "label": "English",
        "iso3": "eng",
        "M": ("en-US-AndrewMultilingualNeural", "+3%", 1.00),
        "F": ("en-US-AriaNeural", "+3%", 1.00),
        "M2": ("en-US-ChristopherNeural", "+3%", 0.97),
    },
    "ja": {
        "label": "Japanese",
        "iso3": "jpn",
        "M": ("ja-JP-KeitaNeural", "+4%", 1.00),
        "F": ("ja-JP-NanamiNeural", "+4%", 1.00),
        "M2": ("ja-JP-KeitaNeural", "+3%", 0.94),
    },
    "ko": {
        "label": "Korean",
        "iso3": "kor",
        "M": ("ko-KR-InJoonNeural", "+4%", 1.00),
        "F": ("ko-KR-SunHiNeural", "+4%", 1.00),
        "M2": ("ko-KR-InJoonNeural", "+3%", 0.94),
    },
}

SAMPLE_RATE = 48000


def parse_ts(ts: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0.0
    h, mn, s, ms = map(int, m.groups())
    return h * 3600 + mn * 60 + s + ms / 1000.0


def parse_srt(srt_path: Path):
    text = srt_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    entries = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        idx_num = int(re.sub(r"\D", "", lines[0]) or (len(entries) + 1))
        start_s, end_s = lines[1].split("-->")
        content = " ".join(lines[2:]).strip()
        if not content:
            continue
        entries.append({
            "num": idx_num,
            "start": parse_ts(start_s),
            "end": parse_ts(end_s),
            "text": content,
        })
    return entries


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


def load_speakers(out_dir: Path) -> dict:
    sp_file = out_dir / "_work" / "speakers.json"
    if not sp_file.exists():
        sp_file = out_dir / "speakers.json"
    role_map = {}
    if sp_file.exists():
        data = json.loads(sp_file.read_text(encoding="utf-8"))
        for role, nums in data.items():
            for n in nums:
                role_map[int(n)] = role
    return role_map


async def synth_one(text: str, voice: str, rate: str, out_mp3: Path, sem: asyncio.Semaphore):
    import edge_tts

    if out_mp3.exists() and out_mp3.stat().st_size > 500:
        return True
    no_ellip = text.replace("...", "").replace("…", "").strip()
    no_trail = no_ellip.rstrip(".!?。！？ ").strip()
    variants = [
        (text, rate),
        (no_trail or text, rate),
        (no_trail or text, "+0%"),
        (f"{no_trail or text},", "+0%"),
        (text, "+0%"),
        (no_trail or text, "+2%"),
    ]
    async with sem:
        for attempt, (txt, rt) in enumerate(variants):
            try:
                if out_mp3.exists():
                    out_mp3.unlink()
                comm = edge_tts.Communicate(txt, voice=voice, rate=rt)
                await comm.save(str(out_mp3))
                if out_mp3.exists() and out_mp3.stat().st_size > 500:
                    await asyncio.sleep(0.15)
                    return True
            except Exception as e:
                if out_mp3.exists():
                    out_mp3.unlink()
                if attempt == len(variants) - 1:
                    print(f"  [WARN] TTS failed after {len(variants)} attempts for '{text[:30]}': {e}", flush=True)
                    return False
                await asyncio.sleep(1.2 * (attempt + 1))
    return False


async def synth_all(entries, lang_cfg: dict, role_map: dict, cache_dir: Path):
    sem = asyncio.Semaphore(2)
    tasks = []
    for i, item in enumerate(entries):
        role = role_map.get(item["num"], "M")
        voice, rate, _pitch = lang_cfg[role]
        out_mp3 = cache_dir / f"line_{i:04d}.mp3"
        tasks.append(synth_one(item["text"], voice, rate, out_mp3, sem))
    return await asyncio.gather(*tasks)


def mp3_to_pcm_samples(mp3_path: Path, max_dur: float, pitch_factor: float = 1.0) -> array.array:
    if not mp3_path.exists() or mp3_path.stat().st_size <= 500:
        return array.array("h")
    trimmed_wav = mp3_path.with_suffix(".trim.wav")
    af_parts = [
        "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB",
        "areverse",
        "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB",
        "areverse",
    ]
    if abs(pitch_factor - 1.0) > 0.005:
        shifted_sr = int(round(SAMPLE_RATE * pitch_factor))
        af_parts.append(f"aresample={SAMPLE_RATE},asetrate={shifted_sr},aresample={SAMPLE_RATE},atempo={1.0/pitch_factor:.4f}")

    res = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(mp3_path),
            "-af", ",".join(af_parts),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(trimmed_wav),
        ],
    )
    if res.returncode != 0 or not trimmed_wav.exists() or trimmed_wav.stat().st_size <= 44:
        return array.array("h")

    with wave.open(str(trimmed_wav), "rb") as wf:
        actual_dur = wf.getnframes() / SAMPLE_RATE

    source_wav = trimmed_wav
    if max_dur > 0.35 and actual_dur > max_dur:
        tempo = min(actual_dur / max_dur, 1.45)
        fitted_wav = mp3_path.with_suffix(".fit.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(trimmed_wav),
                "-af", f"atempo={tempo:.4f}",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                str(fitted_wav),
            ],
            check=True,
        )
        source_wav = fitted_wav

    with wave.open(str(source_wav), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    return samples


def build_ducked_ambient_pcm(merged_mp4: Path, zh_entries: list, total_samples: int) -> array.array:
    """
    Extract stereo center-cancelled BGM + SFX track (L - R removes centered Chinese vocals
    while keeping stereo background music, punches, bells, and sirens), then smoothly duck
    it under dubbed speech (lower during dialogue, fuller during action scenes).
    """
    with tempfile.TemporaryDirectory(prefix="amb_") as tmp:
        amb_wav = Path(tmp) / "bgm_sfx.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(merged_mp4),
                "-vn",
                "-af", "pan=mono|c0=0.5*c0-0.5*c1,highpass=f=70,lowpass=f=11000,volume=2.2,alimiter=limit=0.85",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                str(amb_wav),
            ],
            check=True,
        )
        with wave.open(str(amb_wav), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        orig = array.array("h")
        orig.frombytes(raw)

    # Pad/trim to exact total_samples
    if len(orig) < total_samples:
        orig.extend([0] * (total_samples - len(orig)))
    elif len(orig) > total_samples:
        orig = orig[:total_samples]

    # Merge dialogue intervals so BGM ducks smoothly during speech and rises during action
    pad = 0.25
    speech_blocks = []
    for e in zh_entries:
        s = max(0.0, e["start"] - pad)
        t = e["end"] + pad
        if speech_blocks and s <= speech_blocks[-1][1]:
            speech_blocks[-1][1] = max(speech_blocks[-1][1], t)
        else:
            speech_blocks.append([s, t])

    # Build gain envelope: 0.70 during action gaps, 0.28 under dubbed speech
    action_gain = 0.70
    speech_gain = 0.26
    ramp_samples = int(0.20 * SAMPLE_RATE)

    ambient = array.array("h", [int(x * action_gain) for x in orig])

    for s, t in speech_blocks:
        i0 = max(0, int(round(s * SAMPLE_RATE)))
        i1 = min(total_samples, int(round(t * SAMPLE_RATE)))
        length = i1 - i0
        if length <= 0:
            continue
        r = min(ramp_samples, length // 2)
        for k in range(length):
            if r > 0 and k < r:
                frac = k / r
                g = action_gain + (speech_gain - action_gain) * frac
            elif r > 0 and k > length - r:
                frac = (length - k) / r
                g = action_gain + (speech_gain - action_gain) * frac
            else:
                g = speech_gain
            ambient[i0 + k] = int(orig[i0 + k] * g)

    return ambient


def build_language_audio(lang: str, srt_path: Path, role_map: dict, ambient_pcm: array.array, total_dur: float, out_m4a: Path):
    if out_m4a.exists() and out_m4a.stat().st_size > 500_000:
        print(f"[SKIP] {out_m4a.name} already exists!", flush=True)
        return
    cfg = VOICE_PROFILES[lang]
    entries = parse_srt(srt_path)
    print(f"\n[DUB {lang.upper()}] Synthesizing {len(entries)} lines ({cfg['label']}, multi-voice M/F/M2)...", flush=True)

    cache_dir = out_m4a.parent / "_work" / f"_tts_cache_{lang}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(synth_all(entries, cfg, role_map, cache_dir))

    total_samples = len(ambient_pcm)
    timeline = array.array("h", ambient_pcm)
    cursor_end = 0
    ok_lines = 0

    for i, item in enumerate(entries):
        mp3_path = cache_dir / f"line_{i:04d}.mp3"
        if not mp3_path.exists() or mp3_path.stat().st_size <= 500:
            continue

        start_s = item["start"]
        next_start = entries[i + 1]["start"] if i + 1 < len(entries) else total_dur
        slot_end = max(item["end"], min(next_start - 0.06, item["end"] + 1.0))
        slot_dur = max(0.4, slot_end - start_s)

        role = role_map.get(item["num"], "M")
        pitch_factor = cfg[role][2]
        samples = mp3_to_pcm_samples(mp3_path, slot_dur, pitch_factor)
        if not samples:
            continue

        start_idx = int(round(start_s * SAMPLE_RATE))
        if start_idx < cursor_end + int(0.03 * SAMPLE_RATE):
            start_idx = cursor_end + int(0.03 * SAMPLE_RATE)

        end_idx = min(total_samples, start_idx + len(samples))
        if start_idx >= total_samples:
            break

        copy_len = end_idx - start_idx
        for j in range(copy_len):
            val = timeline[start_idx + j] + samples[j]
            if val > 32767:
                val = 32767
            elif val < -32768:
                val = -32768
            timeline[start_idx + j] = val
        cursor_end = end_idx
        ok_lines += 1

    raw_wav = cache_dir / f"full_{lang}.wav"
    with wave.open(str(raw_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(timeline.tobytes())

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
    out_dur = probe_duration(out_m4a)
    print(f"[OK] Created {out_m4a.name} ({ok_lines}/{len(entries)} lines, duration={out_dur:.2f}s / video={total_dur:.2f}s)", flush=True)


def render_clean_16x9_video(merged_mp4: Path, vi_m4a: Path, out_mp4: Path, sub_blur_rect: tuple):
    bx, by, bw, bh = sub_blur_rect
    filter_complex = (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
        "gblur=sigma=32,eq=brightness=-0.16:saturation=0.85[bg];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        "split=2[fgbase][fgsub];"
        f"[fgsub]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius=38:luma_power=4:chroma_radius=22:chroma_power=3[subblur];"
        f"[fgbase][subblur]overlay={bx}:{by},"
        "eq=contrast=1.03:saturation=1.02,"
        "scale=608:1080:flags=lanczos[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
    )

    print(f"\n[RENDER] Encoding 1920x1080 16:9 YouTube master -> {out_mp4.name} ...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "warning", "-stats",
            "-i", str(merged_mp4),
            "-i", str(vi_m4a),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-r", "30",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-shortest",
            str(out_mp4),
        ],
        check=True,
    )
    print(f"[OK] Rendered YouTube master: {out_mp4}", flush=True)


def build_4lang_preview_mkv(video_mp4: Path, audio_map: dict, srt_map: dict, out_mkv: Path):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_mp4)]
    langs = ["vi", "en", "ja", "ko"]
    for lang in langs:
        cmd += ["-i", str(audio_map[lang])]
    for lang in langs:
        cmd += ["-i", str(srt_map[lang])]

    cmd += ["-map", "0:v:0"]
    for idx in range(1, 1 + len(langs)):
        cmd += ["-map", f"{idx}:a:0"]
    for idx in range(1 + len(langs), 1 + 2 * len(langs)):
        cmd += ["-map", f"{idx}:0"]

    cmd += ["-c:v", "copy", "-c:a", "copy", "-c:s", "srt"]
    for i, lang in enumerate(langs):
        iso = VOICE_PROFILES[lang]["iso3"]
        label = VOICE_PROFILES[lang]["label"]
        cmd += [
            f"-metadata:s:a:{i}", f"language={iso}",
            f"-metadata:s:a:{i}", f"title={label}",
            f"-metadata:s:s:{i}", f"language={iso}",
            f"-metadata:s:s:{i}", f"title={label}",
        ]
    cmd += [
        "-disposition:a", "0", "-disposition:a:0", "default",
        "-disposition:s", "0", "-disposition:s:0", "default",
        str(out_mkv),
    ]
    subprocess.run(cmd, check=True)
    print(f"[OK] Created 4-language switchable preview MKV: {out_mkv}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: step2_dub_and_render.py <project_out_dir>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    merged_mp4 = work_dir / "01_merged_raw.mp4"
    if not merged_mp4.exists():
        merged_mp4 = out_dir / "01_merged_raw.mp4"
    zh_srt = work_dir / "zh.srt"
    if not zh_srt.exists():
        zh_srt = out_dir / "zh.srt"
    if not merged_mp4.exists():
        raise FileNotFoundError(f"Missing {merged_mp4}")

    total_dur = probe_duration(merged_mp4)
    total_samples = int(round(total_dur * SAMPLE_RATE))
    print(f"[INFO] Master video duration: {total_dur:.3f}s", flush=True)

    role_map = load_speakers(out_dir)
    zh_entries = parse_srt(out_dir / "vi.srt")  # Use cleaned speech timestamps so fight gaps stay intact!
    print("[INFO] Preparing ducked ambient SFX track (100% muted during Chinese speech)...", flush=True)
    ambient_pcm = build_ducked_ambient_pcm(merged_mp4, zh_entries, total_samples)

    audio_map = {}
    srt_map = {}
    for lang in ["vi", "en", "ja", "ko"]:
        srt_path = out_dir / f"{lang}.srt"
        if not srt_path.exists():
            raise FileNotFoundError(f"Missing translated SRT: {srt_path}")
        srt_map[lang] = srt_path
        out_m4a = out_dir / f"audio_{lang}.m4a"
        build_language_audio(lang, srt_path, role_map, ambient_pcm, total_dur, out_m4a)
        audio_map[lang] = out_m4a

    # Heavy clean blur over original Chinese subtitle band (in 1080x1920 coords)
    sub_blur_rect = (40, 1410, 1000, 190)
    final_mp4 = out_dir / "final_youtube_16x9_vi.mp4"
    render_clean_16x9_video(merged_mp4, audio_map["vi"], final_mp4, sub_blur_rect)

    preview_mkv = out_dir / "preview_4_ngon_ngu.mkv"
    build_4lang_preview_mkv(final_mp4, audio_map, srt_map, preview_mkv)
    print("\n[ALL DONE] Ready for review!", flush=True)


if __name__ == "__main__":
    main()

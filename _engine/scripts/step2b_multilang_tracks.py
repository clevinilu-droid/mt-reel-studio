"""
Step 2b: Fast Multi-Language Dubbing (EN/JA/KO/ES...) + Exact Golden-Hook Audio/SRT Packaging.

For every requested language:
  1) Synthesizes 4-archetype dialogue turns concurrently via Edge-TTS Neural voices.
  2) Reuses mix_streaming_crowd_aware() to preserve 100% original stereo SFX/music/crowd cheers.
  3) Prepends the EXACT Golden Hook audio clip from render_meta.json (if any) and exports clean
     YouTube-ready files directly into:
       output/<movie>/audio/audio_<lang>.m4a   (exact length & 0.000s sync with <movie>.mp4)
       output/<movie>/subtitles/<lang>.srt     (exact hook-shifted timestamps)

Usage:
  python _engine/scripts/step2b_multilang_tracks.py <movie_out_dir> [--langs vi,en,ja,ko,es] [--concurrency 12] [--force]
"""
import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import edge_tts

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step1e_translate_multilang import resolve_work_dir
from step2_dub_and_render import parse_srt, probe_duration
from step2_gemini_tts_dub import group_entries_into_turns, load_cast_and_emotions
from step2_vieneu_stock_dub import (
    build_synced_srt,
    mix_streaming_crowd_aware,
    pick_auto_hook_range,
)

# 4-archetype Neural voices per language (all verified live in edge-tts 7.3.1; regional accent consistent per language)
MULTILANG_VOICE_TABLE = {
    "en": {
        "NAM_TRE": ("en-US-AndrewMultilingualNeural", "+4%", "+0Hz"),
        "NAM_GIA": ("en-US-ChristopherNeural", "+2%", "-8Hz"),
        "NU_TRE": ("en-US-AvaMultilingualNeural", "+4%", "+0Hz"),
        "NU_GIA": ("en-US-AriaNeural", "+2%", "-5Hz"),
    },
    "ja": {
        "NAM_TRE": ("ja-JP-KeitaNeural", "+5%", "+2Hz"),
        "NAM_GIA": ("ja-JP-KeitaNeural", "+1%", "-14Hz"),
        "NU_TRE": ("ja-JP-NanamiNeural", "+5%", "+3Hz"),
        "NU_GIA": ("ja-JP-NanamiNeural", "+1%", "-12Hz"),
    },
    "ko": {
        "NAM_TRE": ("ko-KR-InJoonNeural", "+5%", "+0Hz"),
        "NAM_GIA": ("ko-KR-HyunsuMultilingualNeural", "+2%", "-10Hz"),
        "NU_TRE": ("ko-KR-SunHiNeural", "+5%", "+2Hz"),
        "NU_GIA": ("ko-KR-SunHiNeural", "+1%", "-12Hz"),
    },
    "es": {
        "NAM_TRE": ("es-MX-JorgeNeural", "+4%", "+0Hz"),
        "NAM_GIA": ("es-US-AlonsoNeural", "+2%", "-8Hz"),
        "NU_TRE": ("es-MX-DaliaNeural", "+4%", "+0Hz"),
        "NU_GIA": ("es-US-PalomaNeural", "+2%", "-6Hz"),
    },
    "pt": {
        "NAM_TRE": ("pt-BR-AntonioNeural", "+4%", "+0Hz"),
        "NAM_GIA": ("pt-BR-AntonioNeural", "+1%", "-12Hz"),
        "NU_TRE": ("pt-BR-FranciscaNeural", "+4%", "+0Hz"),
        "NU_GIA": ("pt-BR-FranciscaNeural", "+1%", "-8Hz"),
    },
    "id": {
        "NAM_TRE": ("id-ID-ArdiNeural", "+4%", "+0Hz"),
        "NAM_GIA": ("id-ID-ArdiNeural", "+1%", "-12Hz"),
        "NU_TRE": ("id-ID-GadisNeural", "+4%", "+0Hz"),
        "NU_GIA": ("id-ID-GadisNeural", "+1%", "-8Hz"),
    },
}

ROLE_FALLBACK = {
    "M_LEAD": "NAM_TRE", "M_POLICE": "NAM_TRE", "M_BOSS": "NAM_GIA", "M_DOCTOR": "NAM_GIA",
    "F_SISTER": "NU_TRE", "F_LEAD": "NU_TRE", "F_BOSS": "NU_GIA", "F_NURSE": "NU_GIA",
    "NAM_CHINH": "NAM_TRE", "PHU_NAM": "NAM_TRE", "NAM_PHAN_DIEN": "NAM_GIA", "NAM_TRUNG_NIEN": "NAM_GIA",
    "NU_CHINH": "NU_TRE", "PHU_NU": "NU_TRE", "NU_PHAN_DIEN": "NU_GIA", "NU_TRUNG_NIEN": "NU_GIA",
}


async def _synth_turn(sem: asyncio.Semaphore, text: str, voice: str, rate: str, pitch: str, out_wav: Path):
    """Synthesize one turn with edge-tts MP3 -> 48kHz stereo WAV with retry."""
    tmp_mp3 = out_wav.with_suffix(".mp3")
    async with sem:
        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                await comm.save(str(tmp_mp3))
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_mp3), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)],
                    check=True,
                )
                tmp_mp3.unlink(missing_ok=True)
                return
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Edge-TTS failed for '{text[:40]}': {exc}") from exc
                await asyncio.sleep(1.2 * (attempt + 1))


async def synthesize_turns_parallel(lang: str, turns: list[dict], cache_dir: Path, concurrency: int, force: bool) -> list[Path]:
    """Synthesize all dialogue turns for one language concurrently."""
    voices = MULTILANG_VOICE_TABLE[lang]
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    turn_files: list[Path] = []

    for t in turns:
        role = ROLE_FALLBACK.get(t["role"], t["role"])
        if role not in voices:
            role = "NAM_TRE"
        voice, rate, pitch = voices[role]
        key_hash = hashlib.sha256(f"{lang}|{voice}|{rate}|{pitch}|{t['text']}".encode("utf-8")).hexdigest()[:8]
        out_wav = cache_dir / f"turn_{t['turn_idx']:04d}_{key_hash}.wav"
        turn_files.append(out_wav)
        if force or not out_wav.exists() or out_wav.stat().st_size <= 2000:
            tasks.append(_synth_turn(sem, t["text"], voice, rate, pitch, out_wav))

    if tasks:
        print(f"[{lang.upper()} TTS] Synthesizing {len(tasks)} turns (concurrency={concurrency}, cached={len(turns)-len(tasks)})...", flush=True)
        await asyncio.gather(*tasks)
    else:
        print(f"[{lang.upper()} TTS] All {len(turns)} turns already cached.", flush=True)
    return turn_files


def prepend_hook_to_audio(raw_m4a: Path, hook: tuple[float, float] | None, out_m4a: Path):
    """Apply identical Golden Hook audio prepend & crossfade as render_hq_single_pass."""
    out_m4a.parent.mkdir(parents=True, exist_ok=True)
    if hook is None:
        shutil.copy2(raw_m4a, out_m4a)
        return

    h_start, h_end = hook
    h_dur = h_end - h_start
    fade_d = min(0.20, h_dur / 4.0)
    fade_st = max(0.0, h_dur - fade_d)
    filter_complex = (
        f"[0:a]afade=t=out:st={fade_st:.3f}:d={fade_d:.3f},asetpts=PTS-STARTPTS[ha];"
        f"[1:a]afade=t=in:st=0:d={fade_d:.3f},asetpts=PTS-STARTPTS[ma];"
        f"[ha][ma]concat=n=2:v=0:a=1[aout]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{h_start:.3f}", "-t", f"{h_dur:.3f}", "-i", str(raw_m4a),
            "-i", str(raw_m4a),
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(out_m4a),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synced multi-language audio tracks & subtitles into output/<movie>/.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--langs", type=str, default="vi,en,ja,ko,es", help="Comma-separated languages to package (default: vi,en,ja,ko,es)")
    parser.add_argument("--concurrency", type=int, default=12, help="Concurrent Edge-TTS calls (default: 12)")
    parser.add_argument("--max-gap", type=float, default=0.55)
    parser.add_argument("--max-chars", type=int, default=150)
    parser.add_argument("--max-span", type=float, default=6.5)
    parser.add_argument("--force", action="store_true", help="Re-synthesize and re-export all non-vi audio tracks")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    work_dir = resolve_work_dir(out_dir)
    tracks_dir = work_dir / "tracks"

    merged_mp4 = work_dir / "01_merged_raw.mp4"
    inst_wav = work_dir / "02_instrumental_mdx.wav"
    vocals_wav = work_dir / "02_vocals_mdx.wav"
    if not inst_wav.exists():
        raise FileNotFoundError(f"Missing required file: {inst_wav}")

    meta_path = work_dir / "render_meta.json"
    rm = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if merged_mp4.exists():
        total_dur = probe_duration(merged_mp4)
    elif rm.get("raw_duration_s"):
        total_dur = float(rm["raw_duration_s"])
    else:
        raise FileNotFoundError(f"Missing both {merged_mp4} and raw_duration_s in {meta_path}")

    hook = None
    if rm.get("hook_range"):
        hook = (float(rm["hook_range"][0]), float(rm["hook_range"][1]))
    elif not meta_path.exists() and (tracks_dir / "vi.srt").exists():
        hook = pick_auto_hook_range(parse_srt(tracks_dir / "vi.srt"), total_dur)

    _, line_meta = load_cast_and_emotions(out_dir, work_dir)
    speech_srt = work_dir / "zh_clean.srt"
    speech_entries = parse_srt(speech_srt) if speech_srt.exists() else None

    out_audio_dir = out_dir / "audio"
    out_sub_dir = out_dir / "subtitles"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    t0 = time.time()
    for lang in langs:
        srt_path = tracks_dir / f"{lang}.srt"
        if not srt_path.exists():
            raise FileNotFoundError(f"Missing translated SRT for '{lang}': {srt_path} (run step1e_translate_multilang.py first)")
        entries = parse_srt(srt_path)
        sp_entries = speech_entries if speech_entries is not None else entries

        raw_m4a = tracks_dir / f"audio_{lang}.m4a"
        if lang != "vi":
            if lang not in MULTILANG_VOICE_TABLE:
                raise ValueError(f"No voice table for '{lang}'")
            if args.force or not raw_m4a.exists() or raw_m4a.stat().st_size < 100_000:
                t_l = time.time()
                turns = group_entries_into_turns(entries, line_meta, lang, args.max_gap, args.max_chars, args.max_span)
                cache_dir = work_dir / f"_edge_cache_{lang}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                turn_files = asyncio.run(synthesize_turns_parallel(lang, turns, cache_dir, args.concurrency, args.force))
                mix_streaming_crowd_aware(turns, turn_files, sp_entries, inst_wav, vocals_wav, cache_dir, total_dur, raw_m4a)
                print(f"[OK] Raw {lang} mix ready in {time.time()-t_l:.1f}s -> {raw_m4a.name}", flush=True)
            else:
                print(f"[SKIP] Raw mix exists: {raw_m4a.name}", flush=True)
        else:
            if not raw_m4a.exists():
                raise FileNotFoundError(f"Missing VI audio {raw_m4a} (run step2_vieneu_stock_dub.py first)")

        # Export hook-synchronized SRT and M4A into output/<movie>/subtitles/ and output/<movie>/audio/
        synced_tmp = build_synced_srt(entries, tracks_dir, hook, lang=lang)
        final_sub = out_sub_dir / f"{lang}.srt"
        shutil.copy2(synced_tmp, final_sub)

        final_aud = out_audio_dir / f"audio_{lang}.m4a"
        if args.force or not final_aud.exists() or final_aud.stat().st_size < 100_000:
            prepend_hook_to_audio(raw_m4a, hook, final_aud)
        print(f"  [PACKAGED] {lang.upper()} -> {final_aud.name} ({probe_duration(final_aud):.2f}s) + {final_sub.name}", flush=True)

    print(f"[ALL DONE] Multi-language audio + subtitle package ready in {time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
Step 2 (Expressive Studio Upgrade): Multi-character emotion-directed dubbing with
Google Gemini 2.5 Flash TTS, mixed in full 48kHz stereo over MDX-Net isolated BGM+SFX.

Features:
  - 7 distinct prebuilt character voices (configurable per movie via _work/cast_and_emotions.json)
  - Smart turn grouping: merges adjacent lines from the same speaker into continuous dramatic turns
    (dramatically improves acting flow AND reduces API calls ~3x for Free Tier quota!)
  - Natural-language acting direction per turn (crying, shouting, stoic, teasing, urgent...)
  - Content-hashed per-turn WAV caching & exponential backoff on 429 RESOURCE_EXHAUSTED
  - Full 48kHz stereo mixing with smooth ducking over _work/02_instrumental_mdx.wav (no L-R distortion)
  - Lossless MP4 remux + 4-language switchable MKV build

Usage:
  python _engine/scripts/step2_gemini_tts_dub.py <movie_out_dir> [--langs vi,en,ja,ko] [--dry-run-turns]
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step2_dub_and_render import parse_srt, probe_duration, render_clean_16x9_video

SAMPLE_RATE = 48000

CHARACTER_VOICES = {
    "M_LEAD": "Orus",      # Calm, low, stoic ex-special forces lead
    "F_BOSS": "Kore",      # Confident, sharp, sultry fight club hostess
    "M_BOSS": "Algenib",   # Gravelly, intimidating boss / thugs
    "F_NURSE": "Aoede",    # Concerned, empathetic female nurse
    "F_POLICE": "Erinome", # Sharp, articulate female police dispatcher / officer
    "M_POLICE": "Fenrir",  # Commanding, loud, tense police captain/officers
    "M_DOCTOR": "Charon",  # Terrified / solemn male doctor
    "F_SISTER": "Leda",    # Heartbroken, tearful, trembling younger sister
}

LEGACY_ROLE_MAP = {
    "M": "M_LEAD",
    "F": "F_BOSS",
    "M2": "M_BOSS",
}

DEFAULT_EMOTIONS = {
    "M_LEAD": "calm, cold, stoic and fearless",
    "F_BOSS": "confident, teasing and sharp",
    "M_BOSS": "gruff, gravelly, mocking and aggressive",
    "F_NURSE": "urgent, anxious and compassionate",
    "F_POLICE": "sharp, tense and commanding",
    "M_POLICE": "forceful, urgent and commanding",
    "M_DOCTOR": "trembling, panicked and apologetic",
    "F_SISTER": "heartbroken, sobbing desperately with a trembling voice",
}

LANG_CONFIG = {
    "vi": {"name": "Vietnamese", "iso3": "vie", "label": "Vietnamese"},
    "en": {"name": "English", "iso3": "eng", "label": "English"},
    "ja": {"name": "Japanese", "iso3": "jpn", "label": "Japanese"},
    "ko": {"name": "Korean", "iso3": "kor", "label": "Korean"},
}


def load_cast_and_emotions(out_dir: Path, work_dir: Path | None = None) -> tuple[dict, dict]:
    """Return (voice_overrides, line_meta_by_num)."""
    if work_dir is None:
        work_dir = SCRIPTS_DIR.parent / "work" / out_dir.name
    cast_file = work_dir / "cast_and_emotions.json"
    if not cast_file.exists():
        cast_file = out_dir / "_work" / "cast_and_emotions.json"
    if cast_file.exists():
        raw = json.loads(cast_file.read_text(encoding="utf-8"))
        voices = dict(CHARACTER_VOICES)
        voices.update(raw.get("voices", {}))
        lines = {}
        for k, v in raw.get("lines", {}).items():
            role = v.get("role", "M_LEAD")
            role = LEGACY_ROLE_MAP.get(role, role)
            lines[int(k)] = {
                "role": role,
                "emotion": v.get("emotion") or DEFAULT_EMOTIONS.get(role, "natural dramatic delivery"),
                "break": bool(v.get("break", False)),
            }
        return voices, lines

    sp_file = work_dir / "speakers.json"
    if not sp_file.exists():
        sp_file = out_dir / "_work" / "speakers.json"
    if not sp_file.exists():
        sp_file = out_dir / "speakers.json"
    lines = {}
    if sp_file.exists():
        raw = json.loads(sp_file.read_text(encoding="utf-8"))
        for role_key, nums in raw.items():
            mapped = LEGACY_ROLE_MAP.get(role_key, role_key)
            for n in nums:
                lines[int(n)] = {
                    "role": mapped,
                    "emotion": DEFAULT_EMOTIONS.get(mapped, "natural dramatic delivery"),
                    "break": False,
                }
    return dict(CHARACTER_VOICES), lines


def join_turn_texts(texts: list[str], lang: str) -> str:
    if len(texts) == 1:
        return texts[0].strip()
    if lang == "ja":
        buf = []
        for idx, t in enumerate(texts):
            s = t.strip()
            if idx > 0 and buf:
                prev = buf[-1]
                if prev.endswith(("。", "！", "？", "、", "!", "?", "…")):
                    buf.append(s)
                else:
                    buf.append("、" + s)
            else:
                buf.append(s)
        return "".join(buf)
    return " ".join(t.strip() for t in texts if t.strip())


def group_entries_into_turns(
    entries: list[dict],
    line_meta: dict,
    lang: str,
    max_gap: float = 0.75,
    max_chars: int = 280,
    max_span: float = 16.0,
) -> list[dict]:
    """Merge consecutive lines by the same character with short pauses into coherent acted turns."""
    turns = []
    cur = None

    for item in entries:
        num = item["num"]
        meta = line_meta.get(num, {"role": "M_LEAD", "emotion": DEFAULT_EMOTIONS["M_LEAD"], "break": False})
        role = meta["role"]
        emotion = meta["emotion"]
        force_break = meta.get("break", False)

        can_merge = False
        if cur is not None and not force_break and cur["role"] == role:
            gap = item["start"] - cur["end"]
            new_span = item["end"] - cur["start"]
            proj_len = sum(len(x) for x in cur["texts"]) + len(cur["texts"]) + len(item["text"])
            if gap <= max_gap and new_span <= max_span and proj_len <= max_chars:
                can_merge = True

        if can_merge:
            cur["end"] = max(cur["end"], item["end"])
            cur["texts"].append(item["text"])
            cur["nums"].append(num)
            if emotion not in cur["emotions"]:
                cur["emotions"].append(emotion)
        else:
            if cur is not None:
                cur["text"] = join_turn_texts(cur["texts"], lang)
                turns.append(cur)
            cur = {
                "role": role,
                "start": item["start"],
                "end": item["end"],
                "texts": [item["text"]],
                "nums": [num],
                "emotions": [emotion],
            }

    if cur is not None:
        cur["text"] = join_turn_texts(cur["texts"], lang)
        turns.append(cur)

    for idx, t in enumerate(turns):
        t["turn_idx"] = idx
        if len(t["emotions"]) == 1:
            t["direction"] = t["emotions"][0]
        else:
            t["direction"] = f"{t['emotions'][0]}, shifting to {t['emotions'][-1]}"
    return turns


def build_acting_prompt(lang: str, direction: str, text: str) -> str:
    lang_name = LANG_CONFIG[lang]["name"]
    return f"Speak in natural {lang_name}, {direction}:\n{text}"


def extract_retry_delay(err_str: str) -> float | None:
    m = re.search(r"retry\s*(?:in|delay['\":\s]+)\s*([0-9]+(?:\.[0-9]+)?)s", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return None


def decode_audio_bytes_to_wav(raw: bytes | str, mime_type: str, out_wav: Path):
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    if not raw or len(raw) < 100:
        raise RuntimeError("Empty audio bytes returned by Gemini TTS")

    if raw[:4] == b"RIFF":
        tmp_src = out_wav.with_suffix(".src.wav")
        tmp_src.write_bytes(raw)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav)],
            check=True,
        )
        tmp_src.unlink(missing_ok=True)
        return

    m_rate = re.search(r"rate=(\d+)", mime_type or "")
    src_rate = int(m_rate.group(1)) if m_rate else 24000
    tmp_raw = out_wav.with_suffix(".raw.wav")
    with wave.open(str(tmp_raw), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(src_rate)
        wf.writeframes(raw)

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_raw), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav)],
        check=True,
    )
    tmp_raw.unlink(missing_ok=True)


def synth_turn_gemini(
    client,
    types_mod,
    active_models: list[str],
    voice_name: str,
    prompt: str,
    cache_wav: Path,
    pace_delay: float,
    force: bool = False,
) -> tuple[bool, str]:
    if not force and cache_wav.exists() and cache_wav.stat().st_size > 2000:
        return False, "cache"

    cfg = types_mod.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types_mod.SpeechConfig(
            voice_config=types_mod.VoiceConfig(
                prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    last_err = None
    max_rounds = 4
    for round_idx in range(max_rounds):
        if not active_models:
            raise RuntimeError(f"All Gemini TTS models exhausted or unavailable. Last error: {last_err}")
        round_wait_hints = []

        for m in list(active_models):
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=cfg,
                    )
                    audio_part = None
                    finish_reason = None
                    for cand in (resp.candidates or []):
                        finish_reason = getattr(cand, "finish_reason", None)
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_part = part.inline_data
                                    break
                        if audio_part:
                            break
                    if not audio_part:
                        last_err = f"No audio payload on {m} (finish_reason={finish_reason})"
                        print(f"  [FALLBACK] {last_err}; trying next model...", flush=True)
                        break

                    decode_audio_bytes_to_wav(audio_part.data, audio_part.mime_type or "", cache_wav)
                    if pace_delay > 0:
                        time.sleep(pace_delay)
                    return True, m
                except Exception as e:
                    err_s = str(e)
                    last_err = err_s
                    if "PerDay" in err_s or "limit: 0" in err_s:
                        if m in active_models:
                            active_models.remove(m)
                        print(f"  [INFO] Model {m} daily quota exhausted; switching to remaining models: {active_models}", flush=True)
                        break
                    if "400 INVALID_ARGUMENT" in err_s or "404 NOT_FOUND" in err_s or "403 PERMISSION_DENIED" in err_s:
                        if m in active_models:
                            active_models.remove(m)
                        print(f"  [WARN] Model {m} rejected request ({err_s[:120]}); remaining: {active_models}", flush=True)
                        break
                    is_quota = "429" in err_s or "RESOURCE_EXHAUSTED" in err_s or "quota" in err_s.lower()
                    if is_quota:
                        hint = extract_retry_delay(err_s) or 20.0
                        if len(active_models) > 1 or attempt == 1:
                            round_wait_hints.append(hint)
                            print(f"  [429] Model {m} hit rate limit (hint {hint:.0f}s); trying next model...", flush=True)
                            break
                        print(f"  [429] Only {m} remaining -> sleeping {hint:.1f}s before retry...", flush=True)
                        time.sleep(hint)
                    else:
                        if attempt == 0:
                            print(f"  [RETRY] {m} transient error -> sleeping 3.0s ({err_s[:110]})", flush=True)
                            time.sleep(3.0)
                        else:
                            print(f"  [FALLBACK] {m} failed twice ({err_s[:110]}); trying next model...", flush=True)

        if not active_models:
            raise RuntimeError(f"All Gemini TTS models exhausted or unavailable. Last error: {last_err}")
        if round_idx < max_rounds - 1:
            sleep_s = min(round_wait_hints) if round_wait_hints else 20.0
            print(f"  [WAIT] Round {round_idx + 1}/{max_rounds} finished without audio -> sleeping {sleep_s:.1f}s before next round...", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Gemini TTS failed after {max_rounds} rounds across {active_models}: {last_err}")


def prepare_turn_pcm(wav_path: Path, slot_dur: float) -> np.ndarray:
    """Trim leading/trailing silence and gently fit within slot_dur (max 1.30x to preserve emotion)."""
    trimmed = wav_path.with_suffix(".trim.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav_path),
            "-af", "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB,areverse",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
            str(trimmed),
        ],
        check=True,
    )
    with wave.open(str(trimmed), "rb") as wf:
        actual_dur = wf.getnframes() / SAMPLE_RATE

    src_wav = trimmed
    if slot_dur > 0.45 and actual_dur > slot_dur * 1.03:
        tempo = min(actual_dur / slot_dur, 1.30)
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


def load_stereo_bgm(bgm_wav: Path, target_frames: int) -> np.ndarray:
    with wave.open(str(bgm_wav), "rb") as wf:
        sr = wf.getframerate()

    if sr != SAMPLE_RATE:
        with tempfile.TemporaryDirectory(prefix="bgm48_") as tmp:
            tmp_48k = Path(tmp) / "bgm_48k.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(bgm_wav), "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp_48k)],
                check=True,
            )
            with wave.open(str(tmp_48k), "rb") as wf:
                n_ch = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
    else:
        with wave.open(str(bgm_wav), "rb") as wf:
            n_ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch == 1:
        stereo = np.stack([pcm, pcm], axis=1)
    else:
        stereo = pcm.reshape(-1, n_ch)[:, :2]

    if len(stereo) < target_frames:
        pad = np.zeros((target_frames - len(stereo), 2), dtype=np.float32)
        stereo = np.concatenate([stereo, pad], axis=0)
    elif len(stereo) > target_frames:
        stereo = stereo[:target_frames]
    return stereo


def mix_turns_with_stereo_bgm(
    turns: list[dict],
    cache_dir: Path,
    turn_files: list[Path],
    bgm_stereo: np.ndarray,
    total_dur: float,
    out_m4a: Path,
):
    total_frames = len(bgm_stereo)
    action_gain = 0.92
    speech_gain = 0.46
    ramp_frames = int(0.18 * SAMPLE_RATE)

    gain = np.full(total_frames, action_gain, dtype=np.float32)
    speech_mono = np.zeros(total_frames, dtype=np.float32)

    cursor_end = 0
    placed = 0

    for idx, turn in enumerate(turns):
        wav_path = turn_files[idx]
        if not wav_path.exists() or wav_path.stat().st_size <= 2000:
            continue

        start_s = turn["start"]
        next_start = turns[idx + 1]["start"] if idx + 1 < len(turns) else total_dur
        slot_end = max(turn["end"], min(next_start - 0.08, turn["end"] + 1.15))
        slot_dur = max(0.5, slot_end - start_s)

        samples = prepare_turn_pcm(wav_path, slot_dur)
        if len(samples) == 0:
            continue

        start_idx = int(round(start_s * SAMPLE_RATE))
        min_idx = cursor_end + int(0.05 * SAMPLE_RATE) if cursor_end > 0 else 0
        if start_idx < min_idx:
            start_idx = min_idx
        if start_idx >= total_frames:
            break

        end_idx = min(total_frames, start_idx + len(samples))
        copy_len = end_idx - start_idx
        speech_mono[start_idx:end_idx] += samples[:copy_len]
        cursor_end = end_idx
        placed += 1

        # Smoothly duck BGM under this spoken turn
        d0 = max(0, start_idx - ramp_frames)
        d1 = min(total_frames, end_idx + ramp_frames)
        r_in = start_idx - d0
        r_out = d1 - end_idx
        if r_in > 0:
            ramp_down = np.linspace(action_gain, speech_gain, r_in, dtype=np.float32)
            gain[d0:start_idx] = np.minimum(gain[d0:start_idx], ramp_down)
        gain[start_idx:end_idx] = np.minimum(gain[start_idx:end_idx], speech_gain)
        if r_out > 0:
            ramp_up = np.linspace(speech_gain, action_gain, r_out, dtype=np.float32)
            gain[end_idx:d1] = np.minimum(gain[end_idx:d1], ramp_up)

    mixed = bgm_stereo * gain[:, None] + speech_mono[:, None]
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.96:
        mixed = mixed * (0.96 / peak)

    pcm_out = np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16)
    raw_stereo_wav = cache_dir / f"mix_{out_m4a.stem}.wav"
    with wave.open(str(raw_stereo_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_out.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_stereo_wav),
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={total_dur:.3f}",
            "-t", f"{total_dur:.3f}",
            "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-c:a", "aac", "-b:a", "192k",
            str(out_m4a),
        ],
        check=True,
    )
    raw_stereo_wav.unlink(missing_ok=True)
    out_dur = probe_duration(out_m4a)
    print(f"[OK] Mixed {out_m4a.name} ({placed}/{len(turns)} turns, {out_dur:.2f}s stereo)", flush=True)


def resolve_srt_path(tracks_dir: Path, work_dir: Path, out_dir: Path, lang: str) -> Path:
    for d in (tracks_dir, work_dir, out_dir):
        p = d / f"{lang}.srt"
        if p.exists():
            return p
    return tracks_dir / f"{lang}.srt"


def remux_available_tracks(out_dir: Path, work_dir: Path, tracks_dir: Path, merged_mp4: Path):
    base_video = work_dir / "03_clean_16x9_video.mp4"
    present_langs = [l for l in ["vi", "en", "ja", "ko"] if (tracks_dir / f"audio_{l}.m4a").exists()]
    if not present_langs:
        print("[WARN] No mixed audio tracks found in tracks/; skipping final MP4 mux.", flush=True)
        return

    if not base_video.exists():
        seed_audio = tracks_dir / f"audio_{present_langs[0]}.m4a"
        render_clean_16x9_video(merged_mp4, seed_audio, base_video, (40, 1410, 1000, 190))

    final_mp4 = out_dir / f"{out_dir.name}.mp4"
    tmp_mp4 = work_dir / f"{out_dir.name}.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(base_video)]
    for lang in present_langs:
        cmd += ["-i", str(tracks_dir / f"audio_{lang}.m4a")]

    cmd += ["-map", "0:v:0"]
    for idx in range(1, 1 + len(present_langs)):
        cmd += ["-map", f"{idx}:a:0"]

    cmd += ["-c:v", "copy", "-c:a", "copy"]
    for i, lang in enumerate(present_langs):
        iso = LANG_CONFIG[lang]["iso3"]
        label = LANG_CONFIG[lang]["label"]
        cmd += [
            f"-metadata:s:a:{i}", f"language={iso}",
            f"-metadata:s:a:{i}", f"handler_name={label}",
            f"-metadata:s:a:{i}", f"title={label}",
        ]
    cmd += [
        "-disposition:a", "0", "-disposition:a:0", "default",
        "-movflags", "+faststart", "-shortest",
        str(tmp_mp4),
    ]
    subprocess.run(cmd, check=True)
    tmp_mp4.replace(final_mp4)
    print(f"[OK] Built single output product: {final_mp4} (audio tracks: {', '.join(present_langs)})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Emotion-directed Gemini 2.5 Flash TTS + MDX-Net stereo dubbing.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output folder")
    parser.add_argument("--langs", default="vi,en,ja,ko", help="Comma-separated languages (default: vi,en,ja,ko)")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-tts", help="Primary Gemini TTS model ID")
    parser.add_argument(
        "--fallback-models",
        default="gemini-3.1-flash-tts-preview",
        help="Comma-separated fallback TTS models if primary returns FinishReason.OTHER or quota limit",
    )
    parser.add_argument("--max-gap", type=float, default=0.75, help="Max gap (seconds) to merge same-character lines")
    parser.add_argument("--max-chars", type=int, default=280, help="Max characters per grouped turn")
    parser.add_argument("--delay", type=float, default=2.5, help="Seconds to wait between uncached Gemini API calls")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test cap: synthesize at most N turns and stop without mixing (0 = all)")
    parser.add_argument("--force", action="store_true", help="Ignore cached turn WAVs and re-synthesize")
    parser.add_argument("--dry-run-turns", action="store_true", help="Inspect turn grouping & prompts without calling API")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = SCRIPTS_DIR.parent / "work" / out_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = work_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    merged_mp4 = work_dir / "01_merged_raw.mp4"
    if not merged_mp4.exists():
        merged_mp4 = out_dir / "01_merged_raw.mp4"
    inst_wav = work_dir / "02_instrumental_mdx.wav"
    if not merged_mp4.exists():
        raise FileNotFoundError(f"Missing merged video: {merged_mp4}")
    if not inst_wav.exists() and not args.dry_run_turns:
        raise FileNotFoundError(f"Missing MDX instrumental WAV: {inst_wav} (run step1b_separate_mdx.py first)")

    langs = [x.strip() for x in args.langs.split(",") if x.strip() in LANG_CONFIG]
    if not langs:
        raise ValueError("No valid languages requested in --langs")

    voices, line_meta = load_cast_and_emotions(out_dir, work_dir)

    if args.dry_run_turns:
        for lang in langs:
            srt_path = resolve_srt_path(tracks_dir, work_dir, out_dir, lang)
            entries = parse_srt(srt_path)
            turns = group_entries_into_turns(entries, line_meta, lang, args.max_gap, args.max_chars)
            print(f"\n=== DRY RUN [{lang.upper()}]: {len(entries)} lines -> {len(turns)} acted turns ===")
            for t in turns[:10]:
                print(f"  T{t['turn_idx']:02d} [{t['start']:.2f}-{t['end']:.2f}] {t['role']} ({voices[t['role']]}) | {t['direction']}\n       -> {t['text']}")
            print(f"  ... ({len(turns)} total turns)")
        return

    project_root = SCRIPTS_DIR.parent.parent
    load_dotenv(project_root / ".env")
    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "PASTE_YOUR_KEY_HERE":
        print(
            "[NEED API KEY] Chưa tìm thấy GEMINI_API_KEY hợp lệ.\n"
            f"Hãy tạo key miễn phí tại https://aistudio.google.com/apikey và dán vào file: {project_root / '.env'}\n"
            "Ví dụ nội dung file .env:\nGEMINI_API_KEY=AIzaSy...",
            file=sys.stderr,
        )
        sys.exit(2)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    total_dur = probe_duration(merged_mp4)
    total_frames = int(round(total_dur * SAMPLE_RATE))
    print(f"[INFO] Loading MDX stereo instrumental ({inst_wav.name}) for {total_dur:.2f}s master...", flush=True)
    bgm_stereo = load_stereo_bgm(inst_wav, total_frames)

    active_models = [args.model] + [
        m.strip() for m in args.fallback_models.split(",") if m.strip() and m.strip() != args.model
    ]
    for lang in langs:
        srt_path = resolve_srt_path(tracks_dir, work_dir, out_dir, lang)
        if not srt_path.exists():
            raise FileNotFoundError(f"Missing SRT: {srt_path}")
        entries = parse_srt(srt_path)
        turns = group_entries_into_turns(entries, line_meta, lang, args.max_gap, args.max_chars)
        cache_dir = work_dir / f"_gemini_tts_cache_{lang}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[GEMINI TTS {lang.upper()}] {len(entries)} lines grouped into {len(turns)} dramatic turns (models={active_models})", flush=True)
        turn_files = []
        api_calls = 0

        target_turns = turns[: args.limit] if args.limit and args.limit > 0 else turns
        for t in target_turns:
            voice_name = voices.get(t["role"], "Orus")
            prompt = build_acting_prompt(lang, t["direction"], t["text"])
            key_hash = hashlib.sha256(f"{args.model}|{voice_name}|{prompt}".encode("utf-8")).hexdigest()[:8]
            wav_file = cache_dir / f"turn_{t['turn_idx']:04d}_{key_hash}.wav"
            made_call, used_model = synth_turn_gemini(client, types, active_models, voice_name, prompt, wav_file, args.delay, args.force)
            if made_call:
                api_calls += 1
                print(f"  [TTS {lang.upper()}] {t['turn_idx']+1}/{len(turns)} ({t['role']}={voice_name}, model={used_model}, lines {t['nums'][0]}..{t['nums'][-1]}) OK -> {wav_file.name}", flush=True)
            turn_files.append(wav_file)

        if args.limit and args.limit > 0:
            print(f"[SMOKE TEST OK] Synthesized/verified {len(target_turns)} turn(s) for {lang.upper()}; stopping without overwriting {tracks_dir.name}/audio_{lang}.m4a.", flush=True)
            return

        print(f"[INFO] {lang.upper()} synthesis complete ({api_calls} new API calls, {len(turns) - api_calls} cached).", flush=True)
        out_m4a = tracks_dir / f"audio_{lang}.m4a"
        mix_turns_with_stereo_bgm(turns, cache_dir, turn_files, bgm_stereo, total_dur, out_m4a)

    remux_available_tracks(out_dir, work_dir, tracks_dir, merged_mp4)
    print("\n[ALL DONE] Studio MDX-Net + Gemini TTS pipeline finished!", flush=True)


if __name__ == "__main__":
    main()

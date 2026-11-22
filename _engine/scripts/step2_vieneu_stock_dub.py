"""
Step 2 (Offline Stock Cast Dubbing): Dub Vietnamese dialogue using the fixed
8-role studio preset cast from _engine/config/vieneu_stock_cast.json via
VieNeu-TTS (v3turbo, 48kHz ONNX CPU), mix over 02_instrumental_mdx.wav with
Strict Frame Sync, optionally prepend a Golden 5s Hook climax clip, and export
strictly 1 clean MP4 file:
  output/<movie>/<movie>.mp4

Usage:
  python _engine/scripts/step2_vieneu_stock_dub.py <movie_out_dir> [--hook-range START:END] [--force]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
DEFAULT_CAST_CONFIG = ENGINE_DIR / "config" / "vieneu_stock_cast.json"
DEFAULT_WATERMARK = ENGINE_DIR.parent / "branding" / "MT_Reel_Strip_Watermark.png"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step1_merge_and_transcribe import format_ts
from step2_dub_and_render import parse_srt, probe_duration
from step2_gemini_tts_dub import (
    LANG_CONFIG,
    group_entries_into_turns,
    load_cast_and_emotions,
)
from step2_vieneu_clone_dub import SAMPLE_RATE, prepare_strict_pcm

EXPLICIT_ROLE_DEFAULTS = {
    "NAM_CHINH": "NAM_TRE",
    "PHU_NAM": "NAM_TRE",
    "NAM_PHAN_DIEN": "NAM_GIA",
    "NAM_TRUNG_NIEN": "NAM_GIA",
    "NU_CHINH": "NU_TRE",
    "PHU_NU": "NU_TRE",
    "NU_PHAN_DIEN": "NU_GIA",
    "NU_TRUNG_NIEN": "NU_GIA",
    "M_LEAD": "NAM_TRE",
    "M_POLICE": "NAM_TRE",
    "M_BOSS": "NAM_GIA",
    "M_DOCTOR": "NAM_GIA",
    "F_SISTER": "NU_TRE",
    "F_LEAD": "NU_TRE",
    "F_POLICE": "NU_TRE",
    "F_BOSS": "NU_GIA",
    "F_NURSE": "NU_GIA",
}


def auto_archetype_for_role(role: str, roles_cfg: dict | None = None) -> str:
    """Map an arbitrary movie role key to one of the active stock cast archetypes."""
    upper = role.upper()
    if upper in EXPLICIT_ROLE_DEFAULTS:
        mapped = EXPLICIT_ROLE_DEFAULTS[upper]
        if not roles_cfg or mapped in roles_cfg:
            return mapped
    is_female = upper.startswith("F_") or upper.startswith("NU_") or any(
        k in upper for k in ("FEMALE", "WOMAN", "GIRL", "MOM", "MOTHER", "SISTER", "WIFE", "AUNT", "GRANDMA", "QUEEN", "MISTRESS", "NURSE")
    )
    is_older = any(
        k in upper for k in ("GIA", "OLD", "ELDER", "DAD", "MOM", "FATHER", "MOTHER", "GRAND", "DOCTOR", "BUTLER", "BOSS", "UNCLE", "AUNT", "TRUNG_NIEN")
    )
    if is_female:
        return "NU_GIA" if is_older else "NU_TRE"
    return "NAM_GIA" if is_older else "NAM_TRE"


def load_stock_cast_resolver(cast_config_path: Path, work_dir: Path, turns: list[dict]) -> dict[str, str]:
    """Load the 8-role VieNeu stock cast and persist resolved movie-role -> preset voice mapping."""
    cfg = json.loads(cast_config_path.read_text(encoding="utf-8"))
    roles_cfg = cfg["roles"]

    map_file = work_dir / "stock_cast_map.json"
    saved_map = json.loads(map_file.read_text(encoding="utf-8")) if map_file.exists() else {}

    resolved_archetypes: dict[str, str] = {}
    role_to_voice: dict[str, str] = {}
    unique_roles = sorted({t["role"] for t in turns})

    for role in unique_roles:
        entry = saved_map.get(role)
        if role in roles_cfg:
            arch = role
        elif isinstance(entry, dict) and entry.get("archetype") in roles_cfg:
            arch = entry["archetype"]
        else:
            arch = auto_archetype_for_role(role, roles_cfg)
        voice_name = roles_cfg[arch]["voice"]
        resolved_archetypes[role] = {"archetype": arch, "voice": voice_name}
        role_to_voice[role] = voice_name

    if not map_file.exists() or saved_map != resolved_archetypes:
        map_file.write_text(json.dumps(resolved_archetypes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CAST] Saved 8-role stock mapping -> {map_file.name}", flush=True)

    for role, info in resolved_archetypes.items():
        print(f"  - {role:10s} -> {info['archetype']:15s} ({info['voice']})", flush=True)
    return role_to_voice


def parse_hook_range(spec: str | None, total_dur: float) -> tuple[float, float] | None:
    """Parse 'START:END' seconds for Golden 5s Hook."""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid --hook-range '{spec}', expected START:END (seconds)")
    start_s, end_s = float(parts[0]), float(parts[1])
    if start_s < 0 or end_s <= start_s or end_s > total_dur:
        raise ValueError(f"Invalid hook range [{start_s}, {end_s}] for video duration {total_dur:.2f}s")
    return start_s, end_s


def ensure_48k_stereo_wav(src_wav: Path, tmp_wav: Path) -> tuple[Path, bool]:
    """Ensure WAV is 48kHz 2ch PCM without loading it into RAM; returns (path, is_temp)."""
    with wave.open(str(src_wav), "rb") as wf:
        if wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 2 and wf.getsampwidth() == 2:
            return src_wav, False
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src_wav), "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp_wav)],
        check=True,
    )
    return tmp_wav, True


def read_wav_slice(wf: wave.Wave_read | None, start_f: int, count_f: int) -> np.ndarray:
    """Read exact [start_f, start_f + count_f) stereo float32 chunk zero-padded past EOF."""
    out = np.zeros((count_f, 2), dtype=np.float32)
    if wf is None:
        return out
    avail = max(0, min(count_f, wf.getnframes() - start_f))
    if avail > 0:
        wf.setpos(start_f)
        arr = np.frombuffer(wf.readframes(avail), dtype=np.int16).astype(np.float32) / 32768.0
        out[:avail] = arr.reshape(-1, 2)
    return out


def mix_streaming_crowd_aware(
    turns: list[dict],
    turn_files: list[Path],
    speech_entries: list[dict],
    inst_wav: Path,
    vocals_wav: Path,
    cache_dir: Path,
    total_dur: float,
    out_m4a: Path,
    crowd_gain: float = 0.88,
    pad_sec: float = 0.22,
):
    """Stream-mix 48kHz stereo in 30s chunks (<150MB RAM even for 3-hour movies)."""
    total_frames = int(round(total_dur * SAMPLE_RATE))

    # 1. Plan turn timestamps & keep lightweight int16 mono buffers
    planned: list[tuple[int, int, np.ndarray]] = []
    cursor_end = 0
    max_drift = 0.0
    for idx, turn in enumerate(turns):
        wav_path = turn_files[idx]
        if not wav_path.exists() or wav_path.stat().st_size <= 2000:
            continue
        start_s = turn["start"]
        next_start = turns[idx + 1]["start"] if idx + 1 < len(turns) else total_dur
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
        planned.append((start_idx, end_idx, np.clip(samples[: end_idx - start_idx] * 32767.0, -32768, 32767).astype(np.int16)))
        cursor_end = end_idx

    # 2. Merge padded dialogue intervals for crowd-unmute envelope
    mute_blocks: list[list[float]] = []
    for e in speech_entries:
        s = max(0.0, e["start"] - pad_sec)
        t = e["end"] + pad_sec
        if mute_blocks and s <= mute_blocks[-1][1]:
            mute_blocks[-1][1] = max(mute_blocks[-1][1], t)
        else:
            mute_blocks.append([s, t])
    muted_sec = sum(t - s for s, t in mute_blocks)
    print(f"[AMB] Crowd-aware streaming mix (kept {max(0.0, total_dur - muted_sec):.1f}s ambient/crowd, muted {muted_sec:.1f}s dialogue)", flush=True)

    inst_tmp = cache_dir / "_tmp_inst_48k.wav"
    voc_tmp = cache_dir / "_tmp_voc_48k.wav"
    inst_48, inst_is_tmp = ensure_48k_stereo_wav(inst_wav, inst_tmp)
    voc_48, voc_is_tmp = ensure_48k_stereo_wav(vocals_wav, voc_tmp) if vocals_wav.exists() else (None, False)
    raw_wav = cache_dir / f"mix_{out_m4a.stem}.wav"

    action_gain = 0.92
    speech_gain = 0.45
    duck_ramp = int(0.16 * SAMPLE_RATE)
    voc_ramp = int(0.04 * SAMPLE_RATE)
    chunk_frames = 30 * SAMPLE_RATE

    try:
        wf_inst = wave.open(str(inst_48), "rb")
        wf_voc = wave.open(str(voc_48), "rb") if voc_48 else None
        try:
            with wave.open(str(raw_wav), "wb") as wf_out:
                wf_out.setnchannels(2)
                wf_out.setsampwidth(2)
                wf_out.setframerate(SAMPLE_RATE)

                for c0 in range(0, total_frames, chunk_frames):
                    c1 = min(total_frames, c0 + chunk_frames)
                    clen = c1 - c0
                    bed = read_wav_slice(wf_inst, c0, clen)

                    if wf_voc is not None:
                        voc = read_wav_slice(wf_voc, c0, clen)
                        v_gain = np.full(clen, crowd_gain, dtype=np.float32)
                        for ms, mt in mute_blocks:
                            i0 = int(round(ms * SAMPLE_RATE))
                            i1 = int(round(mt * SAMPLE_RATE))
                            if i1 <= c0 or i0 >= c1:
                                continue
                            r = min(voc_ramp, max(1, (i1 - i0) // 2))
                            # Apply fade-down [i0, i0+r), mute [i0+r, i1-r), fade-up [i1-r, i1) clipped to [c0, c1)
                            ls0, le0 = max(c0, i0), min(c1, i0 + r)
                            if le0 > ls0:
                                vals = np.linspace(crowd_gain, 0.0, r, dtype=np.float32)[ls0 - i0 : le0 - i0]
                                v_gain[ls0 - c0 : le0 - c0] = np.minimum(v_gain[ls0 - c0 : le0 - c0], vals)
                            zm0, zm1 = max(c0, i0 + r), min(c1, i1 - r)
                            if zm1 > zm0:
                                v_gain[zm0 - c0 : zm1 - c0] = 0.0
                            ls1, le1 = max(c0, i1 - r), min(c1, i1)
                            if le1 > ls1:
                                vals = np.linspace(0.0, crowd_gain, r, dtype=np.float32)[ls1 - (i1 - r) : le1 - (i1 - r)]
                                v_gain[ls1 - c0 : le1 - c0] = np.minimum(v_gain[ls1 - c0 : le1 - c0], vals)
                        bed = bed + voc * v_gain[:, None]
                        pk = float(np.max(np.abs(bed)))
                        if pk > 0.96:
                            bed = bed * (0.96 / pk)

                    b_gain = np.full(clen, action_gain, dtype=np.float32)
                    speech_mono = np.zeros(clen, dtype=np.float32)
                    for ts, te, s16 in planned:
                        if te + duck_ramp <= c0 or ts - duck_ramp >= c1:
                            continue
                        os0, oe0 = max(c0, ts), min(c1, te)
                        if oe0 > os0:
                            speech_mono[os0 - c0 : oe0 - c0] += s16[os0 - ts : oe0 - ts].astype(np.float32) / 32768.0
                            b_gain[os0 - c0 : oe0 - c0] = np.minimum(b_gain[os0 - c0 : oe0 - c0], speech_gain)
                        d0, d1 = ts - duck_ramp, ts
                        ls0, le0 = max(c0, d0), min(c1, d1)
                        if le0 > ls0:
                            vals = np.linspace(action_gain, speech_gain, duck_ramp, dtype=np.float32)[ls0 - d0 : le0 - d0]
                            b_gain[ls0 - c0 : le0 - c0] = np.minimum(b_gain[ls0 - c0 : le0 - c0], vals)
                        u0, u1 = te, te + duck_ramp
                        ls1, le1 = max(c0, u0), min(c1, u1)
                        if le1 > ls1:
                            vals = np.linspace(speech_gain, action_gain, duck_ramp, dtype=np.float32)[ls1 - u0 : le1 - u0]
                            b_gain[ls1 - c0 : le1 - c0] = np.minimum(b_gain[ls1 - c0 : le1 - c0], vals)

                    mixed = bed * b_gain[:, None] + speech_mono[:, None]
                    pk = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
                    if pk > 0.96:
                        mixed = mixed * (0.96 / pk)
                    wf_out.writeframes(np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16).tobytes())
        finally:
            wf_inst.close()
            if wf_voc is not None:
                wf_voc.close()
    finally:
        if inst_is_tmp:
            inst_tmp.unlink(missing_ok=True)
        if voc_is_tmp:
            voc_tmp.unlink(missing_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_wav),
            "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={total_dur:.3f}",
            "-t", f"{total_dur:.3f}",
            "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-c:a", "aac", "-b:a", "256k",
            str(out_m4a),
        ],
        check=True,
    )
    raw_wav.unlink(missing_ok=True)
    out_dur = probe_duration(out_m4a)
    print(f"[OK] Strict-mixed {out_m4a.name} ({len(planned)}/{len(turns)} turns, {out_dur:.2f}s stereo, max_drift=+{max_drift:.2f}s)", flush=True)


def build_synced_srt(entries: list[dict], tracks_dir: Path, hook: tuple[float, float] | None, lang: str = "vi") -> Path:
    """Return SRT synced to final video timeline (prepends hook lines + shifts main movie)."""
    base_srt = tracks_dir / f"{lang}.srt"
    if hook is None:
        return base_srt

    h_start, h_end = hook
    h_dur = h_end - h_start
    synced: list[dict] = []
    for e in entries:
        if e["end"] > h_start and e["start"] < h_end:
            synced.append({
                "start": max(0.0, e["start"] - h_start),
                "end": min(h_dur, e["end"] - h_start),
                "text": e["text"],
            })
    for e in entries:
        synced.append({
            "start": e["start"] + h_dur,
            "end": e["end"] + h_dur,
            "text": e["text"],
        })

    out_srt = tracks_dir / f"{lang}_youtube.srt"
    with out_srt.open("w", encoding="utf-8") as fh:
        for i, e in enumerate(synced, 1):
            fh.write(f"{i}\n{format_ts(e['start'])} --> {format_ts(e['end'])}\n{e['text'].strip()}\n\n")
    print(f"[SUB] Wrote hook-synced soft subtitle ({len(synced)} lines) -> tracks/{out_srt.name}", flush=True)
    return out_srt


def pick_auto_hook_range(entries: list[dict], total_dur: float, window_sec: float = 7.0) -> tuple[float, float] | None:
    """Select the highest-drama 7s climax window (with !/? rapid dialogue) after the first 30s."""
    if total_dur < 60.0 or len(entries) < 8:
        return None
    best_score = -1.0
    best_window = None
    for e in entries:
        w_start = max(30.0, e["start"] - 0.5)
        w_end = w_start + window_sec
        if w_end > total_dur - 5.0:
            continue
        in_win = [x for x in entries if x["start"] >= w_start and x["end"] <= w_end + 0.3]
        if len(in_win) < 2:
            continue
        exclam = sum(x["text"].count("!") + x["text"].count("?") + x["text"].count("！") + x["text"].count("？") for x in in_win)
        score = len(in_win) * 2.0 + exclam * 3.0
        if score > best_score:
            best_score = score
            best_window = (round(w_start, 2), round(w_end, 2))
    return best_window


def render_hq_single_pass(
    merged_mp4: Path,
    vi_m4a: Path,
    synced_srt: Path,
    out_mp4: Path,
    work_dir: Path,
    hook: tuple[float, float] | None,
    sub_blur_rect: tuple[int, int, int, int] = (40, 1410, 1000, 190),
    crf: int = 17,
    preset: str = "medium",
    watermark_png: Path | None = None,
    encoder: str = "libx264",
):
    """Single-pass high-quality 1920x1080 render with optional strip watermark & soft subtitle."""
    tmp_final = work_dir / f"{out_mp4.stem}.hq_final.tmp.mp4"
    iso = LANG_CONFIG["vi"]["iso3"]
    label = LANG_CONFIG["vi"]["label"]
    bx, by, bw, bh = sub_blur_rect
    has_wm = watermark_png is not None and watermark_png.exists()

    if encoder == "h264_nvenc":
        v_enc_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(max(crf, 18)), "-profile:v", "high", "-r", "30"]
    elif encoder == "h264_qsv":
        v_enc_args = ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", str(max(crf, 18)), "-look_ahead", "0", "-profile:v", "high", "-r", "30"]
    else:
        v_enc_args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-r", "30"]

    def build_video_fx(wm_input_idx: int | None) -> str:
        after_sub = f"[fgraw][subblur]overlay={bx}:{by}[fgblur];[fgblur][{wm_input_idx}:v]overlay={bx}:{by}," if wm_input_idx is not None else f"[fgraw][subblur]overlay={bx}:{by},"
        return (
            "split=2[fgbase][bgsrc];"
            "[bgsrc]scale=480:270:force_original_aspect_ratio=increase,crop=480:270,"
            "boxblur=luma_radius=12:luma_power=2,scale=1920:1080:flags=bilinear,"
            "eq=brightness=-0.16:saturation=0.85[bg];"
            "[fgbase]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
            "split=2[fgraw][fgsub];"
            f"[fgsub]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius=38:luma_power=4:chroma_radius=22:chroma_power=3[subblur];"
            f"{after_sub}"
            "eq=contrast=1.03:saturation=1.02,"
            "scale=608:1080:flags=lanczos[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
        )

    norm_v = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"

    if hook is None:
        wm_idx = 3 if has_wm else None
        cmd = [
            "ffmpeg", "-y", "-v", "warning", "-stats",
            "-i", str(merged_mp4), "-i", str(vi_m4a), "-i", str(synced_srt),
        ]
        if has_wm:
            cmd += ["-i", str(watermark_png)]
        cmd += [
            "-filter_complex", f"[0:v]{norm_v},{build_video_fx(wm_idx)}",
            "-map", "[vout]", "-map", "1:a:0", "-map", "2:0",
            *v_enc_args,
            "-video_track_timescale", "15360",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:a:0", f"language={iso}",
            "-metadata:s:a:0", f"handler_name={label}",
            "-metadata:s:a:0", f"title={label}",
            "-disposition:a:0", "default",
            "-metadata:s:s:0", f"language={iso}",
            "-metadata:s:s:0", f"handler_name={label}",
            "-metadata:s:s:0", f"title={label}",
            "-disposition:s:0", "default",
            "-movflags", "+faststart", "-shortest",
            str(tmp_final),
        ]
        subprocess.run(cmd, check=True)
    else:
        h_start, h_end = hook
        h_dur = h_end - h_start
        fade_d = min(0.20, h_dur / 4.0)
        fade_st = max(0.0, h_dur - fade_d)
        wm_idx = 2 if has_wm else None
        hook_mp4 = work_dir / f"{out_mp4.stem}.hook.tmp.mp4"
        main_mp4 = work_dir / f"{out_mp4.stem}.main.tmp.mp4"
        concat_txt = work_dir / f"{out_mp4.stem}.concat.txt"

        print(f"[HQ RENDER] Two-stage safe {encoder} CRF/CQ={crf} + Hook [{h_start:.1f}s->{h_end:.1f}s] + soft sub" + (f" + watermark ({watermark_png.name})" if has_wm else "") + "...", flush=True)

        # 1. Render Golden Hook clip cleanly in isolation (prevents filtergraph reinit DTS corruption on mixed-res MP4s)
        hook_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-ss", f"{h_start:.3f}", "-t", f"{h_dur:.3f}", "-i", str(merged_mp4),
            "-ss", f"{h_start:.3f}", "-t", f"{h_dur:.3f}", "-i", str(vi_m4a),
        ]
        if has_wm:
            hook_cmd += ["-i", str(watermark_png)]
        hook_cmd += [
            "-filter_complex", f"[0:v]{norm_v},fade=t=out:st={fade_st:.3f}:d={fade_d:.3f},setpts=PTS-STARTPTS,{build_video_fx(wm_idx)}",
            "-map", "[vout]", "-map", "1:a:0",
            *v_enc_args, "-video_track_timescale", "15360",
            "-af", f"afade=t=out:st={fade_st:.3f}:d={fade_d:.3f}",
            "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-shortest", str(hook_mp4),
        ]
        subprocess.run(hook_cmd, check=True)

        # 2. Render Main movie with video-only filter_complex (audio stays independent of video resolution changes)
        main_cmd = [
            "ffmpeg", "-y", "-v", "warning", "-stats",
            "-i", str(merged_mp4), "-i", str(vi_m4a),
        ]
        if has_wm:
            main_cmd += ["-i", str(watermark_png)]
        main_cmd += [
            "-filter_complex", f"[0:v]{norm_v},fade=t=in:st=0:d={fade_d:.3f},{build_video_fx(wm_idx)}",
            "-map", "[vout]", "-map", "1:a:0",
            *v_enc_args, "-video_track_timescale", "15360",
            "-af", f"afade=t=in:st=0:d={fade_d:.3f}",
            "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-shortest", str(main_mp4),
        ]
        subprocess.run(main_cmd, check=True)

        # 3. Lossless concat + mux soft subtitle
        concat_txt.write_text(f"file '{hook_mp4.resolve().as_posix()}'\nfile '{main_mp4.resolve().as_posix()}'\n", encoding="utf-8")
        mux_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-i", str(synced_srt),
            "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0",
            "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
            "-metadata:s:a:0", f"language={iso}",
            "-metadata:s:a:0", f"handler_name={label}",
            "-metadata:s:a:0", f"title={label}",
            "-disposition:a:0", "default",
            "-metadata:s:s:0", f"language={iso}",
            "-metadata:s:s:0", f"handler_name={label}",
            "-metadata:s:s:0", f"title={label}",
            "-disposition:s:0", "default",
            "-movflags", "+faststart",
            str(tmp_final),
        ]
        subprocess.run(mux_cmd, check=True)
        for p_tmp in (hook_mp4, main_mp4, concat_txt):
            p_tmp.unlink(missing_ok=True)

    tmp_final.replace(out_mp4)


def main():
    parser = argparse.ArgumentParser(description="Offline VieNeu-TTS 8-role stock cast dubbing + crowd-preserving HQ render.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory (e.g., output/001_BaoBiao_Ep01_05)")
    parser.add_argument("--cast-config", type=Path, default=DEFAULT_CAST_CONFIG, help="Path to 8-role stock cast JSON")
    parser.add_argument("--hook-range", type=str, default=None, help="Optional Golden Hook START:END in seconds (e.g., 355.0:362.0)")
    parser.add_argument("--hook-auto", action="store_true", help="Automatically detect highest-drama 7s Golden Hook if --hook-range omitted")
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc", "h264_qsv"), default="libx264", help="Video encoder (h264_nvenc on NVIDIA, h264_qsv on Intel Iris Xe)")
    parser.add_argument("--crf", type=int, default=17, help="x264 CRF / NVENC CQ quality (default: 17)")
    parser.add_argument("--preset", type=str, default="medium", help="x264 preset (default: medium)")
    parser.add_argument("--watermark", type=Path, default=DEFAULT_WATERMARK, help="Optional strip watermark PNG")
    parser.add_argument("--no-watermark", action="store_true", help="Disable watermark overlay")
    parser.add_argument("--max-gap", type=float, default=0.55, help="Max gap (seconds) to group same-speaker lines")
    parser.add_argument("--max-chars", type=int, default=150, help="Max characters per turn")
    parser.add_argument("--max-span", type=float, default=6.5, help="Max turn duration (seconds)")
    parser.add_argument("--force", action="store_true", help="Re-synthesize all turns ignoring cache")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = ENGINE_DIR / "work" / out_dir.name
    tracks_dir = work_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    merged_mp4 = work_dir / "01_merged_raw.mp4"
    inst_wav = work_dir / "02_instrumental_mdx.wav"
    vocals_wav = work_dir / "02_vocals_mdx.wav"
    vi_srt = tracks_dir / "vi.srt"
    for req in (merged_mp4, inst_wav, vi_srt, args.cast_config):
        if not req.exists():
            raise FileNotFoundError(f"Missing required file: {req}")

    total_dur = probe_duration(merged_mp4)
    hook = parse_hook_range(args.hook_range, total_dur)

    _, line_meta = load_cast_and_emotions(out_dir, work_dir)
    entries = parse_srt(vi_srt)
    if hook is None and args.hook_auto:
        hook = pick_auto_hook_range(entries, total_dur)
        if hook:
            print(f"[HOOK AUTO] Selected drama climax [{hook[0]:.2f}s -> {hook[1]:.2f}s]", flush=True)
    turns = group_entries_into_turns(entries, line_meta, "vi", args.max_gap, args.max_chars, args.max_span)
    print(f"[VIENEU STOCK] {len(entries)} lines -> {len(turns)} tight scene-synced turns", flush=True)

    role_to_voice = load_stock_cast_resolver(args.cast_config, work_dir, turns)

    cache_dir = work_dir / "_vieneu_stock_cache_vi"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from vieneu import Vieneu
    tts = None
    turn_files = []
    synth_count = 0

    try:
        pending_by_voice: dict[str, list[tuple[dict, Path]]] = {}
        for t in turns:
            role = t["role"]
            preset_voice = role_to_voice[role]
            key_hash = hashlib.sha256(f"v3turbo_preset|{preset_voice}|{t['text']}".encode("utf-8")).hexdigest()[:8]
            out_wav = cache_dir / f"turn_{t['turn_idx']:04d}_{key_hash}.wav"
            turn_files.append(out_wav)

            if not args.force and out_wav.exists() and out_wav.stat().st_size > 2000:
                continue
            pending_by_voice.setdefault(preset_voice, []).append((t, out_wav))

        if pending_by_voice:
            t0 = time.time()
            print("[INFO] Loading VieNeu-TTS (v3turbo, auto GPU/CPU)...", flush=True)
            try:
                tts = Vieneu(mode="v3turbo")
            except Exception as gpu_err:
                print(f"[WARN] VieNeu auto GPU init failed ({gpu_err}); falling back to ONNX 8-thread CPU...", flush=True)
                tts = Vieneu(mode="v3turbo", backend="onnx", threads=max(4, min(8, os.cpu_count() or 4)))
            backend = getattr(tts, "backend", "onnx")
            print(f"[OK] Loaded in {time.time() - t0:.1f}s (backend={backend})", flush=True)

            if backend == "pytorch" and hasattr(tts, "infer_batch"):
                bs = 32
                for preset_voice, items in pending_by_voice.items():
                    for start in range(0, len(items), bs):
                        chunk = items[start : start + bs]
                        ts = time.time()
                        try:
                            wavs = tts.infer_batch([t["text"] for t, _ in chunk], voice=preset_voice, batch_size=bs)
                            for (_, out_wav), wav in zip(chunk, wavs):
                                if wav is None or len(wav) == 0:
                                    wav = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
                                tts.save(wav, str(out_wav))
                                synth_count += 1
                            print(f"  [STOCK BATCH] {preset_voice}: {start + len(chunk)}/{len(items)} turns in {time.time()-ts:.1f}s", flush=True)
                        except Exception as err:
                            print(f"  [WARN] Batch failed ({err}); falling back to per-turn infer...", flush=True)
                            for t, out_wav in chunk:
                                wav = tts.infer(t["text"], voice=preset_voice)
                                tts.save(wav, str(out_wav))
                                synth_count += 1
            else:
                for preset_voice, items in pending_by_voice.items():
                    for t, out_wav in items:
                        ts = time.time()
                        wav = tts.infer(t["text"], voice=preset_voice)
                        tts.save(wav, str(out_wav))
                        synth_count += 1
                        print(f"  [STOCK {t['turn_idx']+1}/{len(turns)}] {t['role']} ({preset_voice}) in {time.time()-ts:.1f}s -> {out_wav.name}", flush=True)
    finally:
        if tts is not None:
            tts.close()

    print(f"[INFO] Synthesis done ({synth_count} new, {len(turns)-synth_count} cached).", flush=True)

    speech_srt = work_dir / "zh_clean.srt"
    speech_entries = parse_srt(speech_srt) if speech_srt.exists() else entries
    vi_m4a = tracks_dir / "audio_vi.m4a"
    mix_streaming_crowd_aware(turns, turn_files, speech_entries, inst_wav, vocals_wav, cache_dir, total_dur, vi_m4a)

    synced_srt = build_synced_srt(entries, tracks_dir, hook)
    final_mp4 = out_dir / f"{out_dir.name}.mp4"
    wm_path = None if args.no_watermark else args.watermark
    render_hq_single_pass(
        merged_mp4, vi_m4a, synced_srt, final_mp4, work_dir, hook,
        crf=args.crf, preset=args.preset, watermark_png=wm_path, encoder=args.encoder,
    )
    out_srt = out_dir / f"{out_dir.name}.srt"
    shutil.copy2(synced_srt, out_srt)
    final_dur = probe_duration(final_mp4)
    meta = {
        "movie": out_dir.name,
        "raw_duration_s": round(total_dur, 3),
        "final_duration_s": round(final_dur, 3),
        "hook_range": list(hook) if hook else None,
        "encoder": args.encoder,
        "watermark": wm_path.name if (wm_path and wm_path.exists()) else None,
        "lines": len(entries),
        "turns": len(turns),
    }
    (work_dir / "render_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ALL DONE] Output ready: {final_mp4} ({final_mp4.stat().st_size/1024/1024:.2f} MB, {final_dur:.2f}s) + {out_srt.name}", flush=True)


if __name__ == "__main__":
    main()

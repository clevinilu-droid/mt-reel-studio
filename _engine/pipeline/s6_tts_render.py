"""Bước 6 (s6_tts_render): Lồng tiếng VieNeu (VI) + VoxCPM2 / Qwen3-TTS-1.7B (EN/JA/KO/ES) + Mix & Render NVENC.

Đặc điểm chuẩn studio:
  - Tiếng Việt (`vi`): dùng `VieNeu TTS` v3turbo (`single_female` -> Trúc Ly, hoặc map theo vai).
  - Ngoại ngữ (`en, ja, ko, es`): ưu tiên `OpenBMB/VoxCPM2` (48kHz, Voice Design không clone giọng gốc + khóa timbre đồng nhất theo ngôn ngữ),
    tự động fallback sang `Qwen3-TTS-1.7B-VoiceDesign` nếu cần.
  - Cache từng câu `tts_cache/<lang>/turn_XXXX.wav` (resume-safe 100%).
  - Trộn lên `04_clean_bed.wav` (giữ nguyên 100% SFX/tiếng reo hò gốc ngoài thoại) -> xuất `audio/audio_<lang>.m4a`.
  - Render HQ bằng `h264_nvenc` (hoặc `h264_qsv` / `libx264`), làm mờ sub Trung 94% chiều ngang + watermark 10% + soft sub;
    ở chế độ Test xuất luôn MP4 riêng cho từng ngôn ngữ bằng `-c:v copy`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from _engine.core.checkpoint import ENGINE_DIR, OUTPUT_ROOT, ROOT_DIR, atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, load_pipeline_config, release_vram
from _engine.pipeline.s1_prepare import detect_gpu_encoder

SR = 48000
LANG_ISO3 = {"vi": "vie", "en": "eng", "ja": "jpn", "ko": "kor", "es": "spa", "zh": "zho"}
QWEN_LANG_MAP = {"en": "English", "ja": "Japanese", "ko": "Korean", "es": "Spanish", "zh": "Chinese"}
EMOTION_INSTRUCT = {
    "NEUTRAL": "natural clear cinematic dubbing tone",
    "ANGRY": "angry intense forceful tone",
    "HAPPY": "warm cheerful upbeat tone",
    "SAD": "soft emotional sorrowful tone",
    "WHISPER": "low quiet urgent whisper",
    "URGENT": "fast tense dramatic tone",
}


def _load_jsonl_map(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out[int(obj["id"])] = obj
        except Exception:
            continue
    return out


def _pick_vieneu_voice(role: dict[str, Any], stock_cfg: dict[str, Any]) -> str:
    """Ánh xạ giới tính + độ tuổi sang 1 trong 4 giọng chuẩn trong `vieneu_stock_cast.json`."""
    g = str(role.get("gender", "MALE")).upper()
    age = str(role.get("age_tone", "YOUNG")).upper()
    arch = ("NU_GIA" if age == "MATURE" else "NU_TRE") if g == "FEMALE" else ("NAM_GIA" if age == "MATURE" else "NAM_TRE")
    roles_cfg = stock_cfg.get("roles", {})
    if isinstance(roles_cfg, dict):
        entry = roles_cfg.get(arch) or roles_cfg.get("NAM_TRE")
        if isinstance(entry, dict) and entry.get("voice"):
            return str(entry["voice"])
        if isinstance(entry, str) and entry:
            return entry
    archetypes = stock_cfg.get("archetypes", {})
    if isinstance(archetypes, dict):
        pick = archetypes.get(arch) or archetypes.get("NAM_TRE")
        if pick:
            return str(pick)
    return "Hải Đăng"


def _synth_vieneu_vi(
    target_dir: Path,
    turns: list[dict[str, Any]],
    vi_map: dict[int, str],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str = "single_female",
) -> list[Path]:
    """Tạo giọng đọc Tiếng Việt (1 giọng Nữ chuẩn Trúc Ly, không clone) theo batch bằng VieNeu TTS v3turbo."""
    cache_dir = target_dir / "tts_cache" / "vi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stock_cfg = read_json_safe(ENGINE_DIR / "config" / "vieneu_stock_cast.json") or {}

    fixed_voice = _pick_vieneu_voice({"gender": "FEMALE", "age_tone": "YOUNG"}, stock_cfg)

    turn_paths: list[Path] = []
    todo: list[tuple[int, str, str, Path]] = []
    for t in turns:
        tid = int(t["id"])
        raw_txt = (vi_map.get(tid) or "").strip()
        txt = re.sub(r"[\u4e00-\u9fff]+", "", raw_txt).strip()
        voice_id = fixed_voice
        h = hashlib.sha1(f"{voice_id}|{txt}".encode("utf-8")).hexdigest()[:8]
        wav_p = cache_dir / f"turn_{tid:04d}_{h}.wav"
        turn_paths.append(wav_p)
        if txt and (not wav_p.exists() or wav_p.stat().st_size <= 2000):
            todo.append((tid, txt, voice_id, wav_p))

    if not todo:
        emit_progress(30.0, f"Tái sử dụng cache VieNeu VI ({len(turns)} câu)")
        return turn_paths

    mode = str(stock_cfg.get("mode") or "v3turbo").strip()
    emit_progress(5.0, f"Đang nạp VieNeu TTS ({mode}, {len(todo)} câu Tiếng Việt)...")
    from vieneu import Vieneu

    try:
        import torch

        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    if mode in ("v3turbo", "v3nano"):
        tts = Vieneu(mode=mode, device="cuda" if use_cuda else "cpu")
    else:
        tts = Vieneu(mode=mode, backbone_device="cuda" if use_cuda else "cpu", codec_device="cuda" if use_cuda else "cpu")
    voice_cache: dict[str, Any] = {}

    # Gom theo từng giọng để chạy infer_batch tốc độ cao trên RTX 3090
    by_voice: dict[str, list[tuple[int, str, Path]]] = {}
    for tid, txt, vid, wp in todo:
        by_voice.setdefault(vid, []).append((tid, txt, wp))

    done = 0
    for vid, items in by_voice.items():
        if vid not in voice_cache:
            try:
                voice_cache[vid] = tts.get_preset_voice(vid)
            except Exception as exc:
                print(f"[S6 VIENEU WARN] Không thấy giọng '{vid}' ({exc}) -> dùng giọng mặc định.", flush=True)
                voice_cache[vid] = tts.get_preset_voice(None)
        v_obj = voice_cache[vid]
        bsz = 32 if use_cuda else 1
        for i in range(0, len(items), bsz):
            chunk = items[i : i + bsz]
            texts = [c[1] for c in chunk]
            if bsz > 1 and hasattr(tts, "infer_batch"):
                audios = tts.infer_batch(texts=texts, voice=v_obj)
            else:
                audios = [tts.infer(text=tx, voice=v_obj) for tx in texts]
            for (_, _, wp), audio_arr in zip(chunk, audios):
                tts.save(audio_arr, str(wp))
                done += 1
            pct = 6.0 + (done / max(1, len(todo))) * 28.0
            emit_progress(pct, f"VieNeu VI: {done}/{len(todo)} câu")

    del tts
    release_vram()
    return turn_paths


def _resolve_foreign_voice_plan(
    target_dir: Path,
    lang: str,
    turns: list[dict[str, Any]],
    txt_map: dict[int, str],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str,
    engine_tag: str,
) -> tuple[bool, list[Path], list[tuple[int, str, dict[str, Any], Path]]]:
    """Lập danh sách WAV đích + phần còn thiếu (chỉ dùng 1 giọng Nữ không clone)."""
    cache_dir = target_dir / "tts_cache" / lang
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_clone = False

    turn_paths: list[Path] = []
    todo: list[tuple[int, str, dict[str, Any], Path]] = []
    for t in turns:
        tid = int(t["id"])
        txt = (txt_map.get(tid) or "").strip()
        r = roles_map.get(tid, {})
        g_eff = "FEMALE"
        sig = f"{engine_tag}|single_female|nonclone|{g_eff}|{r.get('emotion','NEUTRAL')}|{txt}"
        h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:8]
        wp = cache_dir / f"turn_{tid:04d}_{h}.wav"
        turn_paths.append(wp)
        if txt and (not wp.exists() or wp.stat().st_size <= 2000):
            todo.append((tid, txt, r, wp))
    return use_clone, turn_paths, todo


def _ensure_voxcpm_python(cfg: dict[str, Any], prog_base: float) -> Path:
    """Đảm bảo venv `/opt/venvs/voxcpm` sẵn sàng (tự cài 1 lần duy nhất nếu image `:v2` chưa có sẵn)."""
    voxcpm_py = Path(cfg.get("venvs", {}).get("voxcpm", "/opt/venvs/voxcpm/bin/python"))
    if voxcpm_py.exists():
        chk = subprocess.run([str(voxcpm_py), "-c", "import voxcpm, soundfile"], capture_output=True)
        if chk.returncode == 0:
            return voxcpm_py

    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"Không tìm thấy VoxCPM venv tại {voxcpm_py}")

    venv_dir = voxcpm_py.parent.parent
    emit_progress(prog_base, "Đang cài đặt VoxCPM2 vào venv riêng (/opt/venvs/voxcpm, 1 lần duy nhất)...")
    uv_bin = shutil.which("uv")
    if uv_bin:
        subprocess.run([uv_bin, "venv", str(venv_dir), "--python", "python3.11"], check=True)
        subprocess.run(
            [uv_bin, "pip", "install", "--python", str(voxcpm_py), "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"],
            check=True,
        )
        subprocess.run(
            [uv_bin, "pip", "install", "--python", str(voxcpm_py), "voxcpm", "soundfile", "hf-transfer"],
            check=True,
        )
    else:
        subprocess.run(["python3.11", "-m", "venv", str(venv_dir)], check=True)
        subprocess.run(
            [str(voxcpm_py), "-m", "pip", "install", "--no-cache-dir", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"],
            check=True,
        )
        subprocess.run([str(voxcpm_py), "-m", "pip", "install", "--no-cache-dir", "voxcpm", "soundfile", "hf-transfer"], check=True)

    return voxcpm_py


def _synth_voxcpm_foreign(
    target_dir: Path,
    lang: str,
    turns: list[dict[str, Any]],
    txt_map: dict[int, str],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str,
    cfg: dict[str, Any],
    prog_base: float,
    prog_span: float,
) -> list[Path]:
    """Tạo giọng đọc EN / JA / KO / ES bằng OpenBMB/VoxCPM2 (48kHz, không clone giọng gốc)."""
    use_clone, turn_paths, todo = _resolve_foreign_voice_plan(
        target_dir, lang, turns, txt_map, roles_map, voice_mode, engine_tag="voxcpm2"
    )
    if not todo:
        emit_progress(prog_base + prog_span, f"Tái sử dụng cache VoxCPM2 {lang.upper()} ({len(turns)} câu)")
        return turn_paths

    voxcpm_py = _ensure_voxcpm_python(cfg, prog_base)
    cache_dir = target_dir / "tts_cache" / lang
    model_id = cfg.get("models", {}).get("tts_foreign_voxcpm_model", "openbmb/VoxCPM2")
    worker_py = ENGINE_DIR / "pipeline" / "voxcpm_worker.py"

    tasks_payload = []
    for tid, txt, r, wp in todo:
        tasks_payload.append(
            {
                "tid": tid,
                "text": txt,
                "gender": "FEMALE",
                "emotion": str(r.get("emotion", "NEUTRAL")).upper(),
                "out_wav": str(wp),
            }
        )

    manifest_path = cache_dir / "_voxcpm_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "model_id": model_id,
            "lang": lang,
            "voice_mode": "single_female",
            "use_clone": False,
            "anchor_paths": {
                "FEMALE": str(cache_dir / "synth_anchor_female.wav"),
            },
            "ref_paths": {},
            "tasks": tasks_payload,
        },
    )

    emit_progress(prog_base, f"VoxCPM2 {lang.upper()} (Giọng Nữ nhân tạo, {len(todo)} câu)...")
    env = dict(os.environ)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    proc = subprocess.Popen(
        [str(voxcpm_py), "-u", str(worker_py), "--manifest", str(manifest_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"\[VOXCPM_PROGRESS\]\s*(\d+)/(\d+)", line)
        if m:
            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
            if cur % 3 == 0 or cur == tot:
                emit_progress(prog_base + (cur / tot) * prog_span, f"VoxCPM2 {lang.upper()}: {cur}/{tot} câu")
        else:
            print(line, flush=True)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"voxcpm_worker thoát với mã lỗi {rc} cho ngôn ngữ {lang.upper()}")
    return turn_paths


def _synth_voxcpm_batch_foreign(
    target_dir: Path,
    foreign_langs: list[str],
    lang_tmaps: dict[str, dict[int, str]],
    turns: list[dict[str, Any]],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str,
    cfg: dict[str, Any],
    prog_base: float,
    prog_span: float,
) -> dict[str, list[Path]]:
    """Nạp VoxCPM2 ĐÚNG 1 LẦN để tổng hợp đồng loạt tất cả ngoại ngữ (EN/JA/KO/ES), giảm ~3x thời gian nạp."""
    results: dict[str, list[Path]] = {}
    lang_jobs: list[dict[str, Any]] = []
    use_clone_global = False
    total_todo = 0

    for lang in foreign_langs:
        _, turn_paths, todo = _resolve_foreign_voice_plan(
            target_dir, lang, turns, lang_tmaps[lang], roles_map, "single_female", engine_tag="voxcpm2"
        )
        results[lang] = turn_paths
        if todo:
            cache_dir = target_dir / "tts_cache" / lang
            tasks_payload = []
            for tid, txt, r, wp in todo:
                tasks_payload.append(
                    {
                        "tid": tid,
                        "text": txt,
                        "gender": "FEMALE",
                        "emotion": str(r.get("emotion", "NEUTRAL")).upper(),
                        "out_wav": str(wp),
                    }
                )
            total_todo += len(tasks_payload)
            lang_jobs.append(
                {
                    "lang": lang,
                    "anchor_paths": {
                        "FEMALE": str(cache_dir / "synth_anchor_female.wav"),
                    },
                    "ref_paths": {},
                    "tasks": tasks_payload,
                }
            )

    if total_todo == 0:
        emit_progress(prog_base + prog_span, f"Tái sử dụng cache VoxCPM2 ({','.join(foreign_langs).upper()})")
        return results

    voxcpm_py = _ensure_voxcpm_python(cfg, prog_base)
    model_id = cfg.get("models", {}).get("tts_foreign_voxcpm_model", "openbmb/VoxCPM2")
    worker_py = ENGINE_DIR / "pipeline" / "voxcpm_worker.py"
    manifest_path = target_dir / "tts_cache" / "_voxcpm_batch_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "model_id": model_id,
            "voice_mode": "single_female",
            "use_clone": False,
            "languages": lang_jobs,
        },
    )

    emit_progress(prog_base, f"Nạp 1 lần VoxCPM2 cho {', '.join(foreign_langs).upper()} ({total_todo} câu)...")
    env = dict(os.environ)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    proc = subprocess.Popen(
        [str(voxcpm_py), "-u", str(worker_py), "--manifest", str(manifest_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"\[VOXCPM_PROGRESS\]\s*(\d+)/(\d+)(?:\s+([A-Z]+))?", line)
        if m:
            cur, tot = int(m.group(1)), max(1, int(m.group(2)))
            lg_tag = m.group(3) or "FOREIGN"
            if cur % 4 == 0 or cur == tot:
                emit_progress(prog_base + (cur / tot) * prog_span, f"VoxCPM2 {lg_tag}: {cur}/{tot} câu")
        else:
            print(line, flush=True)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"voxcpm_worker batch thoát với mã lỗi {rc}")
    return results


def _synth_qwen3_tts_foreign(
    target_dir: Path,
    lang: str,
    turns: list[dict[str, Any]],
    txt_map: dict[int, str],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str,
    cfg: dict[str, Any],
    prog_base: float,
    prog_span: float,
    model_holder: dict[str, Any] | None = None,
) -> list[Path]:
    """Dự phòng tạo giọng Nữ EN / JA / KO / ES bằng Qwen3-TTS-1.7B-VoiceDesign (không clone)."""
    _, turn_paths, todo = _resolve_foreign_voice_plan(
        target_dir, lang, turns, txt_map, roles_map, "single_female", engine_tag="qwen3"
    )

    if not todo:
        emit_progress(prog_base + prog_span, f"Tái sử dụng cache Qwen3-TTS {lang.upper()} ({len(turns)} câu)")
        return turn_paths

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    models_cfg = cfg.get("models", {})
    model_id = models_cfg.get("tts_foreign_design_model", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    own_model = model_holder is None
    if model_holder is not None and model_holder.get("model_id") == model_id and model_holder.get("model") is not None:
        model = model_holder["model"]
        emit_progress(prog_base, f"Dùng lại Qwen3-TTS cho {lang.upper()} ({len(todo)} câu)...")
    else:
        emit_progress(prog_base, f"Nạp Qwen3-TTS ({lang.upper()}, Giọng Nữ Voice Design)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
        if model_holder is not None:
            model_holder["model_id"] = model_id
            model_holder["model"] = model
    q_lang = QWEN_LANG_MAP.get(lang, "English")

    for idx, (tid, txt, r, wp) in enumerate(todo, start=1):
        emo = EMOTION_INSTRUCT.get(str(r.get("emotion", "NEUTRAL")).upper(), EMOTION_INSTRUCT["NEUTRAL"])
        persona = f"Confident adult female native {q_lang} speaker, {emo}"
        wavs, sr_out = model.generate_voice_design(text=txt, language=q_lang, instruct=persona)
        sf.write(str(wp), wavs[0], sr_out)
        if idx % 5 == 0 or idx == len(todo):
            emit_progress(prog_base + (idx / len(todo)) * prog_span, f"Qwen3-TTS {lang.upper()}: {idx}/{len(todo)} câu")

    if own_model:
        del model
        release_vram()
    return turn_paths


def _synth_foreign_tts(
    target_dir: Path,
    lang: str,
    turns: list[dict[str, Any]],
    txt_map: dict[int, str],
    roles_map: dict[int, dict[str, Any]],
    voice_mode: str,
    cfg: dict[str, Any],
    prog_base: float,
    prog_span: float,
    qwen_holder: dict[str, Any] | None = None,
) -> list[Path]:
    """Ưu tiên VoxCPM2 cho ngoại ngữ; tự động dự phòng bằng Qwen3-TTS nếu VoxCPM2 không sẵn sàng."""
    engine = str(cfg.get("models", {}).get("tts_foreign_engine", "voxcpm2")).lower()
    if engine == "voxcpm2":
        try:
            return _synth_voxcpm_foreign(target_dir, lang, turns, txt_map, roles_map, voice_mode, cfg, prog_base, prog_span)
        except Exception as exc:
            print(f"[S6 VOXCPM WARN] VoxCPM2 lỗi ({exc}) -> chuyển dự phòng sang Qwen3-TTS-VoiceDesign.", flush=True)
    return _synth_qwen3_tts_foreign(target_dir, lang, turns, txt_map, roles_map, voice_mode, cfg, prog_base, prog_span, model_holder=qwen_holder)


def _prepare_slot_pcm(wav_path: Path, slot_dur: float) -> np.ndarray:
    """Chuyển 1 câu TTS sang 48kHz mono float32 và tăng tốc nhẹ (tối đa 1.32x) nếu câu dài hơn khung giờ."""
    tmp_48 = wav_path.with_suffix(".48k.wav")
    try:
        # Đo độ dài gốc của câu TTS
        out_probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(wav_path)],
            text=True,
        ).strip()
        raw_dur = float(out_probe or 0.0)
        af = "aresample=48000"
        if raw_dur > slot_dur > 0.25:
            spd = min(1.32, raw_dur / slot_dur)
            af = f"atempo={spd:.3f},aresample=48000"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(wav_path), "-af", af, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp_48)],
            check=True,
        )
        with wave.open(str(tmp_48), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        pk = float(np.max(np.abs(arr))) + 1e-8
        if pk > 0.05:
            arr = arr * min(1.8, 0.86 / pk)
        return arr
    finally:
        tmp_48.unlink(missing_ok=True)


def _mix_track_streaming(turns: list[dict[str, Any]], turn_files: list[Path], clean_bed_wav: Path, out_m4a: Path):
    """Trộn giọng TTS vào 04_clean_bed.wav theo từng chunk 30s (nhẹ RAM, giữ nguyên 100% SFX ngoài thoại)."""
    out_m4a.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(clean_bed_wav), "rb") as wf_bed:
        total_frames = wf_bed.getnframes()
        total_dur = total_frames / float(SR)

    planned: list[tuple[int, int, np.ndarray]] = []
    cursor_end = 0
    for idx, t in enumerate(turns):
        wp = turn_files[idx]
        if not wp.exists() or wp.stat().st_size <= 2000:
            continue
        start_s = float(t["start"])
        next_s = float(turns[idx + 1]["start"]) if idx + 1 < len(turns) else total_dur
        hard_end = min(next_s - 0.04, max(float(t["end"]) + 0.30, start_s + 0.55))
        slot_dur = max(0.30, hard_end - start_s)
        samples = _prepare_slot_pcm(wp, slot_dur)
        if len(samples) == 0:
            continue
        s_idx = int(round(start_s * SR))
        min_idx = cursor_end + int(0.03 * SR) if cursor_end > 0 else 0
        if s_idx < min_idx:
            s_idx = min_idx
        if s_idx >= total_frames:
            break
        e_idx = min(total_frames, s_idx + len(samples))
        planned.append((s_idx, e_idx, np.clip(samples[: e_idx - s_idx] * 32767.0, -32768, 32767).astype(np.int16)))
        cursor_end = e_idx

    raw_wav = out_m4a.with_suffix(".tmp.wav")
    duck_ramp = int(0.14 * SR)
    chunk_frames = 30 * SR
    try:
        with wave.open(str(clean_bed_wav), "rb") as wf_bed, wave.open(str(raw_wav), "wb") as wf_out:
            wf_out.setnchannels(2)
            wf_out.setsampwidth(2)
            wf_out.setframerate(SR)
            for c0 in range(0, total_frames, chunk_frames):
                c1 = min(total_frames, c0 + chunk_frames)
                clen = c1 - c0
                wf_bed.setpos(c0)
                bed = np.frombuffer(wf_bed.readframes(clen), dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768.0
                gain = np.full(clen, 0.94, dtype=np.float32)
                speech = np.zeros(clen, dtype=np.float32)
                for ts, te, s16 in planned:
                    if te + duck_ramp <= c0 or ts - duck_ramp >= c1:
                        continue
                    o0, o1 = max(c0, ts), min(c1, te)
                    if o1 > o0:
                        speech[o0 - c0 : o1 - c0] += s16[o0 - ts : o1 - ts].astype(np.float32) / 32768.0
                        gain[o0 - c0 : o1 - c0] = np.minimum(gain[o0 - c0 : o1 - c0], 0.45)
                mixed = bed * gain[:, None] + speech[:, None]
                pk = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
                if pk > 0.96:
                    mixed *= 0.96 / pk
                wf_out.writeframes(np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16).tobytes())

        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(raw_wav), "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2", str(out_m4a)],
            check=True,
        )
    finally:
        raw_wav.unlink(missing_ok=True)


def _detect_sub_blur_rect(cfr_mp4: Path, turns: list[dict[str, Any]] | None = None) -> tuple[int, int, int, int]:
    """Đo vị trí dọc thực tế của phụ đề Trung trên video, mở rộng chiều ngang gần hết khung (~94%) để không bao giờ hở chữ dài."""
    default_rect = (32, 1434, 1016, 124)
    try:
        from _engine.core.vram_guard import patch_onnx_threads

        patch_onnx_threads(4)
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return default_rect

    # Chọn tối đa 16 mốc thời gian: ưu tiên 10 câu dài nhất + 6 mốc rải đều
    sample_ts: list[float] = []
    if turns:
        valid = [t for t in turns if float(t.get("end", 0)) > float(t.get("start", 0))]
        by_len = sorted(valid, key=lambda t: len(str(t.get("text", ""))), reverse=True)[:10]
        step = max(1, len(valid) // 6)
        by_spread = valid[::step][:6]
        seen = set()
        for t in by_len + by_spread:
            mid = round((float(t["start"]) + float(t["end"])) / 2.0, 2)
            if mid not in seen:
                seen.add(mid)
                sample_ts.append(mid)
    if not sample_ts:
        sample_ts = [5.0, 25.0, 40.0, 49.0, 70.0, 90.0, 110.0]

    ocr = RapidOCR()
    ys0: list[float] = []
    ys1: list[float] = []
    W, H = 1080, 1920

    try:
        import av

        with av.open(str(cfr_mp4)) as cont:
            st = cont.streams.video[0]
            tb = float(st.time_base)
            W, H = int(st.width or 1080), int(st.height or 1920)
            y_top, y_bot = int(H * 0.56), int(H * 0.94)
            max_line_h = int(H * 0.08)
            for sec in sample_ts:
                try:
                    cont.seek(max(0, int(sec / tb)), stream=st, backward=True)
                    for step_f, frame in enumerate(cont.decode(st)):
                        if frame.pts is not None and float(frame.pts * tb) < sec - 0.15 and step_f < 60:
                            continue
                        arr = frame.to_ndarray(format="bgr24")
                        res, _ = ocr(arr[y_top:y_bot, :])
                        if res:
                            for box, txt, conf in res:
                                if float(conf) < 0.55 or not any("\u4e00" <= c <= "\u9fff" for c in txt):
                                    continue
                                by0 = min(p[1] for p in box) + y_top
                                by1 = max(p[1] for p in box) + y_top
                                if (by1 - by0) > max_line_h:
                                    continue
                                ys0.append(by0)
                                ys1.append(by1)
                        break
                except Exception:
                    continue
    except Exception:
        return default_rect

    bx = max(0, int(round(W * 0.03))) & ~1
    bw = min(W - bx, W - 2 * bx) & ~1
    if not ys0:
        return bx, max(0, int(round(H * 0.74))) & ~1, bw, 124

    if len(ys0) >= 6:
        raw_y0 = float(np.percentile(ys0, 2))
        raw_y1 = float(np.percentile(ys1, 98))
    else:
        raw_y0, raw_y1 = min(ys0), max(ys1)

    pad_y = 20
    y0 = max(int(H * 0.50), int(raw_y0 - pad_y))
    y1 = min(H - 16, int(raw_y1 + pad_y))
    bh = max(100, min(240, y1 - y0))
    by = max(0, min(H - bh, int(round((y0 + y1) / 2.0 - bh / 2.0)))) & ~1
    bh = min(H - by, bh) & ~1
    print(f"[S6 SUB-BLUR] Chiều ngang gần hết ({bw}px) + đo dọc từ {len(ys0)} dòng sub Trung -> ({bx}, {by}, {bw}x{bh})", flush=True)
    return bx, by, bw, bh


def _compute_wm_alpha_scale(wm_png: Path, target_opacity: float = 0.10) -> float:
    """Tính hệ số `aa` cho colorchannelmixer sao cho độ đục cuối của watermark đạt đúng ~10%."""
    try:
        from PIL import Image

        arr = np.array(Image.open(wm_png).convert("RGBA"))
        max_a = int(arr[:, :, 3].max())
        if max_a > 0:
            return round(min(1.0, max(0.02, (target_opacity * 255.0) / float(max_a))), 3)
    except Exception:
        pass
    return round(target_opacity, 3)


def _render_master_mp4(
    cfr_mp4: Path,
    master_audio: Path,
    master_srt: Path | None,
    out_mp4: Path,
    turns: list[dict[str, Any]] | None = None,
    primary_lang: str = "vi",
):
    """Render MP4 9:16 chuẩn chất lượng cao: làm mờ sub Trung + watermark 10% đè lên dải mờ + soft sub."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc_name, hw_args, enc_args = detect_gpu_encoder()
    wm_png = ROOT_DIR / "branding" / "MT_Reel_Strip_Watermark.png"
    iso3 = LANG_ISO3.get(primary_lang, "und")

    bx, by, bw, bh = _detect_sub_blur_rect(cfr_mp4, turns)
    lr = max(6, min(32, min(bw, bh) // 2 - 2))
    cr = max(4, min(18, min(bw, bh) // 4 - 2))

    has_wm = wm_png.exists()
    wm_aa = _compute_wm_alpha_scale(wm_png, target_opacity=0.10) if has_wm else 0.10

    def _build_cmd(use_hw: bool) -> list[str]:
        cmd = ["ffmpeg", "-y", "-v", "warning"]
        if use_hw:
            cmd += [*hw_args]
        cmd += ["-i", str(cfr_mp4), "-i", str(master_audio)]
        wm_idx = None
        if has_wm:
            cmd += ["-i", str(wm_png)]
            wm_idx = 2

        blur_chain = (
            f"[0:v]split=2[fgraw][fgsub];"
            f"[fgsub]crop={bw}:{bh}:{bx}:{by},boxblur=luma_radius={lr}:luma_power=4:chroma_radius={cr}:chroma_power=3[subblur];"
        )
        if wm_idx is not None:
            fc = (
                f"{blur_chain}"
                f"[fgraw][subblur]overlay={bx}:{by}[vblur];"
                f"[{wm_idx}:v]scale={bw}:{bh}:force_original_aspect_ratio=decrease,format=rgba,colorchannelmixer=aa={wm_aa:.3f}[wm];"
                f"[vblur][wm]overlay={bx}+({bw}-w)/2:{by}+({bh}-h)/2,format=yuv420p[vout]"
            )
        else:
            fc = f"{blur_chain}[fgraw][subblur]overlay={bx}:{by},format=yuv420p[vout]"

        v_enc = enc_args if use_hw else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]
        # Không nhúng track mov_text vào trong MP4 để trình phát video không tự bật chữ đè góc màn hình; file .srt vẫn lưu riêng ở subtitles/
        cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "1:a:0", *v_enc, "-c:a", "copy", "-sn", "-metadata:s:a:0", f"language={iso3}", "-movflags", "+faststart"]
        cmd.append(str(out_mp4))
        return cmd

    cmd = _build_cmd(use_hw=(enc_name != "libx264"))
    print(f"[S6 RENDER] Encoder={enc_name} | SubBlur=({bx},{by},{bw}x{bh}) | WM_aa={wm_aa} -> {out_mp4}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0 and enc_name != "libx264":
        print(f"[S6 RENDER WARN] {enc_name} lỗi (exit={res.returncode}) -> tự chuyển sang libx264...", flush=True)
        subprocess.run(_build_cmd(use_hw=False), check=True)
    elif res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)


def _mux_streamcopy_lang_mp4(
    blurred_mp4: Path,
    audio_m4a: Path,
    srt_path: Path | None,
    lang: str,
    out_mp4: Path,
):
    """Nhân bản nhanh (<1s) video đã làm mờ + watermark sang từng ngôn ngữ bằng `-c:v copy -c:a copy`."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    iso3 = LANG_ISO3.get(lang, "und")
    cmd = [
        "ffmpeg", "-y", "-v", "warning", "-i", str(blurred_mp4), "-i", str(audio_m4a),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-sn",
        "-metadata:s:a:0", f"language={iso3}",
        "-movflags", "+faststart", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def run_step6(target_dir: Path, work_rel: str, langs: list[str], voice_mode: str = "single_female", is_test_5min: bool = False) -> dict[str, Any]:
    master = read_json_safe(target_dir / "03_aligned_master.json")
    clean_bed = target_dir / "04_clean_bed.wav"
    cfr_mp4 = target_dir / "01_cfr_raw.mp4"
    if not master or not clean_bed.exists() or not cfr_mp4.exists():
        raise FileNotFoundError("Thiếu file đầu vào từ Bước 1/3/4 cho Bước 6.")

    t0 = time.perf_counter()
    cfg = load_pipeline_config()
    turns = master["turns"]
    roles_map = _load_jsonl_map(target_dir / "05a_roles.jsonl")

    out_base = OUTPUT_ROOT / work_rel / ("preview_5min" if is_test_5min else "")
    audio_out_dir = out_base / "audio"
    sub_out_dir = out_base / "subtitles"
    info_out_dir = out_base / "_info"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    info_out_dir.mkdir(parents=True, exist_ok=True)

    # 6A. Tiếng Việt (VieNeu)
    if "vi" in langs:
        vi_map = {k: v["vi"] for k, v in _load_jsonl_map(target_dir / "05b_sub_vi.jsonl").items()}
        vi_files = _synth_vieneu_vi(target_dir, turns, vi_map, roles_map, voice_mode=voice_mode)
        emit_progress(35.0, "Đang trộn audio_vi.m4a lên nền sạch 04_clean_bed.wav...")
        _mix_track_streaming(turns, vi_files, clean_bed, audio_out_dir / "audio_vi.m4a")
        if (target_dir / "tracks" / "vi.srt").exists():
            shutil.copy2(target_dir / "tracks" / "vi.srt", sub_out_dir / "vi.srt")

    # 6B. Ngoại ngữ (VoxCPM2 nạp 1 lần cho cả EN/JA/KO/ES, tự động dự phòng Qwen3-TTS nếu lỗi)
    foreign_langs = [x for x in langs if x != "vi"]
    if foreign_langs:
        lang_tmaps = {
            lang: {k: v["text"] for k, v in _load_jsonl_map(target_dir / f"05c_sub_{lang}.jsonl").items()}
            for lang in foreign_langs
        }
        engine = str(cfg.get("models", {}).get("tts_foreign_engine", "voxcpm2")).lower()
        lang_wavs: dict[str, list[Path]] = {}
        if engine == "voxcpm2":
            try:
                lang_wavs = _synth_voxcpm_batch_foreign(
                    target_dir, foreign_langs, lang_tmaps, turns, roles_map, voice_mode, cfg, 38.0, 38.0
                )
            except Exception as exc:
                print(f"[S6 VOXCPM WARN] VoxCPM2 batch lỗi ({exc}) -> chuyển dự phòng sang Qwen3-TTS.", flush=True)

        span = 45.0 / len(foreign_langs)
        qwen_holder: dict[str, Any] = {}
        try:
            for idx, lang in enumerate(foreign_langs):
                wavs = lang_wavs.get(lang)
                if wavs is None:
                    wavs = _synth_qwen3_tts_foreign(
                        target_dir, lang, turns, lang_tmaps[lang], roles_map, voice_mode, cfg, 38.0 + idx * span, span * 0.85, model_holder=qwen_holder
                    )
                _mix_track_streaming(turns, wavs, clean_bed, audio_out_dir / f"audio_{lang}.m4a")
                if (target_dir / "tracks" / f"{lang}.srt").exists():
                    shutil.copy2(target_dir / "tracks" / f"{lang}.srt", sub_out_dir / f"{lang}.srt")
        finally:
            if qwen_holder:
                qwen_holder.clear()
                release_vram()

    qa_html = target_dir / "tracks" / "kiem_tra_dich.html"
    if qa_html.exists():
        shutil.copy2(qa_html, sub_out_dir / "kiem_tra_dich.html")

    # Gom glossary, relationship_bible & các file kịch bản đã dịch vào _info/ để lưu trữ lâu dài trên Drive
    for ref_name in (
        "glossary.json",
        "relationship_bible.json",
        "03_aligned_master.json",
        "05a_roles.jsonl",
        "05b_sub_vi.jsonl",
    ):
        rp = target_dir / ref_name
        if rp.exists():
            shutil.copy2(rp, info_out_dir / ref_name)

    # 6C. Render MP4 thành phẩm (+ xuất MP4 riêng từng ngôn ngữ ở chế độ Test để nghe kiểm tra ngay)
    emit_progress(86.0, "Đang render MP4 thành phẩm bằng GPU (NVENC/QSV)...")
    primary_lang = "vi" if "vi" in langs else langs[0]
    master_audio = audio_out_dir / f"audio_{primary_lang}.m4a"
    master_srt = sub_out_dir / f"{primary_lang}.srt"
    prefix = "preview_5min" if is_test_5min else "final"
    mp4_name = f"{prefix}_{primary_lang}.mp4"
    out_mp4 = out_base / mp4_name
    _render_master_mp4(cfr_mp4, master_audio, master_srt, out_mp4, turns=turns, primary_lang=primary_lang)

    exported_mp4s = [str(out_mp4)]
    if is_test_5min:
        for extra_lang in langs:
            if extra_lang == primary_lang:
                continue
            extra_audio = audio_out_dir / f"audio_{extra_lang}.m4a"
            extra_srt = sub_out_dir / f"{extra_lang}.srt"
            if extra_audio.exists():
                extra_mp4 = out_base / f"{prefix}_{extra_lang}.mp4"
                emit_progress(95.0, f"Đang xuất MP4 test {extra_lang.upper()} (-c:v copy)...")
                _mux_streamcopy_lang_mp4(out_mp4, extra_audio, extra_srt, extra_lang, extra_mp4)
                exported_mp4s.append(str(extra_mp4))

    elapsed = round(time.perf_counter() - t0, 2)
    meta = {
        "elapsed_s": elapsed,
        "mode": "test_5min" if is_test_5min else "full",
        "langs": langs,
        "voice_mode": voice_mode,
        "output_mp4": str(out_mp4),
        "output_mp4s": exported_mp4s,
    }
    atomic_write_json(info_out_dir / "render_meta.json", meta)
    emit_progress(100.0, f"Hoàn tất xuất phim ({len(exported_mp4s)} file MP4, {elapsed}s)")
    print(f"[S6 DONE] {out_mp4} (+{len(exported_mp4s)-1} bản ngôn ngữ khác) trong {elapsed}s", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Bước 6: VieNeu (VI) + VoxCPM2 (EN/JA/KO/ES) + Mix + Render GPU.")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu")
    parser.add_argument("--work-rel", required=True, help="Đường dẫn tương đối <Series>/<Part>")
    parser.add_argument("--langs", default="vi,en,ja,ko,es", help="Ngôn ngữ xuất")
    parser.add_argument("--voice-mode", default="single_female", help="Chỉ dùng 1 giọng Nữ không clone (single_female)")
    parser.add_argument("--test-5min", action="store_true", help="Xuất vào thư mục preview_5min")
    args = parser.parse_args()
    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    run_step6(args.target_dir.resolve(), work_rel=args.work_rel, langs=langs, voice_mode=args.voice_mode, is_test_5min=args.test_5min)


if __name__ == "__main__":
    main()

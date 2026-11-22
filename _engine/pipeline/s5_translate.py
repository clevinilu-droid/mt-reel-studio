"""Bước 5 (s5_translate): Phân công đúng 2 dòng model mới nhất chạy 100% trên RTX 3090 (vLLM).

Luồng tuần tự tiết kiệm 24 GB VRAM (xong model nào tắt sạch model đó):
  - Pha 5A (Tiếng Trung): Qwen3.5 / Qwen3.8 -> lập `glossary.json`, `relationship_bible.json`,
    gán giới tính/độ tuổi/cảm xúc/xưng hô cho từng câu (`05a_roles.jsonl`, checkpoint mỗi 50 câu).
  - Pha 5B (Tiếng Việt): Gemma 4 (`gemma-4-26B-A4B-it`) -> dịch thoại Việt tự nhiên khớp thời lượng
    (`05b_sub_vi.jsonl`, checkpoint mỗi 40 câu) -> `tracks/vi.srt`.
  - Pha 5C (Ngoại ngữ EN/JA/KO/ES): TranslateGemma-27B -> dịch sạch đa ngôn ngữ
    (`05c_sub_<lang>.jsonl`, checkpoint mỗi 40 câu) -> `tracks/<lang>.srt`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from _engine.core.checkpoint import atomic_write_json, read_json_safe
from _engine.core.vram_guard import emit_progress, load_pipeline_config, release_vram
from _engine.pipeline.s3_audit import format_srt_ts

LANG_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
}


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@contextmanager
def managed_vllm_server(model_id: str, cfg: dict[str, Any]):
    """Tự khởi chạy vLLM OpenAI server cho 1 model và tắt sạch khi thoát để trả 100% 24GB VRAM.

    Nếu biến môi trường `MT_LLM_BASE_URL` đã được đặt sẵn (ví dụ user tự chạy vLLM/Ollama/LM Studio),
    hàm sẽ dùng thẳng endpoint đó mà không khởi chạy tiến trình mới.
    """
    ext_url = os.environ.get("MT_LLM_BASE_URL", "").strip()
    if ext_url:
        yield ext_url.rstrip("/"), model_id
        return

    vcfg = cfg.get("vllm", {})
    host = vcfg.get("host", "127.0.0.1")
    port = int(vcfg.get("port", 8001))
    gpu_util = str(vcfg.get("gpu_memory_utilization", 0.88))
    max_len = str(vcfg.get("max_model_len", 8192))
    base_url = f"http://{host}:{port}/v1"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        gpu_util,
        "--max-model-len",
        max_len,
        "--trust-remote-code",
    ]
    low_id = model_id.lower()
    if any(k in low_id for k in ("26b", "27b", "31b", "32b")) and not any(q in low_id for q in ("awq", "gptq", "fp8", "gguf")):
        cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    import tempfile

    print(f"[S5 vLLM] Khởi chạy {model_id} tại {base_url}...", flush=True)
    log_fh = tempfile.NamedTemporaryFile(mode="w+", prefix="vllm_", suffix=".log", delete=False, encoding="utf-8")
    log_path = Path(log_fh.name)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    try:
        # Chờ vLLM tải weights (lần đầu ~27B có thể mất 5–10 phút) & nạp VRAM (tối đa 900s)
        deadline = time.time() + 900.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_fh.flush()
                err = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
                raise RuntimeError(f"vLLM khởi chạy thất bại cho {model_id}: {err}")
            try:
                _http_json(f"{base_url}/models", timeout=3)
                ready = True
                break
            except Exception:
                time.sleep(2.0)
        if not ready:
            log_fh.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
            raise TimeoutError(f"Quá thời gian chờ vLLM khởi động cho {model_id}: {tail}")
        yield base_url, model_id
    finally:
        print(f"[S5 vLLM] Đang tắt {model_id} để giải phóng sạch 24GB VRAM...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()
        log_path.unlink(missing_ok=True)
        time.sleep(2.0)
        release_vram()


def _parse_json_lenient(raw: str) -> Any:
    txt = (raw or "").strip()
    if "```" in txt:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt)
        if m:
            txt = m.group(1).strip()
    # Bỏ khối <think>...</think> nếu model reasoning sinh ra
    txt = re.sub(r"<think>[\s\S]*?</think>", "", txt).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", txt)
        if m:
            return json.loads(m.group(1))
        raise


def _chat_json(base_url: str, model_id: str, system_prompt: str, user_prompt: str, temp: float = 0.15) -> Any:
    last_err = None
    for attempt in range(3):
        try:
            resp = _http_json(
                f"{base_url}/chat/completions",
                {
                    "model": model_id,
                    "temperature": temp,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=240,
            )
            content = resp["choices"][0]["message"]["content"]
            return _parse_json_lenient(content)
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gọi LLM thất bại sau 3 lần ({model_id}): {last_err}")


def _load_jsonl_map(path: Path, key: str = "id") -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out[int(obj[key])] = obj
        except Exception:
            continue
    return out


def _append_jsonl_batch(path: Path, items: list[dict[str, Any]]):
    with path.open("a", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def _write_srt(turns: list[dict[str, Any]], text_map: dict[int, str], out_srt: Path):
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t in turns:
        tid = int(t["id"])
        txt = (text_map.get(tid) or t.get("text", "")).strip()
        lines.append(f"{tid}\n{format_srt_ts(t['start'])} --> {format_srt_ts(t['end'])}\n{txt}\n")
    out_srt.write_text("\n".join(lines), encoding="utf-8")


def _phase5a_qwen_analyze(
    target_dir: Path,
    turns: list[dict[str, Any]],
    acoustics_map: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    base_glossary: dict[str, Any],
    force: bool = False,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Pha 5A: Dùng Qwen3.5/3.8 lập Glossary tiếng Trung + phân vai giới tính/tuổi/cảm xúc/xưng hô."""
    roles_jsonl = target_dir / "05a_roles.jsonl"
    glossary_path = target_dir / "glossary.json"
    bible_path = target_dir / "relationship_bible.json"
    if force:
        roles_jsonl.unlink(missing_ok=True)

    done_roles = _load_jsonl_map(roles_jsonl)
    glossary = read_json_safe(glossary_path) or dict(base_glossary)
    bible = read_json_safe(bible_path) or {"characters": [], "relationships": []}

    todo_turns = [t for t in turns if int(t["id"]) not in done_roles]
    if not todo_turns and glossary_path.exists():
        emit_progress(32.0, f"Tái sử dụng Pha 5A Qwen ({len(done_roles)} câu đã phân vai)")
        return glossary, done_roles

    model_id = cfg.get("models", {}).get("zh_analyst_model", "Qwen/Qwen3.8-27B")
    emit_progress(5.0, f"Pha 5A: Nạp {model_id} phân tích tiếng Trung & phân vai...")

    sys_prompt = (
        "Bạn là biên kịch kiêm đạo diễn lồng tiếng chuyên phim ngắn Trung Quốc. "
        "Trả về DUY NHẤT JSON hợp lệ, không giải thích thêm."
    )
    batch_size = 50
    with managed_vllm_server(model_id, cfg) as (base_url, mid):
        for i in range(0, len(todo_turns), batch_size):
            batch = todo_turns[i : i + batch_size]
            packed = []
            for t in batch:
                ac = acoustics_map.get(int(t["id"]), {})
                packed.append(
                    {
                        "id": t["id"],
                        "spk_id": t.get("spk_id", "SPK_UNK"),
                        "f0_hz": ac.get("f0_hz"),
                        "gender_hint": ac.get("gender_hint", "UNKNOWN"),
                        "zh": t["text"],
                    }
                )

            user_prompt = (
                "Dựa vào lời thoại tiếng Trung, nhãn người nói `spk_id` và tần số giọng `f0_hz` "
                "(Nam thường <175Hz, Nữ >=175Hz; ưu tiên ngữ cảnh khi F0 nhiễu), hãy xuất JSON gồm:\n"
                "1) `glossary_add`: dict {tên_tiếng_Trung: tên_Hán_Việt_chuẩn}\n"
                "2) `roles`: list cho từng id với các trường: "
                "`id`, `gender` ('MALE'|'FEMALE'), `age_tone` ('YOUNG'|'MATURE'), "
                "`emotion` ('NEUTRAL'|'ANGRY'|'HAPPY'|'SAD'|'WHISPER'|'URGENT'), "
                "`addressing` (gợi ý cặp xưng hô tiếng Việt ngắn gọn, vd 'tôi - cô', 'anh - em', 'tao - mày').\n\n"
                f"Glossary hiện có: {json.dumps(glossary, ensure_ascii=False)}\n"
                f"Danh sách câu:\n{json.dumps(packed, ensure_ascii=False)}"
            )
            res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.1)
            for k, v in (res.get("glossary_add") or {}).items():
                if k and v and k not in glossary:
                    glossary[k] = v
            atomic_write_json(glossary_path, glossary)

            role_items = []
            res_map = {int(r["id"]): r for r in (res.get("roles") or []) if "id" in r}
            for t in batch:
                tid = int(t["id"])
                ac = acoustics_map.get(tid, {})
                r = res_map.get(tid, {})
                g = str(r.get("gender", ac.get("gender_hint", "MALE"))).upper()
                if g not in ("MALE", "FEMALE"):
                    g = "FEMALE" if ac.get("gender_hint") == "FEMALE" else "MALE"
                item = {
                    "id": tid,
                    "gender": g,
                    "age_tone": str(r.get("age_tone", "YOUNG")).upper(),
                    "emotion": str(r.get("emotion", "NEUTRAL")).upper(),
                    "addressing": str(r.get("addressing", "tôi - anh/cô")),
                }
                role_items.append(item)
                done_roles[tid] = item
            _append_jsonl_batch(roles_jsonl, role_items)

            pct = 8.0 + (len(done_roles) / max(1, len(turns))) * 25.0
            emit_progress(pct, f"Pha 5A Qwen: đã phân vai {len(done_roles)}/{len(turns)} câu")

    atomic_write_json(glossary_path, glossary)
    atomic_write_json(bible_path, bible)
    return glossary, done_roles


def _phase5b_gemma4_vi(
    target_dir: Path,
    turns: list[dict[str, Any]],
    glossary: dict[str, Any],
    roles_map: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    force: bool = False,
    conn: tuple[str, str] | None = None,
) -> dict[int, str]:
    """Pha 5B: Dùng Gemma 4 (gemma-4-26B-A4B-it) dịch thoại sang Tiếng Việt tự nhiên khớp thời lượng."""
    vi_jsonl = target_dir / "05b_sub_vi.jsonl"
    if force:
        vi_jsonl.unlink(missing_ok=True)
    done_map = {k: v["vi"] for k, v in _load_jsonl_map(vi_jsonl).items()}
    todo_turns = [t for t in turns if int(t["id"]) not in done_map]
    if not todo_turns:
        emit_progress(65.0, f"Tái sử dụng Pha 5B Gemma 4 ({len(done_map)} câu Việt)")
        return done_map

    model_id = cfg.get("models", {}).get("vi_translate_model", "google/gemma-4-26B-A4B-it")
    emit_progress(35.0, f"Pha 5B: Nạp {model_id} dịch thoại Tiếng Việt...")
    sys_prompt = (
        "Bạn là biên kịch lồng tiếng phim ngắn Tiếng Việt hàng đầu. "
        "Dịch tiếng Trung sang văn nói Tiếng Việt tự nhiên, sắc sảo, không dịch sượng Hán Việt, "
        "độ dài câu khớp thời lượng giây cho phép. Chỉ trả JSON hợp lệ."
    )
    batch_size = 40
    ctx = nullcontext(conn) if conn else managed_vllm_server(model_id, cfg)
    with ctx as (base_url, mid):
        for i in range(0, len(todo_turns), batch_size):
            batch = todo_turns[i : i + batch_size]
            packed = []
            for t in batch:
                tid = int(t["id"])
                r = roles_map.get(tid, {})
                dur = round(max(0.4, float(t["end"]) - float(t["start"])), 2)
                packed.append(
                    {
                        "id": tid,
                        "dur_s": dur,
                        "gender": r.get("gender", "MALE"),
                        "emotion": r.get("emotion", "NEUTRAL"),
                        "addressing": r.get("addressing", ""),
                        "zh": t["text"],
                    }
                )
            user_prompt = (
                f"Bảng tên riêng bắt buộc (Glossary): {json.dumps(glossary, ensure_ascii=False)}\n"
                "Dịch từng câu sau sang Tiếng Việt ngắn gọn vừa miệng đọc trong `dur_s` giây, "
                "tuân thủ đúng `addressing` và `emotion`. Trả JSON dạng: "
                '`{"items": [{"id": 1, "vi": "..."}]}`.\n\n'
                f"{json.dumps(packed, ensure_ascii=False)}"
            )
            res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.25)
            rmap = {int(x["id"]): str(x.get("vi", "")).strip() for x in (res.get("items") or []) if "id" in x}
            out_items = []
            for t in batch:
                tid = int(t["id"])
                vi_txt = rmap.get(tid) or t["text"]
                done_map[tid] = vi_txt
                out_items.append({"id": tid, "vi": vi_txt})
            _append_jsonl_batch(vi_jsonl, out_items)
            pct = 36.0 + (len(done_map) / max(1, len(turns))) * 29.0
            emit_progress(pct, f"Pha 5B Gemma 4 (VI): {len(done_map)}/{len(turns)} câu")

    _write_srt(turns, done_map, target_dir / "tracks" / "vi.srt")
    return done_map


def _phase5c_translategemma_foreign(
    target_dir: Path,
    turns: list[dict[str, Any]],
    langs: list[str],
    glossary: dict[str, Any],
    roles_map: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    force: bool = False,
    conn: tuple[str, str] | None = None,
):
    """Pha 5C: Dùng Gemma 4 dịch sạch sang EN / JA / KO / ES."""
    foreign_langs = [x for x in langs if x != "vi"]
    if not foreign_langs:
        return

    need_any = False
    lang_done: dict[str, dict[int, str]] = {}
    for lang in foreign_langs:
        jp = target_dir / f"05c_sub_{lang}.jsonl"
        if force:
            jp.unlink(missing_ok=True)
        dmap = {k: v["text"] for k, v in _load_jsonl_map(jp).items()}
        lang_done[lang] = dmap
        if len(dmap) < len(turns):
            need_any = True

    if not need_any:
        for lang in foreign_langs:
            _write_srt(turns, lang_done[lang], target_dir / "tracks" / f"{lang}.srt")
        emit_progress(98.0, f"Tái sử dụng Pha 5C ({','.join(foreign_langs)})")
        return

    model_id = cfg.get("models", {}).get("foreign_translate_model", "google/gemma-4-26B-A4B-it")
    emit_progress(67.0, f"Pha 5C: Nạp {model_id} dịch {', '.join(foreign_langs).upper()}...")

    batch_size = 40
    ctx = nullcontext(conn) if conn else managed_vllm_server(model_id, cfg)
    with ctx as (base_url, mid):
        for l_idx, lang in enumerate(foreign_langs):
            lname = LANG_NAMES.get(lang, lang)
            jp = target_dir / f"05c_sub_{lang}.jsonl"
            dmap = lang_done[lang]
            todo = [t for t in turns if int(t["id"]) not in dmap]
            sys_prompt = f"You are a professional dubbing script translator. Translate Chinese dialogue into concise, natural spoken {lname}. Output ONLY valid JSON."
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                packed = [
                    {
                        "id": t["id"],
                        "dur_s": round(max(0.4, float(t["end"]) - float(t["start"])), 2),
                        "emotion": roles_map.get(int(t["id"]), {}).get("emotion", "NEUTRAL"),
                        "zh": t["text"],
                    }
                    for t in batch
                ]
                user_prompt = (
                    f"Translate each Chinese line to natural spoken {lname} matching `dur_s` duration. "
                    f'Return JSON: {{"items": [{{"id": 1, "text": "..."}}]}}.\n'
                    f"Glossary reference: {json.dumps(glossary, ensure_ascii=False)}\n"
                    f"Lines: {json.dumps(packed, ensure_ascii=False)}"
                )
                res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.15)
                rmap = {int(x["id"]): str(x.get("text", "")).strip() for x in (res.get("items") or []) if "id" in x}
                out_items = []
                for t in batch:
                    tid = int(t["id"])
                    txt = rmap.get(tid) or t["text"]
                    dmap[tid] = txt
                    out_items.append({"id": tid, "text": txt})
                _append_jsonl_batch(jp, out_items)
                prog = 68.0 + ((l_idx + len(dmap) / max(1, len(turns))) / len(foreign_langs)) * 30.0
                emit_progress(prog, f"Pha 5C ({lang.upper()}): {len(dmap)}/{len(turns)} câu")

            _write_srt(turns, dmap, target_dir / "tracks" / f"{lang}.srt")


def run_step5(target_dir: Path, langs: list[str], glossary_in: Path | None = None, force: bool = False) -> dict[str, Any]:
    master = read_json_safe(target_dir / "03_aligned_master.json")
    ac_data = read_json_safe(target_dir / "04_f0_speakers.json") or {}
    if not master or not master.get("turns"):
        raise FileNotFoundError("Thiếu 03_aligned_master.json cho Bước 5.")

    t0 = time.perf_counter()
    cfg = load_pipeline_config()
    turns = master["turns"]
    acoustics_map = {int(a["id"]): a for a in ac_data.get("acoustics", [])}
    base_glossary = read_json_safe(glossary_in) if glossary_in else {}
    base_glossary = base_glossary or {}

    glossary, roles_map = _phase5a_qwen_analyze(target_dir, turns, acoustics_map, cfg, base_glossary, force=force)

    models_cfg = cfg.get("models", {})
    vi_model = models_cfg.get("vi_translate_model", "google/gemma-4-26B-A4B-it")
    fg_model = models_cfg.get("foreign_translate_model", "google/gemma-4-26B-A4B-it")
    need_vi = "vi" in langs
    need_fg = any(x != "vi" for x in langs)

    if need_vi and need_fg and vi_model == fg_model:
        # Tái sử dụng 1 lần nạp Gemma 4 cho cả VI và EN/JA/KO/ES (tiết kiệm ~50s reload)
        emit_progress(35.0, f"Nạp {vi_model} dùng chung Pha 5B + 5C...")
        with managed_vllm_server(vi_model, cfg) as conn:
            _phase5b_gemma4_vi(target_dir, turns, glossary, roles_map, cfg, force=force, conn=conn)
            _phase5c_translategemma_foreign(target_dir, turns, langs, glossary, roles_map, cfg, force=force, conn=conn)
    else:
        if need_vi:
            _phase5b_gemma4_vi(target_dir, turns, glossary, roles_map, cfg, force=force)
        _phase5c_translategemma_foreign(target_dir, turns, langs, glossary, roles_map, cfg, force=force)

    elapsed = round(time.perf_counter() - t0, 2)
    meta = {"elapsed_s": elapsed, "langs": langs, "turns": len(turns), "glossary_entries": len(glossary)}
    atomic_write_json(target_dir / "05_translate_meta.json", meta)
    emit_progress(100.0, f"Xong Bước 5 ({','.join(langs).upper()}, {len(turns)} câu, {elapsed}s)")
    print(f"[S5 DONE] {','.join(langs)} trong {elapsed}s", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Bước 5: Qwen phân vai tiếng Trung + Gemma 4 dịch Việt + TranslateGemma dịch ngoại ngữ.")
    parser.add_argument("target_dir", type=Path, help="Thư mục work mục tiêu")
    parser.add_argument("--langs", default="vi,en,ja,ko,es", help="Danh sách ngôn ngữ cách nhau dấu phẩy")
    parser.add_argument("--glossary", type=Path, default=None, help="File glossary.json kế thừa từ Phần 1")
    parser.add_argument("--force", action="store_true", help="Dịch lại từ đầu")
    args = parser.parse_args()
    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    run_step5(args.target_dir.resolve(), langs=langs, glossary_in=args.glossary, force=args.force)


if __name__ == "__main__":
    main()

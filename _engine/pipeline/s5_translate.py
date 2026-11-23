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
import ast
import html
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


def _safe_int(val: Any) -> int | None:
    """Chuyển `id` từ LLM/JSON sang int an toàn; trả None nếu là chuỗi non-numeric như 'SPK_UNK'."""
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        m = re.fullmatch(r"#?(-?\d+)", s)
        if m:
            return int(m.group(1))
    return None


@contextmanager
def _managed_vllm_with_fallback(primary_model: str, fallback_models: list[str], cfg: dict[str, Any], phase_label: str):
    """Khởi chạy vLLM theo danh sách ưu tiên, tự lùi sang model dự phòng nếu EngineCore lỗi."""
    cands = list(dict.fromkeys([m for m in [primary_model, *fallback_models] if m]))
    vllm_ctx = None
    last_exc = None
    for idx, cand_m in enumerate(cands):
        try:
            emit_progress(6.0 if "5A" in phase_label else 35.0, f"{phase_label}: Khởi chạy {cand_m}...")
            ctx = managed_vllm_server(cand_m, cfg)
            conn_pair = ctx.__enter__()
            vllm_ctx = (ctx, conn_pair)
            break
        except Exception as exc:
            last_exc = exc
            if idx + 1 < len(cands):
                print(f"[{phase_label} WARN] {cand_m} lỗi -> thử model dự phòng ({exc})", flush=True)
    if vllm_ctx is None:
        raise RuntimeError(f"Tất cả model {phase_label} đều lỗi: {last_exc}")
    ctx_mgr, conn_pair = vllm_ctx
    try:
        yield conn_pair
    finally:
        ctx_mgr.__exit__(None, None, None)


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


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
        "--language-model-only",
        "--enforce-eager",
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
        # Chờ vLLM tải weights (lần đầu ~17GB mất vài phút) & nạp VRAM (tối đa 900s)
        deadline = time.time() + 900.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_fh.flush()
                full_log = log_path.read_text(encoding="utf-8", errors="replace")
                # In các dòng lỗi gốc EngineCore trước rồi nâng RuntimeError rõ ràng
                err_lines = [ln for ln in full_log.splitlines() if any(k in ln for k in ("ERROR", "Exception", "ValueError", "RuntimeError", "OutOfMemory", "not supported", "AssertionError"))]
                root_msg = "\n".join(err_lines[-12:]) if err_lines else full_log[-2500:]
                print(f"[S5 vLLM LOG]\n{full_log[-3500:]}", flush=True)
                raise RuntimeError(f"vLLM khởi chạy thất bại cho {model_id}:\n{root_msg}")
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


def _try_decode_json_candidate(cand: str) -> Any | None:
    """Thử giải mã một ứng viên JSON bằng các bước nới lỏng an toàn."""
    variants = [
        cand,
        re.sub(r",\s*([}\]])", r"\1", cand),
    ]
    quoted = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', variants[1])
    if quoted != variants[1]:
        variants.append(quoted)

    for v in variants:
        try:
            obj = json.loads(v)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(v):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(v[idx:])
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                continue
        try:
            obj = ast.literal_eval(v)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
    return None


def _parse_json_lenient(raw: str) -> Any:
    txt = (raw or "").strip()
    # Loại bỏ thẻ suy nghĩ đầy đủ hoặc bị cắt cụt (Qwen3.5/3.8, Gemma 4)
    txt = re.sub(r"<think>[\s\S]*?</think>", "", txt, flags=re.IGNORECASE).strip()
    txt = re.sub(r"<thought>[\s\S]*?</thought>", "", txt, flags=re.IGNORECASE).strip()
    for closing in ("</think>", "</thought>"):
        low = txt.lower()
        pos = low.rfind(closing)
        if pos != -1:
            txt = txt[pos + len(closing) :].strip()

    candidates: list[str] = []
    if "```" in txt:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", txt, flags=re.IGNORECASE)
        candidates.extend(reversed([b.strip() for b in blocks if b.strip()]))
    candidates.append(txt)

    for cand in candidates:
        parsed = _try_decode_json_candidate(cand)
        if parsed is not None:
            return parsed

    snippet = repr(txt[:240])
    raise ValueError(f"Không parse được JSON từ output LLM: {snippet}")


def _chat_json(base_url: str, model_id: str, system_prompt: str, user_prompt: str, temp: float = 0.15) -> Any:
    last_err = None
    use_json_mode = True
    use_no_think = True
    max_tokens: int | None = 2400
    for attempt in range(5):
        payload: dict[str, Any] = {
            "model": model_id,
            "temperature": temp,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if use_no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = _http_json(f"{base_url}/chat/completions", payload, timeout=240)
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return _parse_json_lenient(content)
        except Exception as exc:
            last_err = exc
            err_s = str(exc).lower()
            if "http 400" in err_s or "http 422" in err_s:
                if max_tokens is not None and ("max_tokens" in err_s or "max_model_len" in err_s or "context length" in err_s):
                    # Bỏ max_tokens để vLLM tự lấy phần còn lại của max_model_len
                    max_tokens = None
                    continue
                if use_no_think and ("chat_template_kwargs" in err_s or "extra_forbidden" in err_s):
                    use_no_think = False
                    continue
                if use_json_mode and ("response_format" in err_s or "json_object" in err_s):
                    use_json_mode = False
                    continue
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gọi LLM thất bại sau 5 lần ({model_id}): {last_err}")


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

    model_id = cfg.get("models", {}).get("zh_analyst_model", "mattbucci/Qwen3.8-27B-AWQ")
    fallback_models = [
        cfg.get("models", {}).get("zh_analyst_model", "philbert440/Qwen3.8-27B-W4A16-AWQ"),
        "philbert440/Qwen3.8-27B-W4A16-AWQ",
        cfg.get("models", {}).get("vi_translate_model", "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"),
    ]
    emit_progress(5.0, f"Pha 5A: Nạp {model_id} phân tích tiếng Trung & phân vai...")

    sys_prompt = (
        "Bạn là biên kịch kiêm đạo diễn lồng tiếng chuyên phim ngắn Trung Quốc. "
        "Trả về DUY NHẤT JSON hợp lệ, không giải thích thêm."
    )
    batch_size = 15
    with _managed_vllm_with_fallback(model_id, fallback_models, cfg, "Pha 5A") as (base_url, mid):
        for i in range(0, len(todo_turns), batch_size):
            batch = todo_turns[i : i + batch_size]
            packed = []
            for t in batch:
                ac = acoustics_map.get(int(t["id"]), {})
                packed.append(
                    {
                        "id": int(t["id"]),
                        "spk_id": t.get("spk_id", "SPK_UNK"),
                        "f0_hz": ac.get("f0_hz"),
                        "gender_hint": ac.get("gender_hint", "UNKNOWN"),
                        "zh": t["text"],
                    }
                )

            user_prompt = (
                "Đọc LIỀN MẠCH cả đoạn đối thoại tiếng Trung dưới đây để phân vai từng câu.\n"
                "Lưu ý:\n"
                "- `f0_hz` chỉ là tần số tham khảo (nữ nói trầm hoặc ồn có thể tụt xuống 155-175Hz; nam hét có thể lên >190Hz), "
                "NÊN BẮT BUỘC ưu tiên mạch đối thoại & nội dung chữ (vd thấy '小帅哥' = nữ gọi nam trẻ; 3-4 câu liền nhau cùng ý = CÙNG 1 NGƯỜI).\n"
                "- Xác định bối cảnh thời đại `era`: `'MODERN'` (phim hiện đại: tiền mặt/vạn tệ/công ty/xe hơi/quyền anh) hoặc `'HISTORICAL'` (cổ trang/tiên hiệp).\n"
                "- Với phim `'MODERN'`: TUYỆT ĐỐI KHÔNG dùng đại từ cổ trang `'ta'`, `'ngươi'`, `'bổn tọa'`, `'tại hạ'`. "
                "Hãy dùng xưng hô đời thường nhất quán suốt cả cảnh (vd Nữ chủ & khách: `'tôi'`/`'chị'` -> `'cậu'`; côn đồ tức giận: `'tao'` -> `'mày'`; đàn anh -> `'anh'`/`'cậu'`).\n\n"
                "Xuất JSON gồm:\n"
                "1) `era`: `'MODERN'` | `'HISTORICAL'`\n"
                "2) `glossary_add`: CHỈ chứa TÊN RIÊNG thực sự (họ tên nhân vật, tên công ty, địa danh) -> dict {tên_Trung: tên_Hán_Việt}. KHÔNG cho danh từ/động từ thường.\n"
                "3) `roles`: list cho từng câu theo ĐÚNG số nguyên `id` với các trường:\n"
                "   - `id`: số nguyên\n"
                "   - `gender`: `'MALE'` | `'FEMALE'`\n"
                "   - `speaker_role`: vai ngắn gọn (vd `'Nữ chủ sòng'`, `'Nam chính'`, `'Đám đông'`)\n"
                "   - `self_pronoun`: đại từ NGƯỜI NÓI tự xưng bằng Tiếng Việt (vd `'tôi'`, `'chị'`, `'anh'`, `'tao'`; CẤM `'anh'` nếu `gender='FEMALE'`!)\n"
                "   - `other_pronoun`: đại từ NGƯỜI NÓI dùng gọi NGƯỜI NGHE (vd `'cậu'`, `'mày'`, `'em'`, `'ông'`)\n"
                "   - `age_tone`: `'YOUNG'` | `'MATURE'`\n"
                "   - `emotion`: `'NEUTRAL'` | `'ANGRY'` | `'HAPPY'` | `'SAD'` | `'WHISPER'` | `'URGENT'`\n\n"
                f"Glossary hiện có: {json.dumps(glossary, ensure_ascii=False)}\n"
                f"Danh sách câu:\n{json.dumps(packed, ensure_ascii=False)}"
            )
            res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.1)
            if isinstance(res, list):
                res = {"era": "MODERN", "glossary_add": {}, "roles": res}
            elif not isinstance(res, dict):
                res = {}

            era = str(res.get("era") or "MODERN").upper()
            if era not in ("MODERN", "HISTORICAL"):
                era = "MODERN"

            raw_gloss = res.get("glossary_add")
            if isinstance(raw_gloss, dict):
                for k, v in raw_gloss.items():
                    if k and isinstance(v, str) and v.strip() and k not in glossary:
                        glossary[str(k)] = v.strip()
            atomic_write_json(glossary_path, glossary)

            raw_roles = res.get("roles") or res.get("items") or []
            if isinstance(raw_roles, dict):
                raw_roles = [
                    ({"id": rk, **rv} if isinstance(rv, dict) else {"id": rk})
                    for rk, rv in raw_roles.items()
                ]
            res_map: dict[int, dict[str, Any]] = {}
            spk_map: dict[str, dict[str, Any]] = {}
            for r in raw_roles:
                if not isinstance(r, dict):
                    continue
                rid = _safe_int(r.get("id"))
                if rid is not None:
                    res_map[rid] = r
                else:
                    sk = str(r.get("spk_id") or r.get("id") or "").strip()
                    if sk:
                        spk_map[sk] = r

            role_items = []
            for t in batch:
                tid = int(t["id"])
                ac = acoustics_map.get(tid, {})
                r = res_map.get(tid) or spk_map.get(str(t.get("spk_id") or "")) or {}
                g = str(r.get("gender") or ac.get("gender_hint") or "MALE").upper()
                if g not in ("MALE", "FEMALE"):
                    g = "FEMALE" if ac.get("gender_hint") == "FEMALE" else "MALE"
                age_t = str(r.get("age_tone") or "YOUNG").upper()
                if age_t not in ("YOUNG", "MATURE"):
                    age_t = "YOUNG"
                emo = str(r.get("emotion") or "NEUTRAL").upper()
                if emo not in ("NEUTRAL", "ANGRY", "HAPPY", "SAD", "WHISPER", "URGENT"):
                    emo = "NEUTRAL"

                self_p = str(r.get("self_pronoun") or "").strip()
                other_p = str(r.get("other_pronoun") or "").strip()
                # Chặn cứng lỗi nữ tự xưng 'anh' hoặc phim hiện đại xưng cổ trang 'ta/ngươi'
                if g == "FEMALE" and self_p.lower() in ("anh", "chú", "ông", "tao_male"):
                    self_p = "chị" if age_t == "YOUNG" else "tôi"
                if g == "MALE" and self_p.lower() in ("chị", "cô", "dì", "bà"):
                    self_p = "tôi"
                if era == "MODERN":
                    if self_p.lower() in ("ta", "tại hạ", "bổn tọa", "bổn cô nương", ""):
                        self_p = "tôi"
                    if other_p.lower() in ("ngươi", "các hạ", "tiểu tử", ""):
                        other_p = "cậu" if g == "FEMALE" else "mày/cậu"
                self_p = self_p or "tôi"
                other_p = other_p or "cậu"
                spk_role = str(r.get("speaker_role") or ("Nhân vật nữ" if g == "FEMALE" else "Nhân vật nam")).strip()
                addr = f"Người nói ({spk_role}, {g}) tự xưng '{self_p}', gọi người nghe là '{other_p}'"

                item = {
                    "id": tid,
                    "era": era,
                    "gender": g,
                    "age_tone": age_t,
                    "emotion": emo,
                    "speaker_role": spk_role,
                    "self_pronoun": self_p,
                    "other_pronoun": other_p,
                    "addressing": addr,
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
        "Dịch tiếng Trung sang khẩu ngữ Tiếng Việt HIỆN ĐẠI tự nhiên, đời thường, sắc sảo; "
        "TUYỆT ĐỐI KHÔNG dùng đại từ cổ trang ('ta', 'ngươi', 'tại hạ', 'các hạ') khi bối cảnh là MODERN (hiện đại). "
        "BẮT BUỘC tuân thủ đúng chiều xưng hô: người nói tự xưng theo `self_pronoun` (TUYỆT ĐỐI KHÔNG để nhân vật nữ tự xưng 'anh') và gọi người nghe theo `other_pronoun`, giữ nhất quán suốt cảnh đối thoại. "
        "TUYỆT ĐỐI KHÔNG dịch sượng Hán-Việt các từ thông thường (dùng 'thằng Tây', 'cậu em', 'đánh quyền ngầm', 'tiền mặt', 'nộp mạng', 'gấp chăn'). "
        "Độ dài câu khớp thời lượng `dur_s` giây. Chỉ trả JSON hợp lệ."
    )
    batch_size = 15
    zh_fb = cfg.get("models", {}).get("zh_analyst_model", "philbert440/Qwen3.8-27B-W4A16-AWQ")
    ctx = nullcontext(conn) if conn else _managed_vllm_with_fallback(model_id, [zh_fb], cfg, "Pha 5B")
    with ctx as (base_url, mid):
        for i in range(0, len(todo_turns), batch_size):
            batch = todo_turns[i : i + batch_size]
            prev_ctx = [
                {"id": int(pt["id"]), "speaker": roles_map.get(int(pt["id"]), {}).get("speaker_role", ""), "vi": done_map[int(pt["id"])]}
                for pt in turns
                if int(pt["id"]) < int(batch[0]["id"]) and int(pt["id"]) in done_map
            ][-4:]
            packed = []
            for t in batch:
                tid = int(t["id"])
                r = roles_map.get(tid, {})
                dur = round(max(0.4, float(t["end"]) - float(t["start"])), 2)
                packed.append(
                    {
                        "id": tid,
                        "dur_s": dur,
                        "era": r.get("era", "MODERN"),
                        "speaker": r.get("speaker_role", ""),
                        "gender": r.get("gender", "MALE"),
                        "self_pronoun": r.get("self_pronoun", "tôi"),
                        "other_pronoun": r.get("other_pronoun", "cậu"),
                        "emotion": r.get("emotion", "NEUTRAL"),
                        "zh": t["text"],
                    }
                )
            user_prompt = (
                f"Bảng tên nhân vật/địa danh (chỉ áp dụng đúng TÊN RIÊNG): {json.dumps(glossary, ensure_ascii=False)}\n"
                f"Ngữ cảnh 4 câu vừa dịch trước đó (để giữ mạch xưng hô liền lạc): {json.dumps(prev_ctx, ensure_ascii=False)}\n"
                "Dịch từng câu sau sang Tiếng Việt hiện đại ngắn gọn vừa miệng đọc trong `dur_s` giây, "
                "dùng ĐÚNG `self_pronoun` (người nói tự xưng) và `other_pronoun` (gọi đối phương). Trả JSON dạng: "
                '`{"items": [{"id": 1, "vi": "..."}]}`.\n\n'
                f"{json.dumps(packed, ensure_ascii=False)}"
            )
            res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.25)
            raw_items = res if isinstance(res, list) else (res.get("items") or res.get("roles") or [])
            if isinstance(raw_items, dict):
                raw_items = [{"id": k, "vi": v} if not isinstance(v, dict) else {"id": k, **v} for k, v in raw_items.items()]
            rmap: dict[int, str] = {}
            for x in raw_items:
                if not isinstance(x, dict):
                    continue
                xid = _safe_int(x.get("id"))
                if xid is not None:
                    rmap[xid] = str(x.get("vi") or x.get("text") or x.get("translation") or "").strip()
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


def _build_qa_html_and_backtrans(
    target_dir: Path,
    turns: list[dict[str, Any]],
    vi_map: dict[int, str],
    lang_done: dict[str, dict[int, str]],
    roles_map: dict[int, dict[str, Any]],
    base_url: str,
    mid: str,
) -> Path:
    """Dịch ngược nhanh các bản ngoại ngữ (EN/JA/KO/ES) sang Tiếng Việt để người dùng tự kiểm tra 100% độ chuẩn."""
    foreign_langs = [lg for lg in lang_done.keys() if lang_done.get(lg)]
    back_vi: dict[str, dict[int, str]] = {lg: {} for lg in foreign_langs}
    sys_qa = "You are a bilingual quality reviewer. Back-translate each foreign subtitle line faithfully into concise Vietnamese so an editor can verify its meaning. Return ONLY valid JSON."

    for lg in foreign_langs:
        dmap = lang_done[lg]
        for i in range(0, len(turns), 20):
            chunk = turns[i : i + 20]
            packed = [{"id": int(t["id"]), "txt": dmap.get(int(t["id"]), "")} for t in chunk if dmap.get(int(t["id"]))]
            if not packed:
                continue
            try:
                res = _chat_json(
                    base_url,
                    mid,
                    sys_qa,
                    f'Translate these {LANG_NAMES.get(lg, lg)} lines back into concise natural Vietnamese. Return JSON: {{"items": [{{"id": 1, "vi": "..."}}]}}.\n{json.dumps(packed, ensure_ascii=False)}',
                    temp=0.05,
                )
                raw_items = res if isinstance(res, list) else (res.get("items") or [])
                if isinstance(raw_items, dict):
                    raw_items = [{"id": k, "vi": v} if not isinstance(v, dict) else {"id": k, **v} for k, v in raw_items.items()]
                for x in raw_items:
                    if isinstance(x, dict) and (xid := _safe_int(x.get("id"))) is not None:
                        back_vi[lg][xid] = str(x.get("vi") or x.get("text") or "").strip()
            except Exception as exc:
                print(f"[S5 QA WARN] Back-translate {lg} bỏ qua ({exc})", flush=True)

    cols = "".join(f"<th>{lg.upper()} &amp; Dịch ngược VI</th>" for lg in foreign_langs)
    rows_html = []
    for t in turns:
        tid = int(t["id"])
        r = roles_map.get(tid, {})
        ts = f"{format_srt_ts(float(t['start']))[:-4]}→{format_srt_ts(float(t['end']))[:-4]}"
        role_tag = html.escape(f"{r.get('speaker_role','')} ({r.get('self_pronoun','')}/{r.get('other_pronoun','')})")
        fcells = []
        for lg in foreign_langs:
            ftxt = html.escape(lang_done[lg].get(tid, ""))
            btxt = html.escape(back_vi[lg].get(tid, ""))
            fcells.append(f"<td><div>{ftxt}</div><div class='bt'>↳ {btxt}</div></td>")
        rows_html.append(
            f"<tr><td>{tid}</td><td class='ts'>{ts}<div class='role'>{role_tag}</div></td>"
            f"<td>{html.escape(str(t.get('text','')))}</td>"
            f"<td class='vi'>{html.escape(vi_map.get(tid,''))}</td>{''.join(fcells)}</tr>"
        )

    doc = (
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'/>"
        "<title>Bảng đối chiếu dịch 5 ngôn ngữ — MT Reel</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:18px;margin:0}"
        "h2{margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:13px}"
        "th,td{border:1px solid #334155;padding:7px 9px;vertical-align:top;text-align:left}"
        "th{background:#1e293b;position:sticky;top:0}.ts{white-space:nowrap;color:#94a3b8;font-size:12px}"
        ".role{color:#38bdf8;font-size:11px;margin-top:2px}.vi{color:#86efac;font-weight:500}"
        ".bt{color:#fde68a;font-size:12px;margin-top:3px}</style></head><body>"
        f"<h2>Bảng đối chiếu &amp; kiểm tra dịch ngược ({len(turns)} câu)</h2>"
        f"<table><thead><tr><th>#</th><th>Giờ / Vai</th><th>Trung gốc</th><th>Tiếng Việt</th>{cols}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></body></html>"
    )
    qa_path = target_dir / "tracks" / "kiem_tra_dich.html"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(doc, encoding="utf-8")
    return qa_path


def _phase5c_translategemma_foreign(
    target_dir: Path,
    turns: list[dict[str, Any]],
    langs: list[str],
    glossary: dict[str, Any],
    roles_map: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    force: bool = False,
    conn: tuple[str, str] | None = None,
    vi_map: dict[int, str] | None = None,
):
    """Pha 5C: Dùng Gemma 4 dịch sạch sang EN / JA / KO / ES + xuất bảng dịch ngược Tiếng Việt để kiểm tra."""
    foreign_langs = [x for x in langs if x != "vi"]
    if not foreign_langs:
        return

    vi_ref = vi_map or {k: v["vi"] for k, v in _load_jsonl_map(target_dir / "05b_sub_vi.jsonl").items()}
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

    qa_html = target_dir / "tracks" / "kiem_tra_dich.html"
    if not need_any and qa_html.exists():
        for lang in foreign_langs:
            _write_srt(turns, lang_done[lang], target_dir / "tracks" / f"{lang}.srt")
        emit_progress(98.0, f"Tái sử dụng Pha 5C ({','.join(foreign_langs)})")
        return

    model_id = cfg.get("models", {}).get("foreign_translate_model", "google/gemma-4-26B-A4B-it")
    emit_progress(67.0, f"Pha 5C: Nạp {model_id} dịch {', '.join(foreign_langs).upper()}...")

    batch_size = 15
    zh_fb = cfg.get("models", {}).get("zh_analyst_model", "philbert440/Qwen3.8-27B-W4A16-AWQ")
    ctx = nullcontext(conn) if conn else _managed_vllm_with_fallback(model_id, [zh_fb], cfg, "Pha 5C")
    with ctx as (base_url, mid):
        for l_idx, lang in enumerate(foreign_langs):
            lname = LANG_NAMES.get(lang, lang)
            jp = target_dir / f"05c_sub_{lang}.jsonl"
            dmap = lang_done[lang]
            todo = [t for t in turns if int(t["id"]) not in dmap]
            sys_prompt = (
                f"You are a professional dubbing script translator for modern short dramas. "
                f"Translate Chinese dialogue into concise, natural spoken {lname} matching character register (`speaker_role`, `self_pronoun`->`other_pronoun`) and `vi_ref` intent. Output ONLY valid JSON."
            )
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                packed = [
                    {
                        "id": int(t["id"]),
                        "dur_s": round(max(0.4, float(t["end"]) - float(t["start"])), 2),
                        "speaker": roles_map.get(int(t["id"]), {}).get("speaker_role", ""),
                        "emotion": roles_map.get(int(t["id"]), {}).get("emotion", "NEUTRAL"),
                        "vi_ref": vi_ref.get(int(t["id"]), ""),
                        "zh": t["text"],
                    }
                    for t in batch
                ]
                user_prompt = (
                    f"Translate each Chinese line to natural spoken {lname} matching `dur_s` duration and `vi_ref` context/politeness. "
                    f'Return JSON: {{"items": [{{"id": 1, "text": "..."}}]}}.\n'
                    f"Proper-noun Glossary: {json.dumps(glossary, ensure_ascii=False)}\n"
                    f"Lines: {json.dumps(packed, ensure_ascii=False)}"
                )
                res = _chat_json(base_url, mid, sys_prompt, user_prompt, temp=0.15)
                raw_items = res if isinstance(res, list) else (res.get("items") or [])
                if isinstance(raw_items, dict):
                    raw_items = [{"id": k, "text": v} if not isinstance(v, dict) else {"id": k, **v} for k, v in raw_items.items()]
                rmap: dict[int, str] = {}
                for x in raw_items:
                    if not isinstance(x, dict):
                        continue
                    xid = _safe_int(x.get("id"))
                    if xid is not None:
                        rmap[xid] = str(x.get("text") or x.get("translation") or "").strip()
                out_items = []
                for t in batch:
                    tid = int(t["id"])
                    txt = rmap.get(tid) or t["text"]
                    dmap[tid] = txt
                    out_items.append({"id": tid, "text": txt})
                _append_jsonl_batch(jp, out_items)
                prog = 68.0 + ((l_idx + len(dmap) / max(1, len(turns))) / len(foreign_langs)) * 27.0
                emit_progress(prog, f"Pha 5C ({lang.upper()}): {len(dmap)}/{len(turns)} câu")

            _write_srt(turns, dmap, target_dir / "tracks" / f"{lang}.srt")

        emit_progress(96.0, "Đang xuất bảng dịch ngược 5 ngôn ngữ (kiem_tra_dich.html)...")
        _build_qa_html_and_backtrans(target_dir, turns, vi_ref, lang_done, roles_map, base_url, mid)


def run_step5(target_dir: Path, langs: list[str], glossary_in: Path | None = None, force: bool = False) -> dict[str, Any]:
    master = read_json_safe(target_dir / "03_aligned_master.json")
    ac_data = read_json_safe(target_dir / "04_f0_speakers.json") or {}
    if not master or not master.get("turns"):
        raise FileNotFoundError("Thiếu 03_aligned_master.json cho Bước 5.")

    t0 = time.perf_counter()
    cfg = load_pipeline_config()
    turns = master["turns"]
    acoustics_map = {
        k: a
        for a in ac_data.get("acoustics", [])
        if isinstance(a, dict) and (k := _safe_int(a.get("id"))) is not None
    }
    base_glossary = read_json_safe(glossary_in) if glossary_in else {}
    base_glossary = base_glossary or {}

    glossary, roles_map = _phase5a_qwen_analyze(target_dir, turns, acoustics_map, cfg, base_glossary, force=force)

    models_cfg = cfg.get("models", {})
    vi_model = models_cfg.get("vi_translate_model", "google/gemma-4-26B-A4B-it")
    fg_model = models_cfg.get("foreign_translate_model", "google/gemma-4-26B-A4B-it")
    need_vi = "vi" in langs
    need_fg = any(x != "vi" for x in langs)

    if need_vi and need_fg and vi_model == fg_model:
        # Tái sử dụng 1 lần nạp Gemma 4 cho cả VI và EN/JA/KO/ES (tiết kiệm ~50s reload), tự lùi Qwen3.8 nếu lỗi nạp
        zh_fb = models_cfg.get("zh_analyst_model", "philbert440/Qwen3.8-27B-W4A16-AWQ")
        emit_progress(35.0, f"Nạp {vi_model} dùng chung Pha 5B + 5C...")
        with _managed_vllm_with_fallback(vi_model, [zh_fb], cfg, "Pha 5B+5C") as conn:
            vi_done = _phase5b_gemma4_vi(target_dir, turns, glossary, roles_map, cfg, force=force, conn=conn)
            _phase5c_translategemma_foreign(target_dir, turns, langs, glossary, roles_map, cfg, force=force, conn=conn, vi_map=vi_done)
    else:
        vi_done = _phase5b_gemma4_vi(target_dir, turns, glossary, roles_map, cfg, force=force) if need_vi else None
        _phase5c_translategemma_foreign(target_dir, turns, langs, glossary, roles_map, cfg, force=force, vi_map=vi_done)

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

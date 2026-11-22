"""
Step 1d (3-Layer Fast Chunked Context Translation, Persistent Glossary & Role Mapping):
  Layer 1: Extract per-line acoustic pitch (F0 Hz) & RMS energy from 02_vocals_mdx.wav,
           then lock a persistent Hán-Việt character name, terminology & pronoun glossary
           (glossary.json) across ALL episodes (250 lines/slice so 100+ ep movies never drift).
  Layer 2: Translate in fast batches (~45 lines/call) strictly enforcing the locked glossary.
  Layer 3: Polish spoken Vietnamese consistency mapped to the 4 natural voices
           (NAM_TRE, NAM_GIA, NU_TRE, NU_GIA).

Usage:
  python _engine/scripts/step1d_translate_ocr_srt.py <movie_out_dir> [--glossary path/to/glossary.json]
"""
import argparse
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
ROOT_DIR = ENGINE_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step1_merge_and_transcribe import format_ts
from step1c_ocr_hardsub import clean_cjk_key
from step2_dub_and_render import parse_srt

FALLBACK_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-4-neon",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
STOCK_ARCHETYPES = ("NAM_TRE", "NAM_GIA", "NU_TRE", "NU_GIA")


def coalesce_flicker_entries(entries: list[dict], max_gap: float = 0.45) -> list[dict]:
    """Merge identical back-to-back OCR lines split by brief visual flicker."""
    merged: list[dict] = []
    for e in entries:
        if merged:
            prev = merged[-1]
            gap = e["start"] - prev["end"]
            if gap <= max_gap and clean_cjk_key(e["text"]) == clean_cjk_key(prev["text"]):
                prev["end"] = max(prev["end"], e["end"])
                if len(e["text"]) > len(prev["text"]):
                    prev["text"] = e["text"]
                continue
        merged.append({"num": len(merged) + 1, "start": e["start"], "end": e["end"], "text": e["text"]})
    return merged


def profile_vocals_acoustics(work_dir: Path, entries: list[dict]) -> list[dict]:
    """Measure median fundamental pitch (F0 Hz) and RMS energy via streamed WAV seeks (<5MB RAM)."""
    voc_wav = work_dir / "02_vocals_mdx.wav"
    if not voc_wav.exists():
        return [{"f0_hz": None, "rms": 0.0} for _ in entries]

    profiles = []
    with wave.open(str(voc_wav), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        n_total = wf.getnframes()
        hop = int(0.02 * sr)
        win = int(0.045 * sr)
        min_lag, max_lag = int(sr / 360), int(sr / 75)

        for e in entries:
            i0 = max(0, min(n_total, int(e["start"] * sr)))
            i1 = max(i0, min(n_total, int(e["end"] * sr)))
            if i1 - i0 < win:
                profiles.append({"f0_hz": None, "rms": 0.0})
                continue
            wf.setpos(i0)
            raw = wf.readframes(i1 - i0)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch > 1:
                chunk = chunk.reshape(-1, n_ch).mean(axis=1)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            f0s = []
            for k in range(0, len(chunk) - win, hop):
                seg = chunk[k : k + win]
                if float(np.sqrt(np.mean(seg ** 2))) < 0.025:
                    continue
                seg = seg - np.mean(seg)
                ac = np.correlate(seg, seg, mode="full")[win - 1 :]
                if ac[0] <= 0:
                    continue
                pk = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
                if ac[pk] > 0.42 * ac[0]:
                    f0s.append(sr / pk)
            med_f0 = round(float(np.median(f0s)), 1) if f0s else None
            profiles.append({"f0_hz": med_f0, "rms": round(rms, 3)})

    return profiles


def parse_json_lenient(raw: str):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    return json.loads(cleaned)


_PREFERRED_MODEL: str | None = None


def call_gemini_json(client, types_mod, prompt: str, label: str, temperature: float = 0.2):
    """Call fast Gemini models with retry/backoff, dynamic sticky fast-model promotion, and clean fallback."""
    global _PREFERRED_MODEL
    last_err = None
    retry_waits = (10, 20, 40, 60)
    for attempt in range(len(retry_waits) + 1):
        ordered = ([_PREFERRED_MODEL] + [m for m in FALLBACK_MODELS if m != _PREFERRED_MODEL]) if _PREFERRED_MODEL else list(FALLBACK_MODELS)
        for model_name in ordered:
            try:
                t0 = time.time()
                print(f"[GEMINI] {label} -> {model_name}...", flush=True)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types_mod.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=temperature,
                    ),
                )
                data = parse_json_lenient(resp.text)
                _PREFERRED_MODEL = model_name
                print(f"  [OK] {label} in {time.time() - t0:.1f}s ({model_name})", flush=True)
                return data
            except Exception as err:
                last_err = err
                print(f"  [WARN] {model_name} failed ({err}), next fallback...", flush=True)
        if attempt < len(retry_waits):
            wait_s = retry_waits[attempt]
            print(f"  [RETRY] Temporary Gemini outage on {label}; sleeping {wait_s}s before round {attempt + 2}...", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"All Gemini models failed during {label}: {last_err}")


def merge_glossary_locked(base: dict, incoming: dict) -> dict:
    """Merge newly discovered character names, terms, and pairs while preserving locked entries."""
    merged = {
        "story_summary": (base.get("story_summary") or incoming.get("story_summary") or "").strip(),
        "character_names": dict(base.get("character_names") or {}),
        "terminology": dict(base.get("terminology") or {}),
        "relationship_bible": list(base.get("relationship_bible") or []),
    }
    if base.get("story_summary") and incoming.get("story_summary") and incoming["story_summary"] not in base["story_summary"]:
        merged["story_summary"] = f"{base['story_summary']} | {incoming['story_summary']}".strip(" |")

    for k, v in (incoming.get("character_names") or {}).items():
        if k and v and k not in merged["character_names"]:
            merged["character_names"][k] = v.strip()

    for k, v in (incoming.get("terminology") or {}).items():
        if k and v and k not in merged["terminology"]:
            merged["terminology"][k] = v.strip()

    existing_pairs = {p.get("pair", "").strip().lower() for p in merged["relationship_bible"] if isinstance(p, dict)}
    for p in incoming.get("relationship_bible") or []:
        if not isinstance(p, dict):
            continue
        key = p.get("pair", "").strip().lower()
        if key and key not in existing_pairs:
            merged["relationship_bible"].append(p)
            existing_pairs.add(key)

    return merged


def translate_3_layers(
    entries: list[dict],
    work_dir: Path,
    glossary_path: Path | None = None,
    batch_size: int = 45,
) -> tuple[dict, list[dict]]:
    """Run Layer 1 Glossary/Bible -> Layer 2 Chunked Draft -> Layer 3 Chunked Polish."""
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY in .env")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta", timeout=60_000))
    acoustics = profile_vocals_acoustics(work_dir, entries)
    print(f"[ACOUSTICS] Measured F0/energy cues for {len(entries)} lines.", flush=True)

    payload_lines = [
        {
            "id": i + 1,
            "t": f"{e['start']:.2f}-{e['end']:.2f}",
            "dur": round(e["end"] - e["start"], 2),
            "f0": acoustics[i]["f0_hz"],
            "rms": acoustics[i]["rms"],
            "zh": e["text"],
        }
        for i, e in enumerate(entries)
    ]

    # Step A: Load existing locked glossary if present, then extract/extend across slices of 250 lines
    glossary_doc: dict = {"story_summary": "", "character_names": {}, "terminology": {}, "relationship_bible": []}
    if glossary_path and glossary_path.exists():
        glossary_doc = merge_glossary_locked(glossary_doc, json.loads(glossary_path.read_text(encoding="utf-8")))
        print(f"[GLOSSARY] Loaded locked glossary ({len(glossary_doc['character_names'])} names, {len(glossary_doc['relationship_bible'])} pairs) -> {glossary_path.name}", flush=True)

    compact_script = [f"{x['id']} [{x['f0']}Hz]: {x['zh']}" for x in payload_lines]
    slice_size = 250
    for s_idx in range(0, len(compact_script), slice_size):
        chunk = compact_script[s_idx : s_idx + slice_size]
        lbl = f"Layer 1 Glossary ({s_idx + 1}..{s_idx + len(chunk)})"
        glossary_prompt = f"""Đọc đoạn kịch bản phim ngắn bên dưới (kèm cao độ giọng thật `Hz`: ~85-155Hz thường là Nam trầm/trung niên, ~155-190Hz Nam trẻ, ~195-310Hz Nữ).
Nhiệm vụ: Tạo/bổ sung **Glossary cố định** để dịch đồng nhất từ tập 1 đến tập cuối:
1. `character_names`: Map MỌI tên nhân vật tiếng Trung xuất hiện sang tên **Hán-Việt cố định** (ví dụ: `"林凡": "Lâm Phàm"`, `"苏清雪": "Tô Thanh Tuyết"`). KHÔNG thay đổi các tên đã khóa trong Glossary cũ!
2. `terminology`: Map các địa danh, công ty, thế lực, danh hiệu, đơn vị tiền tệ/xưng danh lặp lại sang tiếng Việt cố định (ví dụ: `"地下黑拳": "sới quyền ngầm"`, `"三十万": "ba trăm nghìn tệ"`).
3. `relationship_bible`: Khóa cứng cặp xưng hô 2 chiều cố định (ví dụ: Bà chủ sới ↔ Nam chính: `chị - cậu` / `tôi - chị`, Em gái ↔ Nam chính: `em - anh`). Giữ nguyên các cặp đã có.

Trả về JSON object:
{{"story_summary": "...", "character_names": {{"...": "..."}}, "terminology": {{"...": "..."}}, "relationship_bible": [{{"pair": "...", "a_to_b": "...", "b_to_a": "..."}}]}}

Glossary đã khóa hiện tại (CHỈ thêm mới nếu thiếu, KHÔNG sửa lại mục đã có):
{json.dumps(glossary_doc, ensure_ascii=False)}

Đoạn kịch bản cần quét:
{json.dumps(chunk, ensure_ascii=False)}"""
        inc = call_gemini_json(client, types, glossary_prompt, lbl, temperature=0.15)
        glossary_doc = merge_glossary_locked(glossary_doc, inc)

    bible = glossary_doc.get("relationship_bible", [])
    print(f"[GLOSSARY] Locked {len(glossary_doc['character_names'])} names, {len(glossary_doc['terminology'])} terms, {len(bible)} relationship pairs.", flush=True)

    # Step B: Chunked Draft Translation + 4-Role Tagging
    draft_by_id: dict[int, dict] = {}
    for start_idx in range(0, len(payload_lines), batch_size):
        batch = payload_lines[start_idx : start_idx + batch_size]
        ids_label = f"{batch[0]['id']}..{batch[-1]['id']}"
        draft_prompt = f"""Dựa vào `Glossary & Relationship Bible` đã khóa bên dưới, dịch lô câu {ids_label} (trả về ĐÚNG {len(batch)} phần tử giữ nguyên `id`):
1. `zh`: Sửa lỗi chữ OCR nếu có (ví dụ `爸已经三了` -> `爸已经走了`, `妈也三了` -> `妈也走了`, `一 一个` -> `一个`).
2. `speaker`, `listener`: Vai người nói & người nghe ngắn gọn (dùng đúng tên trong Glossary nếu có tên riêng).
3. `role`: ĐÚNG 1 trong 4 mã:
   - `NAM_TRE` (nam thanh niên / nam chính / cảnh sát trẻ / côn đồ trẻ)
   - `NAM_GIA` (nam trung niên / lớn tuổi / bác sĩ / ông chủ / bố)
   - `NU_TRE` (nữ trẻ / em gái / nữ chính trẻ)
   - `NU_GIA` (nữ trưởng thành / chị / bà chủ sới / y tá / mẹ)
4. `vi`: Khẩu ngữ phim tiếng Việt tự nhiên, **bắt buộc dùng ĐÚNG tên Hán-Việt trong `character_names`, đúng thuật ngữ trong `terminology`, và ĐÚNG ngôi xưng trong `relationship_bible`**, vừa khít thời lượng `dur` (~4–5 âm tiết/giây).
5. `break`: `true` nếu đổi người nói/đổi cảnh, `false` nếu nói liền hơi.

Trả về JSON array: [{{"id": 1, "zh": "...", "speaker": "...", "listener": "...", "role": "NAM_TRE", "vi": "...", "break": true}}]

Glossary & Relationship Bible cố định:
{json.dumps(glossary_doc, ensure_ascii=False)}

Lô câu cần dịch:
{json.dumps(batch, ensure_ascii=False)}"""
        res = call_gemini_json(client, types, draft_prompt, f"Layer 2 Draft ({ids_label})", temperature=0.2)
        items = res if isinstance(res, list) else res.get("lines", [])
        if len(items) != len(batch):
            raise RuntimeError(f"Batch {ids_label} returned {len(items)} items, expected {len(batch)}")
        for it in items:
            draft_by_id[int(it["id"])] = it

    # Step C: Chunked Spoken Vietnamese Polish Pass
    final_items: list[dict] = []
    for start_idx in range(0, len(payload_lines), 60):
        batch_ids = [payload_lines[k]["id"] for k in range(start_idx, min(len(payload_lines), start_idx + 60))]
        sub_draft = [
            {
                "id": i,
                "dur": payload_lines[i - 1]["dur"],
                "speaker": draft_by_id[i].get("speaker", ""),
                "listener": draft_by_id[i].get("listener", ""),
                "role": draft_by_id[i].get("role", "NAM_TRE"),
                "zh": draft_by_id[i].get("zh", payload_lines[i - 1]["zh"]),
                "vi": draft_by_id[i]["vi"],
                "break": bool(draft_by_id[i].get("break", False)),
            }
            for i in batch_ids
        ]
        ids_label = f"{batch_ids[0]}..{batch_ids[-1]}"
        polish_prompt = f"""Bạn là biên tập viên lồng tiếng Việt. Rà soát lô {ids_label} theo đúng `Glossary & Relationship Bible`:
1. Ép đồng nhất 100% **tên nhân vật Hán-Việt (`character_names`)**, **thuật ngữ (`terminology`)**, và **xưng hô giữa `speaker -> listener` (`relationship_bible`)** — tuyệt đối không tập đầu gọi một tên/ngôi xưng, tập sau đổi sang tên/ngôi xưng khác!
2. Sửa văn dịch máy/Hán-Việt cứng thành khẩu ngữ đời thường ("bằng không" -> "không thì", "thả nạn nhân" -> "thả con tin", tiền tệ thống nhất 1 kiểu tự nhiên).
3. Giữ ngắn gọn hợp `dur` để TTS đọc tự nhiên, đủ dấu câu.

Trả về JSON array đúng {len(sub_draft)} phần tử: [{{"id": 1, "vi": "...", "role": "NAM_TRE", "break": true}}]

Glossary & Relationship Bible:
{json.dumps({"character_names": glossary_doc["character_names"], "terminology": glossary_doc["terminology"], "relationship_bible": bible}, ensure_ascii=False)}

Dữ liệu nháp:
{json.dumps(sub_draft, ensure_ascii=False)}"""
        res = call_gemini_json(client, types, polish_prompt, f"Layer 3 Polish ({ids_label})", temperature=0.15)
        polished = res if isinstance(res, list) else res.get("lines", [])
        if len(polished) != len(sub_draft):
            raise RuntimeError(f"Polish {ids_label} returned {len(polished)} items, expected {len(sub_draft)}")
        for p in polished:
            idx = int(p["id"])
            d = draft_by_id[idx]
            role = p.get("role") or d.get("role") or "NAM_TRE"
            if role not in STOCK_ARCHETYPES:
                role = "NAM_TRE"
            final_items.append({
                "id": idx,
                "zh": d.get("zh", entries[idx - 1]["text"]).strip(),
                "speaker": d.get("speaker", ""),
                "listener": d.get("listener", ""),
                "role": role,
                "vi": p["vi"].strip(),
                "break": bool(p.get("break", d.get("break", False))),
            })

    return glossary_doc, final_items


def write_srt(entries: list[dict], text_key: str, out_path: Path):
    with out_path.open("w", encoding="utf-8") as fh:
        for i, e in enumerate(entries, 1):
            fh.write(f"{i}\n")
            fh.write(f"{format_ts(e['start'])} --> {format_ts(e['end'])}\n")
            fh.write(f"{e[text_key].strip()}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Fast 3-Layer Acoustic + Persistent Glossary translation to 4-voice cast.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--cast-config", type=Path, default=ENGINE_DIR / "config" / "vieneu_stock_cast.json")
    parser.add_argument("--glossary", type=Path, default=None, help="Optional shared series glossary JSON path (locks names/pronouns across parts)")
    args = parser.parse_args()

    out_dir = args.movie_out_dir.resolve()
    work_dir = ENGINE_DIR / "work" / out_dir.name
    tracks_dir = work_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    ocr_srt = work_dir / "zh_ocr.srt"
    if not ocr_srt.exists():
        raise FileNotFoundError(f"Missing {ocr_srt}. Run step1c_ocr_hardsub.py first.")

    raw_entries = parse_srt(ocr_srt)
    entries = coalesce_flicker_entries(raw_entries)
    print(f"[INFO] Coalesced {len(raw_entries)} OCR blocks -> {len(entries)} dialogue lines.", flush=True)

    work_glossary = work_dir / "glossary.json"
    glossary_target = args.glossary.resolve() if args.glossary else work_glossary
    bible_doc, items = translate_3_layers(entries, work_dir, glossary_path=glossary_target)
    by_id = {int(it["id"]): it for it in items}

    stock_cfg = json.loads(args.cast_config.read_text(encoding="utf-8"))["roles"]
    cast_lines = {}
    stock_map = {}

    for i, e in enumerate(entries, 1):
        it = by_id[i]
        role = it["role"] if it["role"] in stock_cfg else "NAM_TRE"
        e["zh_clean"] = it["zh"]
        e["vi"] = it["vi"]
        cast_lines[str(i)] = {
            "role": role,
            "speaker": it["speaker"],
            "listener": it["listener"],
            "emotion": "natural dramatic delivery",
            "break": it["break"],
        }
        stock_map[role] = {"archetype": role, "voice": stock_cfg[role]["voice"]}

    zh_clean_srt = work_dir / "zh_clean.srt"
    vi_srt = tracks_dir / "vi.srt"
    cast_json = work_dir / "cast_and_emotions.json"
    stock_map_json = work_dir / "stock_cast_map.json"
    bible_json = work_dir / "relationship_bible.json"

    write_srt(entries, "zh_clean", zh_clean_srt)
    write_srt(entries, "vi", vi_srt)
    cast_json.write_text(json.dumps({"voices": {}, "lines": cast_lines}, ensure_ascii=False, indent=2), encoding="utf-8")
    stock_map_json.write_text(json.dumps(stock_map, ensure_ascii=False, indent=2), encoding="utf-8")
    bible_json.write_text(json.dumps(bible_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    glossary_payload = json.dumps(bible_doc, ensure_ascii=False, indent=2)
    work_glossary.write_text(glossary_payload, encoding="utf-8")
    if glossary_target != work_glossary:
        glossary_target.parent.mkdir(parents=True, exist_ok=True)
        glossary_target.write_text(glossary_payload, encoding="utf-8")

    print(f"[OK] Saved 3-layer translation ({len(entries)} lines) + {work_glossary.name} + tracks/{vi_srt.name}", flush=True)


if __name__ == "__main__":
    main()

"""
Bước 4: Dịch đa ngôn ngữ (vi, en, ja, ko) khóa chặt Hồ Sơ Nhân Vật & Ma Trận Xưng Hô
+ gán `voice_role` (`NAM_TRE`, `NAM_GIA`, `NU_TRE`, `NU_GIA`) vào `role_lines.json`.
Chạy theo từng lô có ngữ cảnh trượt, tự động resume theo checkpoint nếu mạng gián đoạn.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent


def parse_ts(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_prompt(profile_md: str, glossary_json: str, prev_ctx: list[dict], batch: list[dict], part_note: str) -> str:
    ctx_txt = "\n".join(
        f"[{c['id']}] ({c['tag']}, {c['f0_hz']}Hz) ZH: {c['text']} -> VI: {c.get('vi', '')}"
        for c in prev_ctx
    )
    in_items = [
        {
            "id": x["id"],
            "dur_s": round(max(0.2, x["end"] - x["start"]), 2),
            "tag_hint": x["tag"],
            "f0_hz": x["f0_hz"],
            "zh": x["text"],
        }
        for x in batch
    ]
    return f"""You are an expert multilingual drama subtitle translator and dialogue director for Chinese vertical short dramas.
Part context: {part_note}

MANDATORY CHARACTER GLOSSARY & PRONOUN RULES:
{profile_md}

FIXED CHARACTER NAMES JSON:
{glossary_json}

STRICT RULES:
1. Keep exact 1-to-1 line IDs. Never skip, merge, or reorder IDs.
2. Natural, concise spoken lines sized cleanly to fit `dur_s` seconds (TTS dubbing needs natural crisp phrasing, not wordy filler!).
3. Enforce fixed Vietnamese/EN/JA/KO character names and honorifics/pronouns (`anh/em`, `tôi/cô`, `Chủ tịch Trương`, `Trần Vân Đông`, `Trương Vũ Hàm`, `Đình Đình`, etc.) consistently with context!
4. Choose `voice_role` ONLY from: `"NAM_TRE"`, `"NAM_GIA"`, `"NU_TRE"`, `"NU_GIA"`.
   - Use `tag_hint` (`NAM` -> `NAM_TRE`/`NAM_GIA`, `NU` -> `NU_TRE`/`NU_GIA`) combined with character context (e.g. Chairman Zhang -> `NAM_GIA`, Boss Lady/older woman -> `NU_GIA`, Chen Yundong -> `NAM_TRE`, Zhang Yuhan/Tingting/Captain Li -> `NU_TRE`).
5. Return ONLY a JSON array of objects with keys: `id`, `voice_role`, `vi`, `en`, `ja`, `ko`.

RECENT CONTEXT:
{ctx_txt or "(Start of part)"}

LINES TO TRANSLATE:
{json.dumps(in_items, ensure_ascii=False)}
"""


def translate_batch(client: genai.Client, prompt: str, expected_ids: list[int]) -> list[dict]:
    for model_name in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.25,
                    ),
                )
                txt = resp.text.strip()
                # Bảo vệ trường hợp bọc ```json
                txt = re.sub(r"^```json\s*|\s*```$", "", txt)
                data = json.loads(txt)
                if isinstance(data, dict) and "items" in data:
                    data = data["items"]
                by_id = {int(x["id"]): x for x in data if isinstance(x, dict) and "id" in x}
                if all(eid in by_id for eid in expected_ids):
                    return [by_id[eid] for eid in expected_ids]
                print(f"[STEP4] Missing IDs from {model_name} attempt {attempt+1}: got {len(by_id)}/{len(expected_ids)}", flush=True)
            except Exception as e:
                print(f"[STEP4] {model_name} err attempt {attempt+1}: {str(e)[:140]}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed translating batch starting ID {expected_ids[0]}")


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "role_lines.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for lang in ("vi", "en", "ja", "ko"):
        blocks = [f"{r['id']}\n{r['start']} --> {r['end']}\n{r[lang]}\n" for r in records]
        (out_dir / f"{lang}.srt").write_text("\n".join(blocks), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate voice_profile.json to 4 languages + role_lines.json")
    ap.add_argument("voice_profile", type=Path)
    ap.add_argument("profile_md", type=Path)
    ap.add_argument("glossary_json", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--part-note", default="P1")
    ap.add_argument("--batch-size", type=int, default=45)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    vp = json.loads(args.voice_profile.read_text(encoding="utf-8"))
    prof_txt = args.profile_md.read_text(encoding="utf-8")
    glos_txt = args.glossary_json.read_text(encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "translate_ckpt.json"
    done_by_id: dict[int, dict] = {}
    if ckpt_path.exists():
        for rec in json.loads(ckpt_path.read_text(encoding="utf-8")):
            done_by_id[int(rec["id"])] = rec

    records: list[dict] = []
    for i in range(0, len(vp), args.batch_size):
        batch = vp[i : i + args.batch_size]
        if all(int(x["id"]) in done_by_id for x in batch):
            records.extend(done_by_id[int(x["id"])] for x in batch)
            continue

        prev_ctx = records[-8:]
        prompt = build_prompt(prof_txt, glos_txt, prev_ctx, batch, args.part_note)
        translated = translate_batch(client, prompt, [int(x["id"]) for x in batch])

        for src, tr in zip(batch, translated):
            vr = str(tr.get("voice_role", "")).strip()
            if vr not in ("NAM_TRE", "NAM_GIA", "NU_TRE", "NU_GIA"):
                vr = "NAM_TRE" if src["tag"] == "NAM" else "NU_TRE"
            rec = {
                "id": int(src["id"]),
                "start": fmt_ts(src["start"]),
                "end": fmt_ts(src["end"]),
                "zh": src["text"],
                "voice_role": vr,
                "vi": str(tr["vi"]).strip(),
                "en": str(tr["en"]).strip(),
                "ja": str(tr["ja"]).strip(),
                "ko": str(tr["ko"]).strip(),
            }
            done_by_id[rec["id"]] = rec
            records.append(rec)

        ckpt_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[STEP4] {args.part_note}: {len(records)}/{len(vp)} lines done", flush=True)

    write_outputs(records, args.out_dir)
    print(f"[OK] Step 4 completed: {len(records)} lines -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

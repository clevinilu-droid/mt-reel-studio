"""
Step 1e: Translate script + locked glossary into multiple YouTube dubbing/subtitle languages
(en, ja, ko, es, pt, id) with duration-matched natural spoken phrasing.

Reads:
  - _work/zh_clean.srt (fallback: tracks/vi.srt)
  - _work/tracks/vi.srt (for contextual clarity)
  - _work/glossary.json
  - _work/cast_and_emotions.json

Writes:
  - _work/tracks/<lang>.srt (e.g. en.srt, ja.srt, ko.srt, es.srt)

Usage:
  python _engine/scripts/step1e_translate_multilang.py <movie_out_dir> [--langs en,ja,ko,es] [--force]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR.parent
ROOT_DIR = ENGINE_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step1_merge_and_transcribe import format_ts
from step1d_translate_ocr_srt import call_gemini_json
from step2_dub_and_render import parse_srt

LANG_SPECS = {
    "en": {
        "name": "English",
        "style": (
            "Natural American cinematic spoken dialogue (~3-4 words/sec, punchy drama tone, "
            "consistent Romanized character names like Chen Yundong). "
            "Preserve social hierarchy from Vietnamese `vi` & `relationship_bible` via forms of address "
            "(e.g., CEO/Miss Zhang/Boss vs Sis/Bro/Kid/Punk) and polite vs blunt phrasing"
        ),
    },
    "ja": {
        "name": "Japanese",
        "style": (
            "Natural Japanese drama spoken dialogue (~5-7 mora/sec, consistent character names). "
            "STRICT HONORIFICS LOCK from `relationship_bible` + Vietnamese `vi`: "
            "use polite Keigo (`です/ます`, `社長/さん/様`) toward CEO/boss/elders/patients; "
            "use casual/rough Tameguchi (`俺/お前/だ/だろ`) for enemies/thugs/close siblings (`兄貴/妹`). "
            "NEVER flip politeness register mid-relationship between the same two characters"
        ),
    },
    "ko": {
        "name": "Korean",
        "style": (
            "Natural Korean drama spoken dialogue (~5-6 syllables/sec, Hangul transliterated character names). "
            "STRICT SPEECH-LEVEL LOCK from `relationship_bible` + Vietnamese `vi`: "
            "use Jondaetmal (`-요/-습니다`, `대표님/사장님/어르신`) when speaking up to CEO/boss/elders; "
            "use Banmal (`너/나`, `오빠/야`, no `-요`) for siblings/lovers/thugs/looking down on rivals. "
            "NEVER mix banmal and jondaetmal inconsistently for the same pair"
        ),
    },
    "es": {
        "name": "Spanish",
        "style": (
            "Natural neutral Latin-American Spanish dubbing dialogue (~3-4 words/sec; NEVER use Spain `vosotros`). "
            "STRICT PRONOUN LOCK from `relationship_bible` + Vietnamese `vi`: "
            "use `usted` + titles (`Señora/Jefa/Señor`) toward CEO/boss/elders, and `tú` toward siblings/friends/rivals"
        ),
    },
    "pt": {
        "name": "Portuguese",
        "style": "Natural Brazilian Portuguese dubbing dialogue (~3-4 words/sec), matching `você/senhor(a)` hierarchy from `vi`",
    },
    "id": {
        "name": "Indonesian",
        "style": "Natural conversational Indonesian drama dialogue matching `aku/kamu` vs `saya/Anda/Kak/Bu/Pak` from `vi`",
    },
}


def translate_language(
    client,
    types,
    lang: str,
    zh_entries: list[dict],
    vi_by_num: dict[int, str],
    cast_by_idx: dict[int, dict],
    glossary_doc: dict,
    out_srt: Path,
    batch_size: int = 55,
):
    """Translate all subtitle lines into target language in duration-aware batches."""
    spec = LANG_SPECS[lang]
    payload = []
    for i, e in enumerate(zh_entries):
        meta = cast_by_idx.get(i + 1, {})
        payload.append({
            "id": i + 1,
            "dur": round(max(0.4, e["end"] - e["start"]), 2),
            "role": meta.get("role", "NAM_TRE"),
            "speaker": meta.get("speaker", ""),
            "zh": e["text"],
            "vi": vi_by_num.get(i + 1, ""),
        })

    translated: dict[int, str] = {}
    for s_idx in range(0, len(payload), batch_size):
        batch = payload[s_idx : s_idx + batch_size]
        ids_lbl = f"{batch[0]['id']}..{batch[-1]['id']}"
        prompt = f"""Translate drama dialogue batch {ids_lbl} into **{spec['name']}** (`{lang}`) for voiceover dubbing.
Style rule: {spec['style']}.
Strict constraints:
1. Return a JSON array with EXACTLY {len(batch)} items, each containing `{{"id": <int>, "text": "<translated line>"}}`.
2. Keep character names and special terms consistent with this locked series glossary (adapted appropriately to {spec['name']}):
{json.dumps(glossary_doc, ensure_ascii=False)}
3. Keep each line concise so spoken TTS fits naturally inside `dur` seconds without rushing. Match tone/respect from `role`, `speaker`, and `vi`.

Lines:
{json.dumps(batch, ensure_ascii=False)}"""
        res = call_gemini_json(client, types, prompt, f"Translate {lang.upper()} ({ids_lbl})", temperature=0.2)
        items = res if isinstance(res, list) else res.get("lines", [])
        if len(items) != len(batch):
            raise RuntimeError(f"[{lang}] Batch {ids_lbl} returned {len(items)} items, expected {len(batch)}")
        for it in items:
            translated[int(it["id"])] = str(it.get("text", "")).strip()

    with out_srt.open("w", encoding="utf-8") as fh:
        for i, e in enumerate(zh_entries, 1):
            txt = translated.get(i) or vi_by_num.get(i) or e["text"]
            fh.write(f"{i}\n{format_ts(e['start'])} --> {format_ts(e['end'])}\n{txt}\n\n")
    print(f"[OK] Saved {lang} subtitles ({len(zh_entries)} lines) -> {out_srt}", flush=True)


def resolve_work_dir(out_dir: Path) -> Path:
    """Resolve _engine/work/<Series>/<Part> or _engine/work/<slug> cleanly."""
    output_root = (ENGINE_DIR.parent / "output").resolve()
    rel_work = None
    try:
        rel = out_dir.resolve().relative_to(output_root)
        if len(rel.parts) >= 2:
            rel_work = rel
    except ValueError:
        pass

    if rel_work is not None and (ENGINE_DIR / "work" / rel_work).exists():
        work_dir = ENGINE_DIR / "work" / rel_work
    else:
        slug = out_dir.name
        meta_file = out_dir / "render_meta.json"
        if meta_file.exists():
            try:
                slug = json.loads(meta_file.read_text(encoding="utf-8")).get("movie") or slug
            except Exception:
                pass
        elif rel_work is not None:
            slug = str(rel_work)
        elif not (ENGINE_DIR / "work" / slug).exists():
            mp4s = sorted(out_dir.glob("*.mp4"))
            if mp4s:
                slug = mp4s[0].stem
        work_dir = ENGINE_DIR / "work" / slug
    tracks_dir = work_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # Sync light metadata downloaded from cloud (_light_meta/) into work_dir if missing locally
    light_dir = out_dir / "_light_meta"
    if light_dir.is_dir():
        for fname in ("zh_clean.srt", "zh_ocr.srt", "glossary.json", "relationship_bible.json", "cast_and_emotions.json", "render_meta.json"):
            src = light_dir / fname
            dst = work_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        light_tracks = light_dir / "tracks"
        if light_tracks.is_dir():
            for tf in light_tracks.iterdir():
                if tf.is_file() and not (tracks_dir / tf.name).exists():
                    shutil.copy2(tf, tracks_dir / tf.name)
    for fname in ("glossary.json", "relationship_bible.json", "render_meta.json"):
        src = out_dir / fname
        dst = work_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return work_dir


def main():
    parser = argparse.ArgumentParser(description="Translate script to EN/JA/KO/ES etc. using locked glossary.")
    parser.add_argument("movie_out_dir", type=Path, help="Per-movie output directory")
    parser.add_argument("--langs", type=str, default="en,ja,ko,es", help="Comma-separated target language codes (default: en,ja,ko,es)")
    parser.add_argument("--batch-size", type=int, default=55, help="Lines per Gemini batch (default: 55)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target language SRTs")
    args = parser.parse_args()

    for env_cand in (ROOT_DIR / ".env", ENGINE_DIR.parent / ".env", Path.cwd() / ".env"):
        if env_cand.exists():
            load_dotenv(env_cand)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY")

    out_dir = args.movie_out_dir.resolve()
    work_dir = resolve_work_dir(out_dir)
    tracks_dir = work_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    zh_srt = work_dir / "zh_clean.srt"
    vi_srt = tracks_dir / "vi.srt"
    src_srt = zh_srt if zh_srt.exists() else vi_srt
    if not src_srt.exists():
        raise FileNotFoundError(f"Missing zh_clean.srt and tracks/vi.srt in {work_dir}")

    zh_entries = parse_srt(src_srt)
    vi_entries = parse_srt(vi_srt) if vi_srt.exists() else []
    vi_by_num = {i + 1: e["text"] for i, e in enumerate(vi_entries)}

    glossary_path = work_dir / "glossary.json"
    glossary_doc = json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.exists() else {}

    cast_path = work_dir / "cast_and_emotions.json"
    cast_by_idx: dict[int, dict] = {}
    if cast_path.exists():
        raw_cast = json.loads(cast_path.read_text(encoding="utf-8"))
        if isinstance(raw_cast, dict):
            for k, v in (raw_cast.get("lines") or {}).items():
                if isinstance(v, dict):
                    cast_by_idx[int(k)] = v
        elif isinstance(raw_cast, list):
            for row in raw_cast:
                if isinstance(row, dict) and "idx" in row:
                    cast_by_idx[int(row["idx"])] = row

    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip() and x.strip().lower() != "vi"]
    unsupported = [x for x in langs if x not in LANG_SPECS]
    if unsupported:
        raise ValueError(f"Unsupported langs {unsupported}; supported: {list(LANG_SPECS)}")

    pending = []
    for lang in langs:
        dst = tracks_dir / f"{lang}.srt"
        if not args.force and dst.exists() and dst.stat().st_size > 100 and len(parse_srt(dst)) == len(zh_entries):
            print(f"[SKIP] {dst.name} already complete ({len(zh_entries)} lines).", flush=True)
        else:
            pending.append((lang, dst))

    if not pending:
        return

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta", timeout=60_000))

    t0 = time.time()
    for lang, dst in pending:
        t_l = time.time()
        print(f"[TRANSLATE] Generating {lang.upper()} ({LANG_SPECS[lang]['name']}) for {len(zh_entries)} lines...", flush=True)
        translate_language(client, types, lang, zh_entries, vi_by_num, cast_by_idx, glossary_doc, dst, args.batch_size)
        print(f"  -> {lang.upper()} finished in {time.time() - t_l:.1f}s", flush=True)
    print(f"[DONE] All target translations complete in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()

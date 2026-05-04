"""
annotate_diamonds_items.py — Rate each scenario on 8 DIAMONDS-style dimensions using
the official-sounding item lists (1–7 Likert per dimension).

API (OpenAI-compatible): key/base same resolution as annotate_schwartz_values.py
(DIAMONDS_API_KEY / SCHWARTZ_API_KEY / …). Model: ``DIAMONDS_MODEL`` only, default
``claude-sonnet-4-6``.

Usage:
    python scripts/annotate_diamonds_items.py --dry-run
    python scripts/annotate_diamonds_items.py --limit 2 --workers 4
    python scripts/annotate_diamonds_items.py --input data/scenarios/scenarios_diamonds_zh.json \\
        --output data/scenarios/scenarios_diamonds_zh_diamonds_items.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

sys.path.insert(0, str(PROJECT_ROOT / "constructions" / "scenario"))
from scenarios_diamonds_utils import flatten_scenarios, group_scenarios_by_stage  # noqa: E402


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env(ENV_PATH)


def _resolve_api_base() -> str:
    raw = (
        os.getenv("DIAMONDS_API_BASE")
        or os.getenv("SCHWARTZ_API_BASE")
        or os.getenv("TRANSLATE_API_BASE")
        or os.getenv("API_BASE")
        or os.getenv("ANNOTATE_API_BASE")
        or ""
    )
    if not raw:
        raise SystemExit(
            "API base is not set. Set one of DIAMONDS_API_BASE / SCHWARTZ_API_BASE / "
            "TRANSLATE_API_BASE / API_BASE / ANNOTATE_API_BASE in .env."
        )
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


API_KEY = (
    os.getenv("DIAMONDS_API_KEY")
    or os.getenv("SCHWARTZ_API_KEY")
    or os.getenv("TRANSLATE_API_KEY")
    or os.getenv("ANNOTATE_API_KEY")
    or os.getenv("AIHUBMIX_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_BASE = _resolve_api_base()
MODEL = (os.getenv("DIAMONDS_MODEL") or "").strip() or "claude-sonnet-4-6"
MAX_RETRIES = int(os.getenv("DIAMONDS_MAX_RETRIES", "3"))
MAX_TOKENS = int(os.getenv("DIAMONDS_MAX_TOKENS", "4096"))
PARSE_RETRIES = int(os.getenv("DIAMONDS_PARSE_RETRIES", "2"))

# Canonical dimension keys (JSON output must use these exact strings).
DIAMONDS_DIMENSION_KEYS: tuple[str, ...] = (
    "Duty",
    "Intellect",
    "Adversity",
    "Mating",
    "Positivity",
    "Negativity",
    "Deception",
    "Sociality",
)

# Items as provided for the rating rubric (English; holistic 1–7 per dimension).
DIAMONDS_ITEMS: dict[str, tuple[str, ...]] = {
    "Duty": (
        "A job needs to be done.",
        "P is counted on to do something.",
        "Minor details are important.",
        "Task-oriented thinking is called for.",
    ),
    "Intellect": (
        "The situation includes intellectual or cognitive stimuli.",
        "The situation affords an opportunity to demonstrate intellectual capacity.",
        "The situation affords an opportunity to express unusual ideas or points of view.",
        "The situation evokes values regarding lifestyles or politics.",
    ),
    "Adversity": (
        "I am being criticized.",
        "I am being blamed for something.",
        "I am being threatened by someone or something.",
        "I am being dominated or bossed around.",
    ),
    "Mating": (
        "Potential sexual or romantic partners are present.",
        "The situation includes stimuli that could be construed sexually.",
        "Physical attractiveness is relevant.",
        "Members of the other sex are present.",
    ),
    "Positivity": (
        "The situation is enjoyable.",
        "The situation is playful.",
        "The situation is humorous or potentially humorous.",
        "The situation is simple and clear-cut.",
    ),
    "Negativity": (
        "The situation is anxiety-inducing.",
        "The situation could entail stress or trauma.",
        "The situation would make some people tense and upset.",
        "The situation entails frustration.",
    ),
    "Deception": (
        "It is possible to deceive someone.",
        "A person or activity could be undermined or sabotaged.",
        "The situation may cause feelings of hostility.",
        "Someone in this situation might be deceitful.",
    ),
    "Sociality": (
        "Social interaction is possible.",
        "Close personal relationships are present or could develop.",
        "Behavior of others presents a wide range of interpersonal cues.",
        "A reassuring other person is present.",
    ),
}


def _items_rubric_text() -> str:
    lines: list[str] = []
    for key in DIAMONDS_DIMENSION_KEYS:
        lines.append(f"### {key}")
        for it in DIAMONDS_ITEMS[key]:
            lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines).strip()


SYSTEM_PROMPT = f"""You are a research assistant coding scenarios on situational characteristics.

For each scenario, assign ONE integer score from 1 to 7 for EACH of the following eight dimensions.
Use this rubric: the four bullet items under a dimension describe facets of that dimension; your score
should reflect how characteristic the scenario is for that dimension as a whole (not an average of four separate mini-scores unless they clearly diverge—use judgment).

Likert scale:
- 1 = not characteristic at all; the dimension essentially does not apply.
- 4 = moderately characteristic; clearly present to some extent.
- 7 = extremely characteristic; central to the situation.

Dimensions and items:
{_items_rubric_text()}

Output ONLY valid JSON with exactly these keys:
{{
  "dimensions": {{
    "Duty": {{ "score": <int 1-7>, "rationale": "<one short sentence, same language as scenario narrative>" }},
    "Intellect": {{ ... }},
    "Adversity": {{ ... }},
    "Mating": {{ ... }},
    "Positivity": {{ ... }},
    "Negativity": {{ ... }},
    "Deception": {{ ... }},
    "Sociality": {{ ... }}
  }},
  "annotation_note": "<optional brief note; may be empty string>"
}}

Rules:
- Every dimension key must appear exactly once under "dimensions" with "score" and "rationale".
- "score" must be an integer from 1 through 7 inclusive.
- In rationale and annotation_note: do NOT use the raw ASCII double-quote character (") inside strings—it breaks JSON. Use 「」 for Chinese quotes, or rephrase.
- Do not output markdown fences or any text outside the JSON object.
"""

client = OpenAI(base_url=API_BASE, api_key=API_KEY or "missing-key")


def _sanitize_json(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _strip_markdown_fence(content: str) -> str:
    content = content.strip()
    if "```" not in content:
        return content
    idx = content.find("```")
    chunk = content[idx + 3 :].lstrip()
    if chunk.lower().startswith("json"):
        chunk = chunk[4:].lstrip()
    close = chunk.rfind("```")
    if close != -1:
        chunk = chunk[:close]
    return chunk.strip()


def _extract_json_object(s: str) -> str | None:
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        return None
    return s[start : end + 1]


def call_llm(user_payload: str) -> str:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            kwargs: dict = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
                "temperature": 0.2,
            }
            if MAX_TOKENS > 0:
                kwargs["max_tokens"] = MAX_TOKENS
            completion = client.chat.completions.create(**kwargs)
            choice = completion.choices[0]
            raw = choice.message.content or ""
            stripped = _strip_markdown_fence(raw)
            if not stripped.strip():
                reason = getattr(choice, "finish_reason", None)
                raise RuntimeError(
                    f"empty model content (finish_reason={reason!r}, raw_len={len(raw)})"
                )
            return stripped
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(2 + attempt)
            else:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {e}") from last_err
    raise RuntimeError(str(last_err))


def _parse_json(raw: str) -> dict:
    raw = raw or ""
    stripped = _strip_markdown_fence(raw)
    candidates: list[str] = []
    for blob in (stripped, raw):
        if blob.strip():
            candidates.append(_sanitize_json(blob))
        inner = _extract_json_object(blob)
        if inner:
            sanitized = _sanitize_json(inner)
            if sanitized not in candidates:
                candidates.append(sanitized)

    last_err: Exception | None = None
    for c in candidates:
        if not c.strip():
            continue
        try:
            return json.loads(c)
        except json.JSONDecodeError as e:
            last_err = e
    preview = repr(raw[:800]) if raw else "''"
    raise ValueError(
        f"not valid JSON (response length={len(raw)}). First 800 chars: {preview}"
    ) from last_err


def _digest_scenario(scenario: dict, max_chars: int) -> dict:
    te = scenario.get("trigger_event") or {}
    msg = te.get("message_content") or ""
    ar = te.get("action_required")
    if ar is None and "action_required" in scenario:
        ar = scenario["action_required"]
    blob = {
        "id": scenario.get("id"),
        "stage": scenario.get("stage"),
        "name": scenario.get("name"),
        "category": scenario.get("category"),
        "diamonds_dimension": scenario.get("diamonds_dimension"),
        "intensity": scenario.get("intensity"),
        "description_for_agent": scenario.get("description_for_agent"),
        "setting": scenario.get("setting"),
        "context_text": scenario.get("context_text") or "",
        "trigger_sender": te.get("sender"),
        "trigger_message_content": msg,
        "trigger_action_required": ar,
    }
    text = json.dumps(blob, ensure_ascii=False, indent=2)
    while len(text) > max_chars:
        ct = blob["context_text"]
        mc = blob["trigger_message_content"] or ""
        if len(ct) > 400:
            blob["context_text"] = ct[: int(len(ct) * 0.75)] + "\n[... truncated ...]"
        elif len(mc) > 400:
            blob["trigger_message_content"] = mc[: int(len(mc) * 0.75)] + "\n[... truncated ...]"
        else:
            break
        text = json.dumps(blob, ensure_ascii=False, indent=2)
    return blob


def _validate_ratings(obj: dict, sid: str) -> dict:
    if "dimensions" not in obj or not isinstance(obj["dimensions"], dict):
        raise ValueError(f"{sid}: missing key 'dimensions' (object)")
    dim = obj["dimensions"]
    out: dict[str, dict] = {}
    for key in DIAMONDS_DIMENSION_KEYS:
        if key not in dim:
            raise ValueError(f"{sid}: dimensions missing key {key!r}")
        block = dim[key]
        if not isinstance(block, dict):
            raise ValueError(f"{sid}: dimensions[{key!r}] must be an object")
        sc = block.get("score")
        if sc is None or not isinstance(sc, int) or sc < 1 or sc > 7:
            raise ValueError(f"{sid}: dimensions[{key!r}].score must be int 1-7 (got {sc!r})")
        rat = block.get("rationale")
        if rat is None or (isinstance(rat, str) and not str(rat).strip()):
            raise ValueError(f"{sid}: dimensions[{key!r}].rationale must be non-empty string")
        out[key] = {"score": sc, "rationale": str(rat).strip()}
    note = obj.get("annotation_note", "")
    if note is None:
        note = ""
    elif not isinstance(note, str):
        note = str(note)
    return {"dimensions": out, "annotation_note": note.strip()}


def annotate_one(scenario: dict, dry_run: bool, max_chars: int) -> dict:
    sid = scenario.get("id", "?")
    digest = _digest_scenario(scenario, max_chars=max_chars)
    user_text = (
        "Rate the following scenario on all eight dimensions (1-7 Likert each) and output the JSON.\n\n"
        + json.dumps(digest, ensure_ascii=False, indent=2)
    )
    if dry_run:
        print(f"[dry-run] {sid} ({len(user_text)} chars)")
        return copy.deepcopy(scenario)

    repair = ""
    for parse_attempt in range(PARSE_RETRIES + 1):
        raw = call_llm(user_text + repair)
        try:
            ann = _parse_json(raw)
        except ValueError as e:
            if parse_attempt >= PARSE_RETRIES:
                raise RuntimeError(f"{sid}: {e}") from e
            repair = (
                "\n\nYour last reply was not valid JSON. Reply with ONLY one JSON object: "
                'keys "dimensions" (with Duty, Intellect, Adversity, Mating, Positivity, Negativity, '
                'Deception, Sociality — each {{ "score": 1-7 int, "rationale": string }}), '
                'and "annotation_note" (string, can be \"\"). '
                'Do not use ASCII \" inside string values.'
            )
            continue
        try:
            normalized = _validate_ratings(ann, sid)
        except ValueError as e:
            if parse_attempt >= PARSE_RETRIES:
                raise RuntimeError(str(e)) from e
            repair = f"\n\nValidation error: {e}. Fix and output ONLY the JSON object."
            continue

        merged = copy.deepcopy(scenario)
        merged["diamonds_item_likert"] = {
            "scale": "1-7",
            "model": MODEL,
            "dimensions": normalized["dimensions"],
            "annotation_note": normalized["annotation_note"],
        }
        return merged

    raise RuntimeError(f"{sid}: unreachable")


def _ratings_complete(s: dict) -> bool:
    block = s.get("diamonds_item_likert")
    if not isinstance(block, dict):
        return False
    dim = block.get("dimensions")
    if not isinstance(dim, dict):
        return False
    for key in DIAMONDS_DIMENSION_KEYS:
        b = dim.get(key)
        if not isinstance(b, dict):
            return False
        sc = b.get("score")
        if not isinstance(sc, int) or sc < 1 or sc > 7:
            return False
        if not str(b.get("rationale") or "").strip():
            return False
    return True


def build_output(dataset_meta: dict, ordered_flat: list[dict], by_id: dict[str, dict]) -> dict:
    merged_list = [copy.deepcopy(by_id[s["id"]]) for s in ordered_flat]
    return {
        "dataset_meta": dataset_meta,
        "scenarios": group_scenarios_by_stage(merged_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rate each scenario on 8 DIAMONDS item-based dimensions (1-7 Likert) via LLM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite_diamonds_items.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N scenarios (0 = all).")
    parser.add_argument("--workers", type=int, default=int(os.getenv("DIAMONDS_WORKERS", "8")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing diamonds_item_likert in output.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("DIAMONDS_MAX_INPUT_CHARS", "12000")),
        help="Soft cap for scenario digest size sent to the model.",
    )
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print(
            "ERROR: Set DIAMONDS_API_KEY, SCHWARTZ_API_KEY, ANNOTATE_API_KEY, or API_KEY in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_data = json.loads(args.input.read_text(encoding="utf-8"))
    dataset_meta = copy.deepcopy(raw_data.get("dataset_meta", {}))
    ordered_flat = flatten_scenarios(raw_data.get("scenarios", []))
    if args.limit > 0:
        ordered_flat = ordered_flat[: args.limit]

    by_id: dict[str, dict] = {s["id"]: copy.deepcopy(s) for s in ordered_flat}

    if not args.no_resume and args.output.exists():
        prev = json.loads(args.output.read_text(encoding="utf-8"))
        for s in flatten_scenarios(prev.get("scenarios", [])):
            sid = s.get("id")
            if sid in by_id and _ratings_complete(s):
                by_id[sid] = copy.deepcopy(s)

    if args.no_resume:
        for sid in by_id:
            by_id[sid].pop("diamonds_item_likert", None)

    to_run = [s for s in ordered_flat if not _ratings_complete(by_id[s["id"]])]

    marker = "annotate_diamonds_items.py"
    desc = (dataset_meta.get("description") or "").strip()
    if marker not in desc:
        dataset_meta["description"] = (
            desc + f" DIAMONDS item-based 1-7 Likert via {MODEL}; {marker}."
        ).strip()
    dataset_meta["source"] = marker

    write_lock = threading.Lock()

    def persist() -> None:
        payload = json.dumps(
            build_output(dataset_meta, ordered_flat, by_id),
            ensure_ascii=False,
            indent=2,
        )
        args.output.write_text(payload, encoding="utf-8")

    if not to_run:
        print("Nothing to rate (all scenarios already have diamonds_item_likert).")
        if not args.dry_run:
            persist()
        return

    workers = max(1, args.workers)
    print(f"Rating {len(to_run)} scenario(s), model={MODEL}, workers={workers}")

    def task(scn: dict) -> tuple[str, dict]:
        return scn["id"], annotate_one(scn, args.dry_run, args.max_chars)

    if workers == 1:
        it = tqdm(to_run, desc="DIAMONDS") if HAS_TQDM else to_run
        for scn in it:
            sid, updated = task(scn)
            by_id[sid] = updated
            if not args.dry_run:
                persist()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(task, scn): scn["id"] for scn in to_run}
            it = tqdm(as_completed(futs), total=len(futs), desc="DIAMONDS") if HAS_TQDM else as_completed(futs)
            for fut in it:
                sid, updated = fut.result()
                with write_lock:
                    by_id[sid] = updated
                    if not args.dry_run:
                        persist()

    if not args.dry_run:
        print(f"Wrote {args.output}")
    else:
        print("Dry run done (no API calls, no file written).")


if __name__ == "__main__":
    main()

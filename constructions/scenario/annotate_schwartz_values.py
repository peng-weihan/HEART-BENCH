"""
annotate_schwartz_values.py — For each scenario, call an LLM to annotate Schwartz value
tensions as one or more independent 1v1 pairs (no 2v2 / 2v1 / 1v2 composite poles).

Example: pairs [{value_a, value_b, summary_a, summary_b}, ...]; contradiction_label "A vs B; C vs D"

API (OpenAI-compatible): set API_BASE in .env (e.g. ``API_BASE=<YOUR_LLM_GATEWAY>/v1``)
→ POST {API_BASE}/chat/completions. Set ANNOTATE_API_KEY / SCHWARTZ_API_KEY in .env.

Default model: ``SCHWARTZ_MODEL`` only, default ``claude-sonnet-4-6`` (no fallback to ANNOTATE_GT_MODEL / TRANSLATE_MODEL).
Default concurrency: 8 workers (override with SCHWARTZ_WORKERS or --workers).
Default: at most 2 opposition-tension pairs per scenario (SCHWARTZ_MAX_PAIRS).

Usage:
    python scripts/annotate_schwartz_values.py
    python scripts/annotate_schwartz_values.py --input data/scenarios/scenarios_diamonds_zh.json \\
        --output data/scenarios/scenarios_diamonds_zh_schwartz.json
    python scripts/annotate_schwartz_values.py --limit 3 --dry-run
    python scripts/annotate_schwartz_values.py --workers 2
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
        os.getenv("SCHWARTZ_API_BASE")
        or os.getenv("TRANSLATE_API_BASE")
        or os.getenv("API_BASE")
        or os.getenv("ANNOTATE_API_BASE")
        or ""
    )
    if not raw:
        raise SystemExit(
            "API base is not set. Set one of SCHWARTZ_API_BASE / TRANSLATE_API_BASE / "
            "API_BASE / ANNOTATE_API_BASE in .env."
        )
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


API_KEY = (
    os.getenv("SCHWARTZ_API_KEY")
    or os.getenv("TRANSLATE_API_KEY")
    or os.getenv("ANNOTATE_API_KEY")
    or os.getenv("AIHUBMIX_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_BASE = _resolve_api_base()
MODEL = (os.getenv("SCHWARTZ_MODEL") or "").strip() or "claude-sonnet-4-6"
MAX_RETRIES = int(os.getenv("SCHWARTZ_MAX_RETRIES", "3"))
MAX_TOKENS = int(os.getenv("SCHWARTZ_MAX_TOKENS", "4096"))
PARSE_RETRIES = int(os.getenv("SCHWARTZ_PARSE_RETRIES", "2"))
MAX_PAIRS = int(os.getenv("SCHWARTZ_MAX_PAIRS", "2"))
# Optional: set to 2 to reject adjacent values on the circumplex ring (default 1 = only prompt guidance).
MIN_CIRCUMFLEX_STEPS = int(os.getenv("SCHWARTZ_MIN_CIRCUMFLEX_STEPS", "1"))

sys.path.insert(0, str(PROJECT_ROOT / "constructions" / "scenario"))
from scenarios_diamonds_utils import flatten_scenarios, group_scenarios_by_stage  # noqa: E402

SCHWARTZ_BASIC_VALUES = (
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
)

VALUE_ALIASES = {
    "self-direction": "Self-Direction",
    "self direction": "Self-Direction",
    "stimulation": "Stimulation",
    "hedonism": "Hedonism",
    "achievement": "Achievement",
    "power": "Power",
    "security": "Security",
    "conformity": "Conformity",
    "tradition": "Tradition",
    "benevolence": "Benevolence",
    "universalism": "Universalism",
}

# Circumplex order (one direction around the circle; adjacent = compatible, far apart = conflicting).
# Matches Schwartz's standard arrangement: Self-Direction sits between Stimulation and Universalism, etc.
CIRCUMPLEX_ORDER: tuple[str, ...] = (
    "Power",
    "Achievement",
    "Hedonism",
    "Stimulation",
    "Self-Direction",
    "Universalism",
    "Benevolence",
    "Conformity",
    "Tradition",
    "Security",
)
_CIRC_INDEX = {v: i for i, v in enumerate(CIRCUMPLEX_ORDER)}


def _circumplex_min_steps(va: str, vb: str) -> int:
    """Shortest step distance along the 10-value ring (0=same, 1=neighbors, 5=roughly opposite)."""
    i, j = _CIRC_INDEX[va], _CIRC_INDEX[vb]
    d = abs(i - j)
    return min(d, len(CIRCUMPLEX_ORDER) - d)


SYSTEM_PROMPT = f"""You are a social psychologist annotating moral/value dilemmas using Shalom Schwartz's theory of basic values.

Schwartz's 10 BASIC VALUES (use these EXACT English spellings in JSON arrays):
Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism.

Official-style definitions (use when judging which values apply; wording after Schwartz / common reference summaries such as Wikipedia):

Openness to change
- Self-Direction – independent thought and action—choosing, creating, and exploring.
- Stimulation – excitement, novelty and challenge in life.

Self-enhancement
- Hedonism – pleasure or sensuous gratification for oneself.
- Achievement – personal success through demonstrating competence according to social standards.
- Power – social status and prestige, control or dominance over people and resources.

Conservation
- Security – safety, harmony, and stability of society, of relationships, and of self.
- Conformity – restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms.
- Tradition – respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides.

Self-transcendence
- Benevolence – preserving and enhancing the welfare of those with whom one is in frequent personal contact (the 'in-group').
- Universalism – understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.

Higher-level grouping for reasoning (Hedonism counts under Self-enhancement only, not under Openness to change):
- Openness to change: Self-Direction, Stimulation.
- Self-enhancement: Power, Achievement, Hedonism.
- Conservation: Security, Conformity, Tradition.
- Self-transcendence: Benevolence, Universalism.

Two *especially typical* higher-order tensions (meta-axes; use these when they fit—the resulting 1v1 pairs are usually the most *prototypical* value tensions):
- Meta-axis 1 — Openness to change vs Conservation: one value from {{Self-Direction, Stimulation}} opposing one from {{Security, Conformity, Tradition}} (e.g. autonomy or novelty vs safety or norms).
- Meta-axis 2 — Self-enhancement vs Self-transcendence: one value from {{Power, Achievement, Hedonism}} opposing one from {{Universalism, Benevolence}} (e.g. personal success or dominance vs others' welfare or universal principles).
When several pairings are plausible, *prefer* 1v1 pairs that fall on Meta-axis 1 or Meta-axis 2. You may still add other valid 1v1 pairs when the scenario clearly pulls values that are not on those two axes but are well separated on the circumplex (see below).

Circumplex order (fixed; neighbors on this ring are psychologically *compatible* in Schwartz's structure):
1 Power → 2 Achievement → 3 Hedonism → 4 Stimulation → 5 Self-Direction → 6 Universalism → 7 Benevolence → 8 Conformity → 9 Tradition → 10 Security → (back to Power).

Tension guidance for EACH 1v1 pair you output:
- *First*, consider whether the tension matches Meta-axis 1 or Meta-axis 2 above; if yes, those pairs are preferred as the clearest, most canonical oppositions.
- *Additionally*, values *far apart* along this ring (many steps apart; roughly opposite is ~5 steps) support strong tension even when the pair is not on the two meta-axes (e.g. diagonal oppositions on the circle).
- Values *next to each other* on the ring (1 step apart) are usually *not* opposing; do NOT use adjacent pairs as a primary tension unless the scenario text forces a rare edge case—if you must, explain briefly in annotation_note.
- For pairs that are neither on the two meta-axes nor near-opposite on the ring (2–4 steps): use only when the narrative clearly pulls both; prefer meta-axis or larger circumplex separation when multiple candidates exist.

Task:
1. Decompose the scenario into one or more *independent* 1v1 oppositions. Each item is exactly ONE basic value vs ONE other. No 2v2 / 2v1 / 1v2 composite poles.
2. If several distinct tensions exist, output multiple objects in "pairs". If only one clear tension, output a single-element "pairs" array.
3. For each pair: "value_a" and "value_b" are the two endpoints (order: value whose pull you describe in summary_a comes first). Exact spellings from the 10 basic values. "summary_a" and "summary_b": one short sentence each, SAME language as the scenario.
4. "contradiction_label": e.g. "Self-Direction vs Conformity; Benevolence vs Security" — " vs " within a pair, "; " between pairs; salience order.
5. At most {MAX_PAIRS} pairs per scenario (each pair is one opposition tension); if more tensions exist, keep the {MAX_PAIRS} most salient.

Output ONLY valid JSON with exactly these keys:
{{
  "contradiction_label": string,
  "pairs": [
    {{
      "value_a": string,
      "value_b": string,
      "summary_a": string,
      "summary_b": string
    }}
  ],
  "annotation_note": string (optional, brief; may be empty "")
}}

Rules:
- "pairs" must be a non-empty array with at most {MAX_PAIRS} elements; each element must have all four string fields.
- value_a and value_b must each be one of the 10 basic values and must differ. Prefer the two meta-axis tensions when they fit; otherwise follow circumplex-distance guidance; avoid adjacent-on-ring pairs unless truly necessary.
- In summary_a, summary_b, and annotation_note: do NOT use the raw ASCII double-quote character (") inside string values—it breaks JSON. Use 「」 for Chinese quotes, or rephrase without quotes.
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
                "temperature": 0.25,
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


def _normalize_value_name(s: str) -> str | None:
    t = (s or "").strip()
    if t in SCHWARTZ_BASIC_VALUES:
        return t
    key = t.lower().replace("_", " ")
    return VALUE_ALIASES.get(key)


def _validate_annotation(obj: dict, sid: str) -> None:
    if "pairs" not in obj:
        raise ValueError(f"{sid}: missing key 'pairs'")
    pairs = obj["pairs"]
    if not isinstance(pairs, list) or len(pairs) < 1:
        raise ValueError(f"{sid}: pairs must be a non-empty array")
    if len(pairs) > MAX_PAIRS:
        raise ValueError(f"{sid}: at most {MAX_PAIRS} pairs (got {len(pairs)})")
    allowed = set(SCHWARTZ_BASIC_VALUES)
    normalized: list[dict] = []
    for i, p in enumerate(pairs):
        if not isinstance(p, dict):
            raise ValueError(f"{sid}: pairs[{i}] must be an object")
        va, vb = p.get("value_a"), p.get("value_b")
        sa, sb = p.get("summary_a"), p.get("summary_b")
        for key, val in (("value_a", va), ("value_b", vb), ("summary_a", sa), ("summary_b", sb)):
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(f"{sid}: pairs[{i}].{key} must be a non-empty string")
        nva = _normalize_value_name(str(va))
        nvb = _normalize_value_name(str(vb))
        if nva is None or nva not in allowed:
            raise ValueError(f"{sid}: pairs[{i}].value_a invalid: {va!r}")
        if nvb is None or nvb not in allowed:
            raise ValueError(f"{sid}: pairs[{i}].value_b invalid: {vb!r}")
        if nva == nvb:
            raise ValueError(f"{sid}: pairs[{i}] value_a and value_b must differ")
        if MIN_CIRCUMFLEX_STEPS >= 2:
            steps = _circumplex_min_steps(nva, nvb)
            if steps < MIN_CIRCUMFLEX_STEPS:
                raise ValueError(
                    f"{sid}: pairs[{i}] circumplex distance {steps} < "
                    f"SCHWARTZ_MIN_CIRCUMFLEX_STEPS={MIN_CIRCUMFLEX_STEPS} "
                    f"({nva!r} vs {nvb!r}; ring order in system prompt)"
                )
        normalized.append(
            {
                "value_a": nva,
                "value_b": nvb,
                "summary_a": str(sa).strip(),
                "summary_b": str(sb).strip(),
            }
        )
    obj["pairs"] = normalized
    label = (obj.get("contradiction_label") or "").strip()
    if not label:
        obj["contradiction_label"] = "; ".join(f"{p['value_a']} vs {p['value_b']}" for p in normalized)
    else:
        obj["contradiction_label"] = label
    note = obj.get("annotation_note", "")
    if note is None:
        obj["annotation_note"] = ""
    elif not isinstance(note, str):
        obj["annotation_note"] = str(note)


def _digest_scenario(scenario: dict, max_chars: int) -> dict:
    """Compact payload for the model; truncate long narrative."""
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


def annotate_one(scenario: dict, dry_run: bool, max_chars: int) -> dict:
    sid = scenario.get("id", "?")
    digest = _digest_scenario(scenario, max_chars=max_chars)
    user_text = (
        "Analyze the following scenario and output the JSON annotation as specified.\n\n"
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
                "\n\nYour last reply was not valid JSON or failed checks. "
                "Reply with ONLY one JSON object with keys: contradiction_label, pairs (non-empty array of "
                "objects with value_a, value_b, summary_a, summary_b), annotation_note (string, can be \"\"). "
                "Each pair: two different basic values; prefer circumplex-distant oppositions, not adjacent values on the ring. "
                "Do not use ASCII \" inside any string value—use 「」 in Chinese or rephrase so JSON stays valid."
            )
            continue
        try:
            _validate_annotation(ann, sid)
        except ValueError as e:
            if parse_attempt >= PARSE_RETRIES:
                raise RuntimeError(str(e)) from e
            repair = f"\n\nValidation error: {e}. Fix and output ONLY the JSON object."
            continue

        merged = copy.deepcopy(scenario)
        merged["schwartz_value_conflict"] = {
            "contradiction_label": ann["contradiction_label"],
            "pairs": ann["pairs"],
            "annotation_note": ann.get("annotation_note", "") or "",
        }
        return merged


def _annotation_complete(s: dict) -> bool:
    block = s.get("schwartz_value_conflict")
    if not isinstance(block, dict):
        return False
    pairs = block.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 1:
        return False
    p0 = pairs[0]
    if not isinstance(p0, dict):
        return False
    if not (str(p0.get("value_a") or "").strip() and str(p0.get("value_b") or "").strip()):
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
        description="Annotate each scenario with Schwartz value contradiction structure via LLM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite_schwartz.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N scenarios (0 = all).")
    parser.add_argument("--workers", type=int, default=int(os.getenv("SCHWARTZ_WORKERS", "8")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Re-annotate all; ignore existing schwartz block in output.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("SCHWARTZ_MAX_INPUT_CHARS", "12000")),
        help="Soft cap for scenario digest size sent to the model.",
    )
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print(
            "ERROR: Set SCHWARTZ_API_KEY, ANNOTATE_API_KEY, or API_KEY in environment or .env",
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
            if sid in by_id and _annotation_complete(s):
                by_id[sid] = copy.deepcopy(s)

    if args.no_resume:
        for sid in by_id:
            cur = by_id[sid]
            cur.pop("schwartz_value_conflict", None)

    to_run = [s for s in ordered_flat if not _annotation_complete(by_id[s["id"]])]

    marker = "annotate_schwartz_values.py"
    desc = (dataset_meta.get("description") or "").strip()
    if marker not in desc:
        dataset_meta["description"] = (desc + f" Schwartz value-contradiction annotations via {MODEL}; {marker}.").strip()
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
        print("Nothing to annotate (all scenarios already have schwartz_value_conflict.pairs).")
        if not args.dry_run:
            persist()
        return

    workers = max(1, args.workers)
    print(f"Annotating {len(to_run)} scenario(s), model={MODEL}, workers={workers}")

    def task(scn: dict) -> tuple[str, dict]:
        return scn["id"], annotate_one(scn, args.dry_run, args.max_chars)

    if workers == 1:
        it = tqdm(to_run, desc="Schwartz") if HAS_TQDM else to_run
        for scn in it:
            sid, updated = task(scn)
            by_id[sid] = updated
            if not args.dry_run:
                persist()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(task, scn): scn["id"] for scn in to_run}
            it = tqdm(as_completed(futs), total=len(futs), desc="Schwartz") if HAS_TQDM else as_completed(futs)
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

"""
detect_conflicts.py — physical / logical conflict detection.

Flow:
  For every character × scenario pair, take all memories with age <= scenario.age,
  group them by the memory's timeline age, and issue one LLM call per group.
  To keep the server-side prompt cache warm, scheduling is:
    fixed (character, memory-age) → run all scenarios that need that memory
    batch back-to-back, in scenario-id order.
  The user prompt's first half (character info + memory block) stays identical
  across those scenarios and is placed before the scenario block; only the
  second half changes per scenario.
  Different (character, memory-age) groups can run in parallel
  (--scenario-workers / --workers).
  Per-age batch results are merged: if any batch reports a conflict, the
  pair is judged as conflicting overall.

  Optional Claude / compatible-gateway ephemeral prompt cache:
    With ANNOTATE_EPHEMERAL_CACHE=1 the user message is split into multiple
    text parts and `cache_control: {type: ephemeral}` is attached to the
    character+memory block (same shape as test_claude_cache_quwan.py); when
    disabled the messages are plain strings.

  LLM usage / cache log (JSONL, one line per request, on by default):
    Default destination: data/annotations/conflicts/conflicts_llm_usage.jsonl.
    ANNOTATE_CONFLICTS_USAGE_LOG=0 (or false/no/off) disables it; set it to an
    absolute or relative path to redirect the log.
    Each line carries ts_utc, completion_id, model, usage, cache_summary, and
    extra (char/scenario/age etc.).

NOTE: the LLM-facing system / user prompt strings below are intentionally in
Chinese — they steer the LLM that operates on the Chinese narrative dataset
and must keep matching it.

Usage:
    # conflict detection
    python scripts/detect_conflicts.py
    python scripts/detect_conflicts.py --char CHAR_01_N_HIGH
    python scripts/detect_conflicts.py --stage school_age
    python scripts/detect_conflicts.py --workers 4
    python scripts/detect_conflicts.py --dry-run
    python scripts/detect_conflicts.py --limit 10
    python scripts/detect_conflicts.py --no-pair-filter
    python scripts/detect_conflicts.py --pair-filter path/to/policy.json
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from pair_filter import apply_argv_pair_filter

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

# ===== Config =====

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env(path: Path, *, override: bool = False):
    """Load KEY=VALUE lines from path into os.environ.
    If override is True, values from the file replace existing env vars (project .env wins over shell)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and (override or k not in os.environ):
            os.environ[k] = v


_load_env(PROJECT_ROOT / ".env", override=True)

LLM_API_KEY = os.getenv("ANNOTATE_API_KEY")
LLM_API_BASE = os.getenv("ANNOTATE_API_BASE", "").rstrip("/")
DEFAULT_CONFLICTS_MODEL = "gemini-3-flash-preview"
# Conflicts-only model: read ANNOTATE_CONFLICTS_MODEL exclusively, no fallback.
LLM_CONFLICTS_MODEL = os.getenv("ANNOTATE_CONFLICTS_MODEL") or DEFAULT_CONFLICTS_MODEL

# Gemini (LiteLLM / Google path): multipart cache_control + system role -> 400 "system instruction
# should not be set ... when using cached content". Disable ephemeral multipart for these models.
_gemini_model = "gemini" in (LLM_CONFLICTS_MODEL or "").lower()
EPHEMERAL_CACHE = (
    not _gemini_model
    and os.getenv("ANNOTATE_EPHEMERAL_CACHE", "").strip().lower() in ("1", "true", "yes")
)

MAX_RETRIES = 2
DEFAULT_WORKERS = int(os.getenv("ANNOTATE_WORKERS", "12"))
DEFAULT_SCENARIO_WORKERS = int(os.getenv("ANNOTATE_SCENARIO_WORKERS", "12"))

llm_client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

# ===== Data paths =====
SCENARIOS_PATH = PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json"
CHARACTERS_PATH = PROJECT_ROOT / "benchmark" / "characters" / "characters_phase11.json"


def _model_slug(model: str | None) -> str:
    """Convert a model name into a filename-safe slug."""
    if not model:
        return "unknown"
    return re.sub(r"[^\w\-]", "_", model)


OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "annotations" / "conflicts" / "conflicts_detection.json"
CKPT_DIR = PROJECT_ROOT / "benchmark" / "annotations" / "conflicts" / "ckpt"
DEFAULT_CONFLICTS_USAGE_LOG = OUTPUT_PATH.parent / "conflicts_llm_usage.jsonl"


def _resolve_conflicts_usage_log_path() -> Path | None:
    raw = os.getenv("ANNOTATE_CONFLICTS_USAGE_LOG", "").strip()
    if not raw:
        return DEFAULT_CONFLICTS_USAGE_LOG
    low = raw.lower()
    if low in ("0", "false", "no", "off"):
        return None
    if low in ("1", "true", "yes", "on"):
        return DEFAULT_CONFLICTS_USAGE_LOG
    return Path(os.path.expanduser(raw))


CONFLICTS_USAGE_LOG_PATH: Path | None = _resolve_conflicts_usage_log_path()
_usage_log_lock = threading.Lock()

_ckpt_log_lock = threading.Lock()
_logged_ckpt_pairs: set[tuple[str, str]] = set()

_ckpt_rw_guard = threading.Lock()
_ckpt_rw_locks: dict[tuple[str, str], threading.Lock] = {}


def _ckpt_rw_lock(char_id: str, scenario_id: str) -> threading.Lock:
    key = (char_id, scenario_id)
    with _ckpt_rw_guard:
        if key not in _ckpt_rw_locks:
            _ckpt_rw_locks[key] = threading.Lock()
        return _ckpt_rw_locks[key]


def _log_ckpt_resume_once(char_id: str, scenario_id: str, n_done: int, n_total: int) -> None:
    key = (char_id, scenario_id)
    with _ckpt_log_lock:
        if key in _logged_ckpt_pairs:
            return
        _logged_ckpt_pairs.add(key)
    print(f"    [CKPT] {char_id} × {scenario_id}: resuming, {n_done}/{n_total} age-batches done.")

# ===== Helpers =====


def extract_age(timeline: str) -> int | None:
    m = re.search(r"age\s*(\d+)", timeline, re.IGNORECASE)
    return int(m.group(1)) if m else None


def group_memories_by_age(memories: list[dict], max_age: int) -> dict[int, list[dict]]:
    """Return memories with age <= max_age, grouped by their timeline age."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for mem in memories:
        age = extract_age(mem.get("timeline", ""))
        if age is not None and age <= max_age:
            groups[age].append(mem)
    return dict(groups)


def memories_at_timeline_age(memories: list[dict], age: int) -> list[dict]:
    """Memories whose timeline age == `age` (scenario-independent; used to share
    one memory block across many scenarios)."""
    return [m for m in memories if extract_age(m.get("timeline", "")) == age]


# ===== Checkpoint =====


def _ckpt_path(char_id: str, scenario_id: str) -> Path:
    return CKPT_DIR / f"{char_id}__{scenario_id}.json"


def load_ckpt(char_id: str, scenario_id: str) -> dict | None:
    """Load checkpoint as {"completed_ages": [...], "results": [...]}; None if missing."""
    p = _ckpt_path(char_id, scenario_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_ckpt(
    char_id: str,
    scenario_id: str,
    completed_ages: list[int],
    results: list[dict],
    batch_errors: list[dict] | None = None,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    p = _ckpt_path(char_id, scenario_id)
    payload: dict = {"completed_ages": completed_ages, "results": results}
    if batch_errors is not None:
        payload["batch_errors"] = batch_errors
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def delete_ckpt(char_id: str, scenario_id: str) -> None:
    p = _ckpt_path(char_id, scenario_id)
    if p.exists():
        p.unlink()


# ===== LLM =====

_rl_lock = threading.Lock()
_rl_state: dict = {"until": 0.0}

LLM_SYSTEM_PROMPT = """You are a rigorous logical analyst whose job is to detect physical / logical contradictions between life scenarios and personal experiences.

        Your task: given a character's memories from a specific past age range and a hypothetical scenario the character is now facing, decide whether the scenario is physically or logically impossible — i.e. whether **this character could not possibly encounter this scenario at all**.

        ## Dimensions of contradiction

        ### 1. Physical / timeline contradiction
        - The scenario takes place before the character's relevant memory or before the relevant ability could have formed
        - The people / places involved in the scenario no longer exist in the character's life (e.g. already deceased, already split up)
        - The physical conditions required by the scenario contradict the character's current state (e.g. the scenario assumes the character is still in school, but memories show they have been working for years)

        ### 2. Logical / causal contradiction
        - The scenario's preconditions (context) clash with the character's prior irreversible decisions
        - Example: if the character chose A in a particular period, that choice logically rules out scenario B downstream
        - The behaviour the scenario assumes the character will take contradicts that character's clearly-established personality / belief to the point of impossibility
        - Note: "unlikely" or "out of character" alone is NOT a contradiction — it must be a fundamental logical contradiction

        ## Output format

        Output strictly one JSON object (no markdown code block):

        If a contradiction exists, list **each contradicting memory** separately:
        {
        "has_conflict": true,
        "conflicts": [
            {
            "memory_id": "MEM_XX_XXXX",
            "reason": "30-50 words describing the specific contradiction with the scenario."
            },
            {
            "memory_id": "MEM_YY_YYYY",
            "reason": "30-50 words describing the specific contradiction with the scenario."
            }
        ]
        }

        If there is no contradiction:
        {
        "has_conflict": false,
        "conflicts": []
        }

        Note: the conflicts array only contains contradicting memories; each entry independently explains its contradiction."""


def build_char_block(char: dict) -> str:
    bf = char.get("big_five", {})
    return f"""## Character info

- Character ID: {char['id']}
- Name: {char.get('name', '')}
- Archetype: {char.get('archetype', '')}
- Big Five: O={bf.get('openness', 0.5):.2f} C={bf.get('conscientiousness', 0.5):.2f} E={bf.get('extraversion', 0.5):.2f} A={bf.get('agreeableness', 0.5):.2f} N={bf.get('neuroticism', 0.5):.2f}"""


def build_memories_block(char: dict, age: int, memories: list[dict]) -> str:
    mem_lines = [f"## The character's memories and background around age {age} ({len(memories)} entries)\n"]
    for i, mem in enumerate(memories, 1):
        mem_lines.append(f"### [{i}] {mem['id']} ({mem.get('timeline', '')})")
        content = mem.get('content_full')
        if not content:
            raise ValueError(f"Memory {mem.get('id', '?')} missing content_full")
        mem_lines.append(f"- {content}")
        conclusion = mem.get('psych_conclusion')
        if conclusion:
            mem_lines.append(f"- Psych conclusion: {conclusion}")
        policy = mem.get('behavior_policy')
        if policy:
            mem_lines.append(f"- Behaviour tendency: {policy}")
        mem_lines.append("")
    return "\n".join(mem_lines)


def build_scenario_block(scenario: dict) -> str:
    trigger = scenario.get("trigger_event", {})
    return f"""## Current scenario

- Scenario ID: {scenario['id']}
- Stage: {scenario.get('stage', '')} (age {scenario.get('age', '?')})
- DIAMONDS dimension: {scenario.get('diamonds_dimension', '')}
- Description: {scenario.get('description_for_agent', '')}

### Scenario background
{scenario.get('context_text', '')}

### Trigger event
Sender: {trigger.get('sender', '')}
{trigger.get('message_content', '')}
Required action: {trigger.get('action_required', '')}"""


def conflict_closing_instruction(age: int) -> str:
    return (
        f'Based on the character\'s memories and background at age {age}, judge whether the scenario above contains a physical or logical contradiction '
        f'that makes it "impossible" for this character to encounter the scenario. Output JSON.'
    )


def build_user_prompt(scenario: dict, char: dict, age: int, memories: list[dict]) -> str:
    """Character + memory block first, scenario block last — keeps the fixed
    prefix prompt-cacheable across scenarios."""
    char_block = build_char_block(char)
    mem_block = build_memories_block(char, age, memories)
    scenario_block = build_scenario_block(scenario)
    return f"""{char_block}

{mem_block}

---

{scenario_block}

{conflict_closing_instruction(age)}"""


def build_ephemeral_cache_user_blocks(scenario: dict, char: dict, age: int, memories: list[dict]) -> list[dict[str, Any]]:
    """User multi-part content: the memory prefix carries an ephemeral
    cache_control marker; the scenario + closing instruction are the variable
    suffix."""
    char_block = build_char_block(char)
    mem_block = build_memories_block(char, age, memories)
    scenario_block = build_scenario_block(scenario)
    suffix = f"---\n\n{scenario_block}\n\n{conflict_closing_instruction(age)}"
    return [
        {
            "type": "text",
            "text": f"{char_block}\n\n{mem_block}",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": suffix},
    ]


def build_conflict_messages(char: dict, scenario: dict, age: int, memories: list[dict]) -> list[dict[str, Any]]:
    """Claude-compatible: system as a list of text blocks; user as multi-part
    text with cache markers."""
    return [
        {"role": "system", "content": [{"type": "text", "text": LLM_SYSTEM_PROMPT}]},
        {"role": "user", "content": build_ephemeral_cache_user_blocks(scenario, char, age, memories)},
    ]


_THINKING_BUDGET = int(os.getenv("ANNOTATE_THINKING_BUDGET", "0"))


def _assistant_text(message: Any) -> str:
    if message is None:
        return ""
    c = getattr(message, "content", None)
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif hasattr(block, "text") and getattr(block, "text", None):
                parts.append(str(block.text))
        return "".join(parts).strip()
    return (str(c) if c else "").strip()


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _llm_usage_log_record(resp: Any, extra: dict[str, Any] | None) -> dict[str, Any]:
    ud = _usage_to_dict(getattr(resp, "usage", None))
    ptd = ud.get("prompt_tokens_details")
    if not isinstance(ptd, dict):
        ptd = {}
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "completion_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "usage": ud,
        "cache_summary": {
            "prompt_tokens": ud.get("prompt_tokens"),
            "completion_tokens": ud.get("completion_tokens"),
            "total_tokens": ud.get("total_tokens"),
            "cached_prompt_tokens": ptd.get("cached_tokens"),
            "cache_creation_tokens": ptd.get("cache_creation_tokens"),
        },
        "extra": extra or {},
    }


def _append_conflicts_usage_log(resp: Any, extra: dict[str, Any] | None) -> None:
    if CONFLICTS_USAGE_LOG_PATH is None:
        return
    record = _llm_usage_log_record(resp, extra)
    with _usage_log_lock:
        CONFLICTS_USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFLICTS_USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _chat_completion_with_retry(
    messages: list[dict[str, Any]],
    model: str | None = None,
    use_thinking: bool = False,
    usage_log_extra: dict[str, Any] | None = None,
) -> str:
    use_model = model or LLM_CONFLICTS_MODEL
    attempt = 0
    backoff = 10.0
    while True:
        wait = _rl_state["until"] - time.time()
        if wait > 0:
            jitter = random.uniform(0, min(wait * 0.2, 5))
            time.sleep(wait + jitter)

        extra_body = None
        if use_thinking and _THINKING_BUDGET > 0:
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET}}

        try:
            resp = llm_client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.2,
                max_tokens=int(os.getenv("ANNOTATE_MAX_TOKENS", "0")) or None,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            content = _assistant_text(resp.choices[0].message)
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                content = "\n".join(lines)
            _append_conflicts_usage_log(resp, usage_log_extra)
            return content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower() or "rate_limit" in err_str.lower()

            if is_rate_limit:
                m = re.search(r"retry.after[^\d]*(\d+)", err_str, re.IGNORECASE)
                retry_after = int(m.group(1)) if m else backoff
                actual_wait = max(retry_after, backoff)
                with _rl_lock:
                    _rl_state["until"] = max(_rl_state["until"], time.time() + actual_wait)
                print(f"  [rate-limit] sleeping {actual_wait:.0f}s (backoff={backoff:.0f}s)")
                time.sleep(actual_wait)
                backoff = min(backoff * 2, 120)
                continue

            attempt += 1
            if attempt <= MAX_RETRIES:
                print(f"  [retry {attempt}] {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after {attempt} attempts: {e}")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    use_thinking: bool = False,
    usage_log_extra: dict[str, Any] | None = None,
) -> str:
    return _chat_completion_with_retry(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        use_thinking=use_thinking,
        usage_log_extra=usage_log_extra,
    )


def _sanitize_json(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _parse_json_object(raw: str) -> dict:
    text = _sanitize_json(raw)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Try to extract a {...} fragment.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON object from: {raw[:200]}")


# ===== Core detection =====


def detect_age_batch(
    char: dict, scenario: dict, age: int, memories: list[dict]
) -> dict:
    """Call the LLM for one age batch; return the detection result (one entry
    per conflicting memory with its individual reason)."""
    log_extra = {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "memory_timeline_age": age,
        "memory_count": len(memories),
        "ephemeral_cache": EPHEMERAL_CACHE,
    }
    if EPHEMERAL_CACHE:
        raw = _chat_completion_with_retry(
            build_conflict_messages(char, scenario, age, memories),
            usage_log_extra=log_extra,
        )
    else:
        user_prompt = build_user_prompt(scenario, char, age, memories)
        raw = call_llm(LLM_SYSTEM_PROMPT, user_prompt, usage_log_extra=log_extra)
    result = _parse_json_object(raw)

    # Validate that conflict memory_ids are actually in this batch.
    valid_ids = {m["id"] for m in memories}
    conflicts = result.get("conflicts", [])
    if isinstance(conflicts, list):
        conflicts = [
            c for c in conflicts
            if isinstance(c, dict) and c.get("memory_id") in valid_ids
        ]
    else:
        conflicts = []

    # Ensure required fields are present.
    return {
        "age": age,
        "has_conflict": result.get("has_conflict", False),
        "conflicts": conflicts,
        "memory_count": len(memories),
    }


def merge_scenario_batches(
    char: dict, scenario: dict, age_groups: dict[int, list], results_by_age: dict[int, dict], errors: list[dict]
) -> dict:
    """Merge per-age batches in age order into the final result for one
    (character, scenario) pair."""
    scenario_age = scenario["age"]
    all_batch_results = [results_by_age[age] for age in sorted(age_groups.keys())]
    has_overall_conflict = any(r.get("has_conflict", False) for r in all_batch_results)
    all_conflicts: list = []
    for r in all_batch_results:
        if r.get("has_conflict"):
            all_conflicts.extend(r.get("conflicts", []))
    return {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "scenario_age": scenario_age,
        "has_conflict": has_overall_conflict,
        "conflicts": all_conflicts,
        "batch_errors": errors,
    }


def run_batch_with_ckpt(
    char: dict,
    scenario: dict,
    age: int,
    memories: list[dict],
    age_groups: dict[int, list],
) -> dict | None:
    """
    For (char, scenario), if its `age` batch is incomplete, run the LLM and
    write a checkpoint. If this call completes every age batch for the
    scenario, delete the checkpoint and return the merged result; otherwise
    return None.
    The LLM call runs outside the lock; the checkpoint is reloaded before
    writing to avoid concurrent age batches overwriting each other.
    """
    cid, sid = char["id"], scenario["id"]
    rw = _ckpt_rw_lock(cid, sid)
    mem_count = len(memories)

    with rw:
        ckpt = load_ckpt(cid, sid)
        if ckpt:
            completed_ages = set(ckpt["completed_ages"])
            n_done = len(completed_ages)
            if n_done < len(age_groups):
                _log_ckpt_resume_once(cid, sid, n_done, len(age_groups))
        else:
            completed_ages = set()

        if age in completed_ages:
            return None

    try:
        batch_result = detect_age_batch(char, scenario, age, memories)
    except Exception as e:
        print(f"    [WARN] age={age} batch failed ({cid} × {sid}): {e}")
        batch_result = {
            "age": age,
            "has_conflict": False,
            "conflicts": [],
            "memory_count": mem_count,
        }
        batch_error = {"age": age, "error": str(e)}
    else:
        batch_error = None

    with rw:
        ckpt = load_ckpt(cid, sid)
        if ckpt:
            completed_ages = set(ckpt["completed_ages"])
            results_by_age: dict[int, dict] = {}
            for r in ckpt["results"]:
                a = r.get("age", -1)
                results_by_age[a] = r
            errors_acc: list[dict] = list(ckpt.get("batch_errors", []))
        else:
            completed_ages = set()
            results_by_age = {}
            errors_acc = []

        if age in completed_ages:
            return None

        results_by_age[age] = batch_result
        if batch_error is not None:
            errors_acc.append(batch_error)

        completed_ages.add(age)
        flat = [results_by_age[a] for a in sorted(results_by_age.keys())]
        save_ckpt(cid, sid, list(completed_ages), flat, errors_acc)

        if completed_ages != set(age_groups.keys()):
            return None

        delete_ckpt(cid, sid)
        return merge_scenario_batches(char, scenario, age_groups, results_by_age, errors_acc)


def build_char_age_work_units(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, int, list[dict]]]:
    """
    Expand pending (char, scenario) pairs into work units.
    Each unit is (char, memory_timeline_age, scenarios_sorted); within one
    unit the LLM is called serially in scenario order so the
    "character + memory at this age" prefix can be reused by the cache.
    """
    by_char: dict[str, list[dict]] = defaultdict(list)
    char_by_id: dict[str, dict] = {}
    for char, scenario in pairs:
        cid = char["id"]
        char_by_id[cid] = char
        by_char[cid].append(scenario)

    units: list[tuple[dict, int, list[dict]]] = []
    for cid in sorted(by_char.keys()):
        char = char_by_id[cid]
        scenarios = sorted(
            {s["id"]: s for s in by_char[cid]}.values(),
            key=lambda s: s["id"],
        )
        all_memories = char.get("episodic_memory_set", [])
        ages_union: set[int] = set()
        for s in scenarios:
            ages_union.update(group_memories_by_age(all_memories, s["age"]).keys())

        for age in sorted(ages_union):
            memories = memories_at_timeline_age(all_memories, age)
            if not memories:
                continue
            need = [
                s
                for s in scenarios
                if age in group_memories_by_age(all_memories, s["age"])
            ]
            if need:
                units.append((char, age, need))

    return units


def process_char_age_unit(
    char: dict,
    memory_age: int,
    scenarios: list[dict],
) -> tuple[list[dict], list[tuple[str, str, dict]]]:
    """
    For a fixed (char, memory_age), process scenarios in order. Returns
    (pair_errors, finalized_results). `finalized` is a list of
    (cid, sid, merged_dict) where merged_dict is the output of
    merge_scenario_batches.
    """
    all_memories = char.get("episodic_memory_set", [])
    memories = memories_at_timeline_age(all_memories, memory_age)
    pair_errors: list[dict] = []
    finalized: list[tuple[str, str, dict]] = []

    for scenario in scenarios:
        sid = scenario["id"]
        try:
            scenario_age = scenario["age"]
            if scenario_age is None:
                raise ValueError(f"Scenario {sid} missing 'age' field")
            age_groups = group_memories_by_age(all_memories, scenario_age)
            if not age_groups:
                continue
            merged = run_batch_with_ckpt(
                char, scenario, memory_age, memories, age_groups
            )
            if merged is not None:
                finalized.append((char["id"], sid, merged))
        except Exception as e:
            pair_errors.append(
                {"char_id": char["id"], "scenario_id": sid, "error": str(e)}
            )

    return pair_errors, finalized


# ===== Persistence =====

_save_lock = threading.Lock()


def load_existing() -> dict:
    if not OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return data.get("annotations", {})
    except Exception:
        return {}


def load_detected_pairs() -> set[tuple[str, str]]:
    """Set of pairs already detected (conflict or not), used to skip duplicates."""
    annotations = load_existing()
    detected_pairs: set[tuple[str, str]] = set()
    for char_id, scenario_map in (annotations or {}).items():
        if not isinstance(scenario_map, dict):
            continue
        for scenario_id in scenario_map.keys():
            detected_pairs.add((char_id, scenario_id))
    return detected_pairs


def save_annotation(char_id: str, scenario_id: str, annotation: dict) -> None:
    """Persist a single detection result (conflict or not)."""

    with _save_lock:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if OUTPUT_PATH.exists():
            try:
                existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        annotations = existing.get("annotations", {})
        if char_id not in annotations:
            annotations[char_id] = {}
        annotations[char_id][scenario_id] = annotation

        total = sum(len(v) for v in annotations.values())
        output_data = {
            "dataset_meta": {
                "version": "Conflicts_Detection_v1.0",
                "description": (
                    f"Physical/logical conflict detection (all checked pairs). "
                    f"{total} checked scenario pairs total. Screen: {LLM_CONFLICTS_MODEL}."
                ),
                "characters_source": str(CHARACTERS_PATH.relative_to(PROJECT_ROOT)),
                "scenarios_source": str(SCENARIOS_PATH.relative_to(PROJECT_ROOT)),
                "source": "scripts/detect_conflicts.py",
            },
            "annotations": annotations,
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)


# ===== Data loaders =====


def load_scenarios() -> list[dict]:
    data = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", {})
    if isinstance(scenarios, dict):
        return [s for stage_list in scenarios.values() for s in stage_list]
    return scenarios


def load_characters() -> list[dict]:
    data = json.loads(CHARACTERS_PATH.read_text(encoding="utf-8"))
    return data.get("characters", [])



# ===== Main =====


def main():
    dry_run = "--dry-run" in sys.argv
    char_filter = None
    scenario_filter = None
    stage_filter = None
    workers = DEFAULT_WORKERS
    scenario_workers = DEFAULT_SCENARIO_WORKERS
    limit = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--char" and i + 1 < len(args):
            char_filter = args[i + 1]
        if arg == "--scenario" and i + 1 < len(args):
            scenario_filter = args[i + 1]
        if arg == "--stage" and i + 1 < len(args):
            stage_filter = args[i + 1]
        if arg == "--workers" and i + 1 < len(args):
            try:
                workers = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --workers value: {args[i+1]!r}")
                sys.exit(1)
        if arg == "--scenario-workers" and i + 1 < len(args):
            try:
                scenario_workers = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --scenario-workers value: {args[i+1]!r}")
                sys.exit(1)
        if arg == "--limit" and i + 1 < len(args):
            try:
                limit = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --limit value: {args[i+1]!r}")
                sys.exit(1)

    if not dry_run and not LLM_API_KEY:
        print("ERROR: ANNOTATE_API_KEY not set. Use --dry-run to preview.")
        sys.exit(1)

    characters = load_characters()
    scenarios = load_scenarios()
    print(f"Loaded {len(characters)} characters, {len(scenarios)} scenarios.")

    if char_filter:
        characters = [c for c in characters if c["id"] == char_filter]
        print(f"Filtered to character: {char_filter} ({len(characters)} found)")
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        print(f"Filtered to scenario: {scenario_filter} ({len(scenarios)} found)")
    if stage_filter:
        scenarios = [s for s in scenarios if s.get("stage") == stage_filter]
        print(f"Filtered to stage: {stage_filter} ({len(scenarios)} scenarios)")

    detected_pairs = load_detected_pairs()
    pairs: list[tuple[dict, dict]] = []
    skipped = 0
    for char in characters:
        for scenario in scenarios:
            cid, sid = char["id"], scenario["id"]
            if (cid, sid) in detected_pairs:
                skipped += 1
                continue
            pairs.append((char, scenario))

    if skipped:
        print(f"Skipped {skipped} already-detected pairs (resume mode).")

    pairs, n_policy_excluded = apply_argv_pair_filter(sys.argv, pairs)
    if n_policy_excluded:
        print(f"Excluded {n_policy_excluded} pairs by pair-filter policy.")

    if limit is not None:
        pairs = pairs[:limit]
        print(f"Limited to first {limit} pair(s) via --limit.")

    if not pairs:
        print("No pairs to process.")
        return

    no_mem_pairs: list[tuple[dict, dict]] = []
    work_pairs: list[tuple[dict, dict]] = []
    for char, scenario in pairs:
        ag = group_memories_by_age(
            char.get("episodic_memory_set", []), scenario.get("age", 0)
        )
        if not ag:
            no_mem_pairs.append((char, scenario))
        else:
            work_pairs.append((char, scenario))

    # Estimate batch count (independent of scheduling order, total is fixed).
    total_batches_est = 0
    for char, scenario in work_pairs:
        age_groups = group_memories_by_age(
            char.get("episodic_memory_set", []), scenario.get("age", 0)
        )
        total_batches_est += len(age_groups)

    units = build_char_age_work_units(work_pairs)
    pool_workers = max(1, scenario_workers, workers)
    no_mem_ids = {(c["id"], s["id"]) for c, s in no_mem_pairs}

    print(f"Processing {len(pairs)} pairs → ~{total_batches_est} age-batched LLM calls.")
    print(
        f"LLM (screen): {LLM_CONFLICTS_MODEL} | "
        f"Parallel (char × memory-age) groups: {pool_workers} "
        f"(max of --scenario-workers, --workers; scenarios within a group run serially for prompt cache)"
    )
    print(f"  Ephemeral prompt cache (multipart user): {'on' if EPHEMERAL_CACHE else 'off'} (ANNOTATE_EPHEMERAL_CACHE)")
    if _gemini_model and not EPHEMERAL_CACHE:
        print("  (Gemini: ephemeral multipart cache is off — required by gateway / LiteLLM.)")
    if CONFLICTS_USAGE_LOG_PATH is not None:
        print(f"  LLM usage log (JSONL): {CONFLICTS_USAGE_LOG_PATH} (ANNOTATE_CONFLICTS_USAGE_LOG)")

    if dry_run:
        print(f"  No-memory pairs (skip API): {len(no_mem_pairs)}")
        print(f"  Cache-oriented work units: {len(units)} (each unit = one char + one memory timeline age + ordered scenarios)")
        for char, mem_age, sc_list in units[:3]:
            ids_preview = [s["id"] for s in sc_list[:4]]
            more = f" (+{len(sc_list) - 4} more)" if len(sc_list) > 4 else ""
            print(
                f"    {char['id']} @ memory-age {mem_age}: {len(sc_list)} scenario(s) "
                f"{ids_preview}{more}"
            )
        for char, scenario in pairs[:2]:
            if (char["id"], scenario["id"]) in no_mem_ids:
                print(f"\n  {char['id']} × {scenario['id']} (no eligible memories)")
                continue
            age_groups = group_memories_by_age(
                char.get("episodic_memory_set", []), scenario.get("age", 0)
            )
            print(f"\n  {char['id']} × {scenario['id']} (age<={scenario.get('age')})")
            print(f"  Eligible ages: {sorted(age_groups.keys())}")
            print(f"  Age batches: {len(age_groups)}, Total eligible memories: {sum(len(v) for v in age_groups.values())}")
        log_note = (
            f"usage log → {CONFLICTS_USAGE_LOG_PATH}"
            if CONFLICTS_USAGE_LOG_PATH is not None
            else "usage log off (ANNOTATE_CONFLICTS_USAGE_LOG=0)"
        )
        print(
            f"\n[dry-run] {len(pairs)} pairs, ~{total_batches_est} LLM calls. "
            f"Ephemeral cache multipart: {'on' if EPHEMERAL_CACHE else 'off'}. {log_note}. No API calls made."
        )
        return

    errors: list[dict] = []
    # Progress: one tick per skipped pair + one per completed (char × memory-age) work unit so
    # tqdm advances while scenarios are still mid-flight (pairs alone only tick when a scenario fully finishes).
    _pbar_total = len(no_mem_pairs) + len(units)
    pbar = tqdm(total=_pbar_total, desc="Detecting conflicts") if HAS_TQDM else None

    for char, scenario in no_mem_pairs:
        print(f"    [SKIP] {char['id']} × {scenario['id']}: no eligible memories.")
        if pbar is not None:
            pbar.update(1)

    def _process_unit(unit: tuple[dict, int, list[dict]]):
        char, mem_age, sc_list = unit
        return process_char_age_unit(char, mem_age, sc_list)

    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        futures = [executor.submit(_process_unit, u) for u in units]
        for future in as_completed(futures):
            pair_errors, finalized = future.result()
            errors.extend(pair_errors)
            for cid, sid, merged in finalized:
                save_annotation(cid, sid, merged)
                if merged.get("has_conflict", False):
                    print(f"  [CONFLICT] {cid} × {sid}")
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    if errors:
        err_path = OUTPUT_PATH.parent / f"detection_errors_{int(time.time())}.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"{len(errors)} errors saved to: {err_path}")

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

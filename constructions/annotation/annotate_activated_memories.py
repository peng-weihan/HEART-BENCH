"""
annotate_activated_memories.py — Stage 1: binary memory-activation screening.

For every character × scenario, take all memories with age <= scenario.age,
group them by age, and call the LLM once per age group with a relaxed binary
labelling (activated true/false). The result is written to
activated_memories_binary.json (the path varies with ANNOTATE_SCREEN_MODEL).

Scheduling mirrors detect_conflicts.py to maximise prompt-cache reuse: build
work units keyed by (character, memory-timeline-age) and, within a unit, run
scenarios serially in scenario-id order. The user prompt places the
"character + memories at this age" block first and the
"scenario + instruction" block last.
For multi-character runs: **characters are processed strictly serially** (only
one character is in flight at any moment); within a single character, multiple
units still run in parallel up to ``pool_workers``.
With ANNOTATE_EPHEMERAL_CACHE=1 (non-Gemini models only), the user message is
split into multiple parts and the "character + memories at this age" prefix is
explicitly tagged with cache_control: {type: ephemeral} (Claude /
Anthropic-compatible gateways, identical to detect_conflicts.py). Do NOT enable
this on the Gemini path — it returns 400.

Each successful LLM call appends one JSONL line to
data/annotations/<model_slug>/activated_memories_llm_usage.jsonl
(usage, cache summary, character_id / scenario_id / memory_timeline_age, ...).
Set ANNOTATE_ACTIVATED_USAGE_LOG=0 to disable, or set it to a path to override
the output file (same convention as detect_conflicts'
ANNOTATE_CONFLICTS_USAGE_LOG).

Stage-2 finalize writes to
`data/annotations/<ANNOTATE_REFINE_MODEL slug>/finalize_llm_usage.jsonl` by
default; ANNOTATE_FINALIZE_USAGE_LOG=0 disables it, or set it to a path to
override (same convention as ANNOTATE_ACTIVATED_USAGE_LOG).

For stage 2 (refine to a fixed count), use:
**scripts/finalize_activated_memories.py**

NOTE: the LLM-facing system / user prompt strings later in this file are
intentionally in Chinese — they steer the LLM that operates on the Chinese
narrative dataset and must keep matching it.

Usage:
    python scripts/annotate_activated_memories.py
    python scripts/annotate_activated_memories.py --char CHAR_01_N_HIGH
    python scripts/annotate_activated_memories.py --stage school_age
    python scripts/annotate_activated_memories.py --workers 4
    python scripts/annotate_activated_memories.py --dry-run
    python scripts/annotate_activated_memories.py --no-pair-filter   # full cross product
    python scripts/annotate_activated_memories.py --pair-filter path/to/other_policy.json
    ANNOTATE_EPHEMERAL_CACHE=1   # non-Gemini: multipart user + ephemeral cache on memory prefix
    python scripts/annotate_activated_memories.py --retry-batch-errors   # re-run only ages in batch_errors
    python scripts/annotate_activated_memories.py --retry-batch-errors --char CHAR_01_N_HIGH

Repair loop (batch_errors):
    Each ``batch_errors`` element is one failed memory-timeline age batch for that
    (character, scenario). ``--retry-batch-errors`` re-invokes the LLM only for those ages,
    merges into ``activated_memories``, rewrites ``activated_memories_binary.json``, and drops
    the error entry for an age when that retry succeeds. If an age still fails, the error is
    kept or updated. Repeat the same command until ``--retry-batch-errors --dry-run`` reports
    no work (or until you accept remaining errors).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import threading
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from pair_filter import exclusion_set_from_argv, filter_cross_product_pairs

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

# ===== Config =====

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env(path: Path):
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


_load_env(PROJECT_ROOT / ".env")

LLM_API_KEY = os.getenv("ANNOTATE_API_KEY")
LLM_API_BASE = os.getenv("ANNOTATE_API_BASE", "").rstrip("/")
LLM_SCREEN_MODEL = os.getenv("ANNOTATE_SCREEN_MODEL")
LLM_REFINE_MODEL = os.getenv("ANNOTATE_REFINE_MODEL") or LLM_SCREEN_MODEL

MAX_RETRIES = 2
DEFAULT_WORKERS = int(os.getenv("ANNOTATE_WORKERS", "4"))
DEFAULT_SCENARIO_WORKERS = int(os.getenv("ANNOTATE_SCENARIO_WORKERS", "4"))

llm_client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

_gemini_model = "gemini" in (LLM_SCREEN_MODEL or "").lower()
EPHEMERAL_CACHE = (
    not _gemini_model
    and os.getenv("ANNOTATE_EPHEMERAL_CACHE", "").strip().lower() in ("1", "true", "yes")
)

_ckpt_rw_guard = threading.Lock()
_ckpt_rw_locks: dict[tuple[str, str], threading.Lock] = {}
_ckpt_log_lock = threading.Lock()
_logged_ckpt_pairs: set[tuple[str, str]] = set()


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

# ===== Data paths =====

SCENARIOS_PATH = PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json"
CHARACTERS_PATH = PROJECT_ROOT / "benchmark" / "characters" / "characters_phase11.json"


def _model_slug(model: str | None) -> str:
    """Turn a model name into a filename-safe slug, e.g. deepseek/deepseek-chat -> deepseek_deepseek-chat."""
    if not model:
        return "unknown"
    return re.sub(r"[^\w\-]", "_", model)


_SCREEN_SLUG = _model_slug(LLM_SCREEN_MODEL)
_REFINE_SLUG = _model_slug(LLM_REFINE_MODEL)

OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "annotations" / _SCREEN_SLUG / "activated_memories_binary.json"
FINAL_OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "annotations" / _REFINE_SLUG / "activated_memories_final.json"
CKPT_DIR = PROJECT_ROOT / "benchmark" / "annotations" / "ckpt"
DEFAULT_ACTIVATED_USAGE_LOG = OUTPUT_PATH.parent / "activated_memories_llm_usage.jsonl"
DEFAULT_FINALIZE_USAGE_LOG = FINAL_OUTPUT_PATH.parent / "finalize_llm_usage.jsonl"


def _resolve_activated_usage_log_path() -> Path | None:
    raw = os.getenv("ANNOTATE_ACTIVATED_USAGE_LOG", "").strip()
    if not raw:
        return DEFAULT_ACTIVATED_USAGE_LOG
    low = raw.lower()
    if low in ("0", "false", "no", "off"):
        return None
    if low in ("1", "true", "yes", "on"):
        return DEFAULT_ACTIVATED_USAGE_LOG
    return Path(os.path.expanduser(raw))


def _resolve_finalize_usage_log_path() -> Path | None:
    raw = os.getenv("ANNOTATE_FINALIZE_USAGE_LOG", "").strip()
    if not raw:
        return DEFAULT_FINALIZE_USAGE_LOG
    low = raw.lower()
    if low in ("0", "false", "no", "off"):
        return None
    if low in ("1", "true", "yes", "on"):
        return DEFAULT_FINALIZE_USAGE_LOG
    return Path(os.path.expanduser(raw))


ACTIVATED_USAGE_LOG_PATH: Path | None = _resolve_activated_usage_log_path()
FINALIZE_USAGE_LOG_PATH: Path | None = _resolve_finalize_usage_log_path()
_usage_log_lock = threading.Lock()

TARGET_ACTIVATED = 50
STAGE1_BUFFER = 1.5  # Stage-1 target candidate count = TARGET_ACTIVATED * STAGE1_BUFFER

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
    """Memories whose timeline age == `age` (scenario-independent; used to share one memory block across scenarios)."""
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

# Global rate-limit state: any worker that hits 429 updates `until`; all workers check before each request
_rl_lock = threading.Lock()
_rl_state: dict = {"until": 0.0}  # until: time.time() timestamp at which the cooldown ends

LLM_SYSTEM_PROMPT = """You are a professional psychological / personality researcher, fluent in autobiographical memory, personality psychology, and situated-cognition theory.

Your task: given the situation a character is facing right now, judge which of the following memories the situation will activate. These memories come from a particular age range in the character's life, but the activation judgement must be based on psychological linkage between the memory's content and the current situation — NOT on whether the ages are close.

## Psychological mechanisms of memory activation

Whether a memory is activated depends on whether there is a strong enough match between the current situation and the memory along some **retrieval cue** dimension. The dimensions to consider are listed below — a memory is potentially activated if it has a strong link with the current scenario along **any one** of them.

### 1. Situational structural similarity (Encoding Specificity)
The current scenario and the memory share a **structural pattern**, even if the surface content differs. Look at:
- Whether the character's **social position** in the scenario matches that in the memory (e.g. both facing an authority, both being asked for help, both bystanders)
- Whether the **decision structure** in the scenario matches that in the memory (e.g. both dilemmas, both emergencies, both situations requiring compromise)
- Whether the **interpersonal dynamics** in the scenario echo those in the memory (e.g. trust-betrayal, competition-cooperation, intimacy-distance)

### 2. Emotional resonance and emotion schema (Mood-Congruent Memory / Emotion Schema)
The emotional state aroused by the current scenario matches the emotional experience in the memory:
- **Same-class emotion**: the anxiety / shame / anger evoked by the scenario matches the dominant emotion in the memory
- **Emotion-intensity resonance**: high-arousal scenarios tend to activate equally high-arousal memories (fear activates fear; excitement activates excitement)
- **Unfinished emotion**: under-processed emotions in the memory (e.g. suppressed anger, unreleased grief) are more easily re-evoked by similar situations

### 3. Core-belief and self-schema activation (Self-Schema / Core Belief Activation)
The core belief formed in the memory (psych_conclusion) is relevant to the self-cognition the current scenario touches:
- Whether the scenario **challenges or confirms** a self-cognition formed by the memory (e.g. "I'm not good enough", "I can rely on myself")
- Whether the scenario touches the **relational schema** built in the memory (e.g. "authority is frightening", "people end up leaving anyway")
- Whether the scenario evokes the **worldview assumption** in the memory (e.g. "effort gets rewarded", "the world is unfair")

### 4. Behaviour script and procedural linkage (Behavioural Script)
The behavior_policy formed by the memory can serve directly as the action template for the current scenario:
- Whether the memory's behavioural strategy fits the decision the current scenario demands
- Whether the character formed a **habitual response pattern** in similar situations in the past
- Includes **avoidance behaviours** — if the memory's outcome was painful, the character may instead lean toward the opposite action

### 5. Narrative identity and life themes (Narrative Identity)
The memory is a key node in the character's self-narrative:
- **Turning-point memories**: events that mark a change in the character's life direction
- **Origin-story memories**: events the character uses to explain "why I am the way I am"
- **Recurring themes**: patterns that repeat in the character's life (e.g. repeatedly being abandoned, repeatedly excelling under pressure)

### 6. Somatic marker and sensory cues (Somatic Marker)
Sensory elements in the current scenario have a direct link to the sensory experience in the memory:
- Similar physical environment (e.g. hospital, classroom, family dinner table)
- Similar bodily sensation (e.g. the pressure of being watched, the calm of solitude)
- Specific sensory triggers (e.g. the sound of an argument, a particular smell or season)

## Important pointers

- Don't only look at "topic relatedness" — **deep psychological linkage matters more than surface topical similarity**. For example, a memory of being scolded by a teacher in public could be activated by "having professional competence questioned in a meeting" — one is school, the other is workplace.
- Memories from early development (childhood, adolescence) that formed **core beliefs or emotion patterns** remain easily activated by later scenarios, even decades later.
- High-emotional-intensity memories (trauma, major successes, deep interpersonal connections) have a lower activation threshold.

## Output format

Output strictly one JSON array (no markdown code block) containing **only the activated memories**; do NOT include non-activated ones:

[
  {
    "memory_id": "MEM_XX_XXXX",
    "reason": "20-30 words explaining which mechanism activated it (e.g. structural similarity, emotional resonance, etc.)."
  }
]

**JSON syntax (mandatory):** string values may only use ASCII double quotes `"` as the key/value boundary; **the `reason` string MUST NOT contain unescaped ASCII `"` inside it**, or parsing will fail. To quote someone's words or stress a phrase, use single quotes `'` or curly quotes; do NOT place ASCII `"` inside reason.

If no memory in this batch is activated, output an empty array [].

## Notes

- Output activated memories only; do NOT output non-activated ones.
- The end of each batch states a **suggested activation count** — treat it as an **upper bound**: prefer fewer to more. Only memories with a **strong, direct** psychological link to the current situation should be selected; do not pick fuzzy-related ones.
- Do NOT invent memory_ids that are not in the list."""


def build_char_block(char: dict) -> str:
    bf = char.get("big_five", {})
    return f"""## Character info

- Character ID: {char['id']}
- Name: {char.get('name', '')}
- Archetype: {char.get('archetype', '')}
- Big Five: O={bf.get('openness', 0.5):.2f} C={bf.get('conscientiousness', 0.5):.2f} E={bf.get('extraversion', 0.5):.2f} A={bf.get('agreeableness', 0.5):.2f} N={bf.get('neuroticism', 0.5):.2f}"""


def build_memories_candidate_block(age: int, memories: list[dict]) -> str:
    mem_lines = [f"## Candidate memories (age {age}, {len(memories)} entries)\n"]
    for i, mem in enumerate(memories, 1):
        mem_lines.append(f"### [{i}] {mem['id']}")
        content = mem.get("content_full")
        if not content:
            raise ValueError(f"Memory {mem.get('id', '?')} missing content_full")
        mem_lines.append(f"- {content}")
        mem_lines.append("")
    return "\n".join(mem_lines)


def build_scenario_block_activation(scenario: dict) -> str:
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


def activation_closing_instruction(age: int, suggest: int) -> str:
    return (
        f"For each memory above, decide whether it is activated by this scenario; output a JSON array.\n"
        f"**Reference**: about {suggest} memories in this batch are likely to be activated — use this as a count reference. "
        f"If you genuinely judge that more memories have a **strong, direct** psychological link to the current situation, you may exceed this number; "
        f"otherwise, be strict — when in doubt, leave it out."
    )


def build_user_prompt_activation(
    scenario: dict,
    char: dict,
    age: int,
    memories: list[dict],
    suggest: int,
    extra_suffix: str = "",
) -> str:
    """Character + age-bucket memories first, scenario + instruction last — keeps the fixed prefix prompt-cacheable."""
    char_block = build_char_block(char)
    mem_block = build_memories_candidate_block(age, memories)
    scenario_block = build_scenario_block_activation(scenario)
    closing = activation_closing_instruction(age, suggest)
    return f"""{char_block}

{mem_block}

---

{scenario_block}

{closing}{extra_suffix}"""


def build_ephemeral_cache_user_blocks_activation(
    scenario: dict,
    char: dict,
    age: int,
    memories: list[dict],
    suggest: int,
    extra_suffix: str = "",
) -> list[dict[str, Any]]:
    char_block = build_char_block(char)
    mem_block = build_memories_candidate_block(age, memories)
    scenario_block = build_scenario_block_activation(scenario)
    suffix = f"---\n\n{scenario_block}\n\n{activation_closing_instruction(age, suggest)}{extra_suffix}"
    return [
        {
            "type": "text",
            "text": f"{char_block}\n\n{mem_block}",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": suffix},
    ]


def build_activation_messages(
    scenario: dict,
    char: dict,
    age: int,
    memories: list[dict],
    suggest: int,
    extra_suffix: str = "",
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": LLM_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": build_ephemeral_cache_user_blocks_activation(
                scenario, char, age, memories, suggest, extra_suffix=extra_suffix
            ),
        },
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
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(
                block.get("text"), str
            ):
                parts.append(block["text"])
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


def _append_usage_log(resp: Any, extra: dict[str, Any] | None, path: Path | None) -> None:
    if path is None:
        return
    record = _llm_usage_log_record(resp, extra)
    with _usage_log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_activated_usage_log(resp: Any, extra: dict[str, Any] | None) -> None:
    _append_usage_log(resp, extra, ACTIVATED_USAGE_LOG_PATH)


def _usage_log_path_for_dest(
    usage_log_dest: Literal["stage1", "finalize", "none"],
) -> Path | None:
    if usage_log_dest == "none":
        return None
    if usage_log_dest == "finalize":
        return FINALIZE_USAGE_LOG_PATH
    return ACTIVATED_USAGE_LOG_PATH


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    use_thinking: bool = False,
    usage_log_extra: dict[str, Any] | None = None,
    usage_log_dest: Literal["stage1", "finalize", "none"] = "stage1",
) -> str:
    use_model = model or LLM_SCREEN_MODEL
    attempt = 0
    backoff = 10.0  # Initial backoff in seconds (for 429 only)
    while True:
        # Global rate-limit check: if another worker triggered a cooldown, wait it out + jitter
        wait = _rl_state["until"] - time.time()
        if wait > 0:
            jitter = random.uniform(0, min(wait * 0.2, 5))  # 20% of the cooldown duration, capped at 5s
            time.sleep(wait + jitter)

        extra_body = None
        if use_thinking and _THINKING_BUDGET > 0:
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET}}

        try:
            resp = llm_client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=int(os.getenv("ANNOTATE_MAX_TOKENS", "0")) or None,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            content = _assistant_text(resp.choices[0].message)
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                content = "\n".join(lines)
            _append_usage_log(resp, usage_log_extra, _usage_log_path_for_dest(usage_log_dest))
            return content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower() or "rate_limit" in err_str.lower()

            if is_rate_limit:
                # Try to pull a retry-after seconds value from the error message
                m = re.search(r"retry.after[^\d]*(\d+)", err_str, re.IGNORECASE)
                retry_after = int(m.group(1)) if m else backoff
                actual_wait = max(retry_after, backoff)
                with _rl_lock:
                    _rl_state["until"] = max(_rl_state["until"], time.time() + actual_wait)
                print(f"  [rate-limit] sleeping {actual_wait:.0f}s (backoff={backoff:.0f}s)")
                time.sleep(actual_wait)
                backoff = min(backoff * 2, 120)  # Exponential backoff, capped at 120s
                # 429 does not count toward attempt; retry indefinitely until success
                continue

            attempt += 1
            if attempt <= MAX_RETRIES:
                print(f"  [retry {attempt}] {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after {attempt} attempts: {e}")


def call_llm_messages(
    messages: list[dict[str, Any]],
    model: str | None = None,
    use_thinking: bool = False,
    usage_log_extra: dict[str, Any] | None = None,
    usage_log_dest: Literal["stage1", "finalize", "none"] = "stage1",
) -> str:
    use_model = model or LLM_SCREEN_MODEL
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
            _append_usage_log(resp, usage_log_extra, _usage_log_path_for_dest(usage_log_dest))
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


def _sanitize_json(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _parse_json_array(raw: str) -> list:
    text = _sanitize_json(raw)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Try to extract a [...] fragment
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON array from: {raw[:200]}")


JSON_ACTIVATION_RETRY_SUFFIX = (
    "\n\n[Your previous output could not be parsed by json.loads.] The most common cause is an unescaped ASCII `\"` inside `reason`. "
    "Please **re-output the complete** JSON array (no markdown code block); "
    "inside `reason`, use single quotes `'` (or escape as `\\\"`) for any quoted phrase — do NOT place a bare `\"` in the middle of the string."
)


# ===== Core annotation =====


def annotate_age_batch(
    char: dict, scenario: dict, age: int, memories: list[dict], suggest: int
) -> list[dict]:
    """Call the LLM for one age batch; return only activated memory entries."""
    log_extra = {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "memory_timeline_age": age,
        "memory_count": len(memories),
        "suggest_cap": suggest,
        "ephemeral_cache": EPHEMERAL_CACHE,
        "script": "annotate_activated_memories",
    }
    if EPHEMERAL_CACHE:
        raw = call_llm_messages(
            build_activation_messages(scenario, char, age, memories, suggest),
            usage_log_extra=log_extra,
        )
    else:
        user_prompt = build_user_prompt_activation(scenario, char, age, memories, suggest)
        raw = call_llm(LLM_SYSTEM_PROMPT, user_prompt, usage_log_extra=log_extra)
    try:
        results = _parse_json_array(raw)
    except ValueError:
        log_retry = {**log_extra, "kind": "activation_json_retry"}
        if EPHEMERAL_CACHE:
            raw = call_llm_messages(
                build_activation_messages(
                    scenario, char, age, memories, suggest, extra_suffix=JSON_ACTIVATION_RETRY_SUFFIX
                ),
                usage_log_extra=log_retry,
            )
        else:
            user_retry = build_user_prompt_activation(
                scenario, char, age, memories, suggest, extra_suffix=JSON_ACTIVATION_RETRY_SUFFIX
            )
            raw = call_llm(LLM_SYSTEM_PROMPT, user_retry, usage_log_extra=log_retry)
        results = _parse_json_array(raw)

    # Validate: every returned memory_id must belong to this batch
    valid_ids = {m["id"] for m in memories}
    results = [r for r in results if r.get("memory_id") in valid_ids]

    # Mark all as activated=True (already implied by output format)
    for item in results:
        item["activated"] = True

    return results


def annotate_one_scenario(char: dict, scenario: dict, workers: int = 4, age_filter: int = None) -> dict:
    """Process one (character, scenario) pair with age-batch concurrency.

    When `age_filter` is given, only process that age and merge results into the existing data.
    """
    scenario_age = scenario.get("age")
    if scenario_age is None:
        raise ValueError(f"Scenario {scenario['id']} missing 'age' field")

    all_memories = char.get("episodic_memory_set", [])
    age_groups = group_memories_by_age(all_memories, scenario_age)
    eligible_count = sum(len(v) for v in age_groups.values())

    # When age_filter is given, read existing data as the base
    existing_data = None
    if age_filter is not None:
        existing = load_existing()
        cid, sid = char["id"], scenario["id"]
        if cid in existing and sid in existing[cid]:
            existing_data = existing[cid][sid]
            print(f"    [MERGE] Loading existing data for {cid} × {sid}")

    # Fast path: when eligible memories < TARGET_ACTIVATED, treat them all as activated
    if eligible_count < TARGET_ACTIVATED and age_filter is None:
        print(f"    [FAST] {char['id']} × {scenario['id']}: "
              f"eligible {eligible_count} < {TARGET_ACTIVATED}, marking all activated.")
        all_results = [
            {"memory_id": m["id"], "reason": "eligible_count < target, all included"}
            for age in sorted(age_groups.keys())
            for m in age_groups[age]
        ]
        return {
            "character_id": char["id"],
            "scenario_id": scenario["id"],
            "scenario_age": scenario_age,
            "eligible_memory_count": eligible_count,
            "activated_count": eligible_count,
            "age_batches": len(age_groups),
            "fast_path": True,
            "batch_errors": [],
            "activated_memories": all_results,
        }

    # Compute the suggested activation ratio for stage 1
    stage1_target = TARGET_ACTIVATED * STAGE1_BUFFER
    ratio = min(0.5, stage1_target / eligible_count)

    # Load checkpoint if present
    ckpt = load_ckpt(char["id"], scenario["id"])
    if ckpt:
        completed_ages: set[int] = set(ckpt["completed_ages"])
        results_by_age: dict[int, list[dict]] = {r["_age"]: [] for r in []}  # placeholder
        # Rebuild results_by_age from the checkpoint
        results_by_age = {}
        for r in ckpt["results"]:
            a = r.get("_age", -1)
            results_by_age.setdefault(a, []).append(r)
        print(f"    [CKPT] {char['id']} × {scenario['id']}: resuming, "
              f"{len(completed_ages)}/{len(age_groups)} age-batches done.")
    else:
        completed_ages = set()
        results_by_age: dict[int, list[dict]] = {}

    ckpt_lock = threading.Lock()
    errors: list[dict] = []
    errors_lock = threading.Lock()

    pending_ages = [age for age in sorted(age_groups.keys()) if age not in completed_ages]

    # When --age is given, process only that age
    if age_filter is not None:
        pending_ages = [age for age in pending_ages if age == age_filter]
        if not pending_ages:
            print(f"    [SKIP] age={age_filter} not in this scenario or already completed.")
            if existing_data is not None:
                return existing_data
            return {
                "character_id": char["id"],
                "scenario_id": scenario["id"],
                "scenario_age": scenario_age,
                "eligible_memory_count": eligible_count,
                "activated_count": 0,
                "age_batches": len(age_groups),
                "fast_path": False,
                "batch_errors": [],
                "activated_memories": [],
            }

    def process_age(age: int):
        memories = age_groups[age]
        suggest = max(1, round(len(memories) * ratio))
        try:
            batch_results = annotate_age_batch(char, scenario, age, memories, suggest)
            # Tag each result with its age (for ckpt rebuild)
            for r in batch_results:
                r["_age"] = age
            with ckpt_lock:
                results_by_age[age] = batch_results
                completed_ages.add(age)
                # Flatten all completed results and write the ckpt
                flat = [r for a in sorted(results_by_age) for r in results_by_age[a]]
                save_ckpt(char["id"], scenario["id"], list(completed_ages), flat)
        except Exception as e:
            print(f"    [WARN] age={age} batch failed ({char['id']} × {scenario['id']}): {e}")
            fallback = [
                {"memory_id": m["id"], "activated": False, "reason": "", "_error": str(e), "_age": age}
                for m in memories
            ]
            with ckpt_lock:
                results_by_age[age] = fallback
            with errors_lock:
                errors.append({"age": age, "error": str(e)})

    if pending_ages:
        age_pbar = (
            tqdm(total=len(pending_ages), desc=f"{char['id']}×{scenario['id']}", leave=False)
            if HAS_TQDM else None
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_age, age): age for age in pending_ages}
            for future in as_completed(futures):
                future.result()  # Exceptions are re-raised here (caught inside process_age, so unlikely to reach this point)
                if age_pbar is not None:
                    age_pbar.update(1)
        if age_pbar is not None:
            age_pbar.close()

    # Merge results in age order; strip the internal _age tag
    all_results = []
    for age in sorted(age_groups.keys()):
        for r in results_by_age.get(age, []):
            r_clean = {k: v for k, v in r.items() if k != "_age"}
            all_results.append(r_clean)

    activated = [r for r in all_results if r.get("activated")]

    # Scenario fully done — delete the checkpoint
    delete_ckpt(char["id"], scenario["id"])

    # Keep only activated=true records (drop the activated field itself; True is implied)
    activated_records = [
        {"memory_id": r["memory_id"], "reason": r.get("reason", "")}
        for r in activated
    ]

    # With age_filter set: merge into existing data. Drop the old batch_error only when this age succeeds; on failure, keep the old error and append the new one.
    if age_filter is not None:
        if existing_data is not None:
            # Existing activated_memories
            existing_mems = existing_data.get("activated_memories", [])
            # New activated_memories
            new_mems = activated_records
            # Dedupe by memory_id: keep entries from the new data; drop matching IDs from the old data
            existing_ids = {m["memory_id"] for m in new_mems}
            merged = [m for m in existing_mems if m["memory_id"] not in existing_ids] + new_mems
            activated_records = merged
            print(f"    [MERGED] {len(existing_mems)} existing + {len(new_mems)} new = {len(activated_records)} total")
        existing_errors = existing_data.get("batch_errors", []) if existing_data else []
        kept = [e for e in existing_errors if e.get("age") != age_filter]
        run_failures = list(errors)
        errors = kept + run_failures
        if run_failures:
            print(
                f"    [BATCH_ERR] age={age_filter} still failing ({len(run_failures)}); "
                f"{len(kept)} other batch_errors kept"
            )
        else:
            print(f"    [CLEANED] batch_errors: removed age={age_filter}, {len(errors)} remaining")

    return {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "scenario_age": scenario_age,
        "eligible_memory_count": eligible_count,
        "stage1_ratio": round(ratio, 3) if age_filter is None else existing_data.get("stage1_ratio", 0),
        "activated_count": len(activated_records),
        "age_batches": len(age_groups),
        "fast_path": False,
        "batch_errors": errors,
        "activated_memories": activated_records,
    }


def merge_activation_scenario(
    char: dict,
    scenario: dict,
    age_groups: dict[int, list[dict]],
    results_by_age: dict[int, list[dict]],
    errors: list[dict],
) -> dict:
    scenario_age = scenario["age"]
    eligible_count = sum(len(v) for v in age_groups.values())
    stage1_target = TARGET_ACTIVATED * STAGE1_BUFFER
    ratio = min(0.5, stage1_target / eligible_count) if eligible_count else 0.0

    all_results: list[dict] = []
    for age in sorted(age_groups.keys()):
        for r in results_by_age.get(age, []):
            r_clean = {k: v for k, v in r.items() if k != "_age"}
            all_results.append(r_clean)

    activated = [r for r in all_results if r.get("activated")]
    activated_records = [
        {"memory_id": r["memory_id"], "reason": r.get("reason", "")}
        for r in activated
    ]

    return {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "scenario_age": scenario_age,
        "eligible_memory_count": eligible_count,
        "stage1_ratio": round(ratio, 3),
        "activated_count": len(activated_records),
        "age_batches": len(age_groups),
        "fast_path": False,
        "batch_errors": errors,
        "activated_memories": activated_records,
    }


def run_activation_batch_with_ckpt(
    char: dict,
    scenario: dict,
    age: int,
    memories: list[dict],
    age_groups: dict[int, list[dict]],
    suggest: int,
) -> dict | None:
    cid, sid = char["id"], scenario["id"]
    rw = _ckpt_rw_lock(cid, sid)

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
        batch_results = annotate_age_batch(char, scenario, age, memories, suggest)
        for r in batch_results:
            r["_age"] = age
        batch_error = None
    except Exception as e:
        print(f"    [WARN] age={age} batch failed ({cid} × {sid}): {e}")
        batch_results = [
            {"memory_id": m["id"], "activated": False, "reason": "", "_error": str(e), "_age": age}
            for m in memories
        ]
        batch_error = {"age": age, "error": str(e)}

    with rw:
        ckpt = load_ckpt(cid, sid)
        if ckpt:
            completed_ages = set(ckpt["completed_ages"])
            rb: dict[int, list[dict]] = defaultdict(list)
            for r in ckpt["results"]:
                a = r.get("_age", -1)
                rb[a].append(r)
            errors_acc: list[dict] = list(ckpt.get("batch_errors", []))
        else:
            completed_ages = set()
            rb = defaultdict(list)
            errors_acc = []

        if age in completed_ages:
            return None

        rb[age] = list(batch_results)
        if batch_error is not None:
            errors_acc.append(batch_error)

        completed_ages.add(age)
        flat = [r for a in sorted(rb.keys()) for r in rb[a]]
        save_ckpt(cid, sid, list(completed_ages), flat, errors_acc)

        if completed_ages != set(age_groups.keys()):
            return None

        delete_ckpt(cid, sid)
        results_by_age_final = {a: list(rb[a]) for a in sorted(rb.keys())}
        return merge_activation_scenario(char, scenario, age_groups, results_by_age_final, errors_acc)


def build_char_age_work_units(
    pairs: list[tuple[dict, dict]],
    age_filter: int | None = None,
) -> list[tuple[dict, int, list[dict]]]:
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
            if age_filter is not None and age != age_filter:
                continue
            mems = memories_at_timeline_age(all_memories, age)
            if not mems:
                continue
            need = [
                s
                for s in scenarios
                if age in group_memories_by_age(all_memories, s["age"])
            ]
            if need:
                units.append((char, age, need))

    return units


def group_units_by_character_sequential_order(
    units: list[tuple[dict, int, list[dict]]],
) -> list[tuple[str, list[tuple[dict, int, list[dict]]]]]:
    """First-seen character order; each bucket is all units for that character."""
    order: list[str] = []
    buckets: dict[str, list[tuple[dict, int, list[dict]]]] = {}
    for unit in units:
        cid = str(unit[0].get("id", ""))
        if cid not in buckets:
            order.append(cid)
            buckets[cid] = []
        buckets[cid].append(unit)
    return [(cid, buckets[cid]) for cid in order]


def process_activation_char_age_unit(
    char: dict,
    memory_age: int,
    scenarios: list[dict],
) -> tuple[list[dict], list[tuple[str, str, dict]]]:
    all_memories = char.get("episodic_memory_set", [])
    memories_timeline = memories_at_timeline_age(all_memories, memory_age)
    pair_errors: list[dict] = []
    finalized: list[tuple[str, str, dict]] = []

    if not memories_timeline:
        return pair_errors, finalized

    for scenario in scenarios:
        sid = scenario["id"]
        try:
            scenario_age = scenario["age"]
            if scenario_age is None:
                raise ValueError(f"Scenario {sid} missing 'age' field")
            age_groups = group_memories_by_age(all_memories, scenario_age)
            if memory_age not in age_groups:
                continue
            eligible_count = sum(len(v) for v in age_groups.values())
            stage1_target = TARGET_ACTIVATED * STAGE1_BUFFER
            ratio = min(0.5, stage1_target / eligible_count)
            suggest = max(1, round(len(memories_timeline) * ratio))

            merged = run_activation_batch_with_ckpt(
                char, scenario, memory_age, memories_timeline, age_groups, suggest
            )
            if merged is not None:
                finalized.append((char["id"], sid, merged))
        except Exception as e:
            pair_errors.append(
                {"char_id": char["id"], "scenario_id": sid, "error": str(e)}
            )

    return pair_errors, finalized


def build_fast_path_annotation(char: dict, scenario: dict) -> dict:
    scenario_age = scenario["age"]
    age_groups = group_memories_by_age(char.get("episodic_memory_set", []), scenario_age)
    eligible_count = sum(len(v) for v in age_groups.values())
    all_results = [
        {"memory_id": m["id"], "reason": "eligible_count < target, all included"}
        for age in sorted(age_groups.keys())
        for m in age_groups[age]
    ]
    return {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "scenario_age": scenario_age,
        "eligible_memory_count": eligible_count,
        "activated_count": eligible_count,
        "age_batches": len(age_groups),
        "fast_path": True,
        "batch_errors": [],
        "activated_memories": all_results,
    }


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


def collect_batch_error_retry_work(
    existing: dict[str, Any],
    characters: list[dict],
    scenarios: list[dict],
    char_filter: str | None,
    scenario_filter: str | None,
    stage_filter: str | None,
    excluded: set[tuple[str, str]],
) -> list[tuple[dict, dict, list[int]]]:
    """
    From activated_memories_binary.json annotations, collect (char, scenario, ages) where
    batch_errors is non-empty. Respects the same char/scenario/stage filters and pair exclusion
    as a normal run.
    """
    char_by_id = {c["id"]: c for c in characters}
    scen_by_id = {s["id"]: s for s in scenarios}
    pair_ages: dict[tuple[str, str], set[int]] = defaultdict(set)
    for cid, sc_map in existing.items():
        if cid not in char_by_id:
            continue
        if char_filter and cid != char_filter:
            continue
        for sid, ann in sc_map.items():
            if sid not in scen_by_id:
                continue
            if scenario_filter and sid != scenario_filter:
                continue
            scen = scen_by_id[sid]
            if stage_filter and scen.get("stage") != stage_filter:
                continue
            if (cid, sid) in excluded:
                continue
            for e in ann.get("batch_errors") or []:
                a = e.get("age")
                if isinstance(a, int):
                    pair_ages[(cid, sid)].add(a)
    out: list[tuple[dict, dict, list[int]]] = []
    for cid, sid in sorted(pair_ages.keys()):
        out.append((char_by_id[cid], scen_by_id[sid], sorted(pair_ages[(cid, sid)])))
    return out


def save_annotation(char_id: str, scenario_id: str, annotation: dict) -> None:
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
                "version": "Activated_Memories_Binary_v1.0",
                "description": (
                    f"Binary memory activation annotations (per-memory activated/not). "
                    f"{total} scenario annotations total. Screen: {LLM_SCREEN_MODEL}."
                ),
                "characters_source": str(CHARACTERS_PATH.relative_to(PROJECT_ROOT)),
                "scenarios_source": str(SCENARIOS_PATH.relative_to(PROJECT_ROOT)),
                "source": "scripts/annotate_activated_memories.py",
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
    if "--finalize" in sys.argv:
        print(
            "Stage 2 is now a separate script. Run:\n"
            "  python scripts/finalize_activated_memories.py [same flags, without --finalize]",
            file=sys.stderr,
        )
        sys.exit(2)
    dry_run = "--dry-run" in sys.argv
    retry_batch_errors = "--retry-batch-errors" in sys.argv
    char_filter = None
    scenario_filter = None
    stage_filter = None
    age_filter = None
    workers = DEFAULT_WORKERS
    scenario_workers = DEFAULT_SCENARIO_WORKERS
    limit = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--char" and i + 1 < len(args):
            char_filter = args[i + 1]
        if arg == "--scenario" and i + 1 < len(args):
            scenario_filter = args[i + 1]
        if arg == "--age" and i + 1 < len(args):
            try:
                age_filter = int(args[i + 1])
            except ValueError:
                print(f"ERROR: invalid --age value: {args[i+1]!r}")
                sys.exit(1)
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

    if retry_batch_errors and age_filter is not None:
        print("ERROR: --retry-batch-errors cannot be combined with --age (retry derives ages from batch_errors).")
        sys.exit(1)

    excluded = exclusion_set_from_argv(sys.argv)

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

    if retry_batch_errors:
        existing_retry = load_existing()
        work = collect_batch_error_retry_work(
            existing_retry,
            characters,
            scenarios,
            char_filter,
            scenario_filter,
            stage_filter,
            excluded,
        )
        if limit is not None:
            work = work[:limit]
            print(f"Limited to first {limit} pair(s) with batch_errors via --limit.")
        if not work:
            print("No batch_errors entries to retry (or all pairs excluded by filters).")
            return
        n_age_batches = sum(len(ages) for _, _, ages in work)
        print(
            f"[retry-batch-errors] {len(work)} pair(s), {n_age_batches} age-batch LLM call(s) "
            f"(reads {OUTPUT_PATH.name} under {_SCREEN_SLUG})."
        )
        if dry_run:
            for char_w, scenario_w, ages in work[:20]:
                print(f"    {char_w['id']} × {scenario_w['id']}: ages {ages}")
            if len(work) > 20:
                print(f"    ... and {len(work) - 20} more pair(s)")
            print(
                f"[dry-run] Ephemeral cache multipart: {'on' if EPHEMERAL_CACHE else 'off'}. "
                "No API calls."
            )
            return

        errors_retry: list[dict] = []
        err_lock = threading.Lock()
        pbar_r = tqdm(total=len(work), desc="Retry batch_errors") if HAS_TQDM else None

        def _repair_pair(item: tuple[dict, dict, list[int]]) -> None:
            char_w, scenario_w, ages = item
            cid, sid = char_w["id"], scenario_w["id"]
            try:
                last_result: dict | None = None
                for age in ages:
                    last_result = annotate_one_scenario(
                        char_w, scenario_w, workers=workers, age_filter=age
                    )
                    save_annotation(cid, sid, last_result)
                if last_result is not None:
                    na = last_result.get("activated_count", 0)
                    el = last_result.get("eligible_memory_count", 0)
                    be = last_result.get("batch_errors") or []
                    print(
                        f"  OK: {cid} × {sid} -> {na}/{el} activated "
                        f"(retried ages {ages}; {len(be)} batch_error(s) left)"
                    )
            except Exception as e:
                print(f"  FAIL {cid} × {sid}: {e}")
                with err_lock:
                    errors_retry.append({"char_id": cid, "scenario_id": sid, "error": str(e)})
            finally:
                if pbar_r is not None:
                    pbar_r.update(1)

        with ThreadPoolExecutor(max_workers=max(1, scenario_workers)) as executor:
            futures_r = [executor.submit(_repair_pair, w) for w in work]
            for fut in as_completed(futures_r):
                fut.result()
        if pbar_r is not None:
            pbar_r.close()
        if errors_retry:
            err_path_r = OUTPUT_PATH.parent / f"binary_errors_{int(time.time())}.json"
            with open(err_path_r, "w", encoding="utf-8") as f:
                json.dump(errors_retry, f, ensure_ascii=False, indent=2)
            print(f"{len(errors_retry)} errors saved to: {err_path_r}")
        post = load_existing()
        n_err_items = 0
        n_pairs_with_err = 0
        for _cid, sc_map in post.items():
            for _sid, rec in sc_map.items():
                be = rec.get("batch_errors") or []
                if be:
                    n_pairs_with_err += 1
                    n_err_items += len(be)
        if n_err_items:
            print(
                f"Remaining in {OUTPUT_PATH.name}: {n_err_items} batch_errors item(s) "
                f"across {n_pairs_with_err} pair(s). Re-run with --retry-batch-errors or use "
                "--retry-batch-errors --dry-run to preview."
            )
        else:
            print(f"No batch_errors left under output JSON ({OUTPUT_PATH.name}).")
        print(f"Done. Output: {OUTPUT_PATH}")
        return

    existing = load_existing()
    pairs: list[tuple[dict, dict]] = []
    skipped = 0
    for char in characters:
        for scenario in scenarios:
            cid, sid = char["id"], scenario["id"]
            # If --age is given, process even already-finished pairs (repair mode)
            if age_filter is None and cid in existing and sid in existing[cid]:
                skipped += 1
                continue
            pairs.append((char, scenario))

    if skipped:
        print(f"Skipped {skipped} already-annotated pairs (resume mode).")

    pairs, dropped_pf = filter_cross_product_pairs(pairs, excluded)
    if dropped_pf:
        print(f"Excluded {dropped_pf} pairs by pair-filter policy.")

    if limit is not None:
        pairs = pairs[:limit]
        print(f"Limited to first {limit} pair(s) via --limit.")

    if not pairs:
        print("No pairs to process.")
        return

    fast_pairs_list: list[tuple[dict, dict]] = []
    work_pairs_list: list[tuple[dict, dict]] = []
    units: list[tuple[dict, int, list[dict]]] = []
    pool_workers = max(1, scenario_workers, workers)

    if age_filter is None:
        for char, scenario in pairs:
            ag = group_memories_by_age(
                char.get("episodic_memory_set", []), scenario.get("age", 0)
            )
            eligible_count = sum(len(v) for v in ag.values())
            if eligible_count < TARGET_ACTIVATED:
                fast_pairs_list.append((char, scenario))
            else:
                work_pairs_list.append((char, scenario))
        units = build_char_age_work_units(work_pairs_list)
    else:
        work_pairs_list = list(pairs)

    total_batches_est = 0
    for char, scenario in pairs:
        age_groups = group_memories_by_age(
            char.get("episodic_memory_set", []), scenario.get("age", 0)
        )
        total_batches_est += len(age_groups)

    print(f"Processing {len(pairs)} pairs → ~{total_batches_est} age-batched LLM calls.")
    if age_filter is None:
        print(
            f"LLM (screen): {LLM_SCREEN_MODEL} | "
            f"Parallel (char × memory-age) groups: {pool_workers} "
            f"(scenarios in each group serial for prompt cache) | "
            f"Per-pair age-batch workers (legacy): {workers} | "
            f"Across characters: strict serial"
        )
        print(
            f"  Ephemeral prompt cache (multipart user): "
            f"{'on' if EPHEMERAL_CACHE else 'off'} (ANNOTATE_EPHEMERAL_CACHE)"
        )
        if _gemini_model and not EPHEMERAL_CACHE:
            print("  (Gemini: ephemeral multipart cache is off — gateway / LiteLLM.)")
        if ACTIVATED_USAGE_LOG_PATH is not None:
            print(f"  LLM usage log (JSONL): {ACTIVATED_USAGE_LOG_PATH} (ANNOTATE_ACTIVATED_USAGE_LOG)")
        else:
            print("  LLM usage log: off (ANNOTATE_ACTIVATED_USAGE_LOG=0)")
        print(
            f"  Fast-path pairs (no LLM): {len(fast_pairs_list)} | "
            f"Cache work units: {len(units)}"
        )
    else:
        print(
            f"LLM (screen): {LLM_SCREEN_MODEL} | "
            f"--age={age_filter} repair: parallel pairs, {workers} workers per pair"
        )
        if ACTIVATED_USAGE_LOG_PATH is not None:
            print(f"  LLM usage log (JSONL): {ACTIVATED_USAGE_LOG_PATH} (ANNOTATE_ACTIVATED_USAGE_LOG)")
        else:
            print("  LLM usage log: off (ANNOTATE_ACTIVATED_USAGE_LOG=0)")

    if dry_run:
        if age_filter is None:
            print(
                f"  Each unit = one char + one memory timeline age + ordered scenarios "
                f"(same prefix cached across scenarios)."
            )
            for char, mem_age, sc_list in units[:3]:
                ids_preview = [s["id"] for s in sc_list[:4]]
                more = f" (+{len(sc_list) - 4} more)" if len(sc_list) > 4 else ""
                print(
                    f"    {char['id']} @ memory-age {mem_age}: {len(sc_list)} scenario(s) "
                    f"{ids_preview}{more}"
                )
        for char, scenario in pairs[:2]:
            age_groups = group_memories_by_age(
                char.get("episodic_memory_set", []), scenario.get("age", 0)
            )
            print(f"\n  {char['id']} × {scenario['id']} (age<={scenario.get('age')})")
            print(f"  Eligible ages: {sorted(age_groups.keys())}")
            print(f"  Age batches: {len(age_groups)}, Total eligible memories: {sum(len(v) for v in age_groups.values())}")
        _ul = (
            str(ACTIVATED_USAGE_LOG_PATH)
            if ACTIVATED_USAGE_LOG_PATH is not None
            else "off (ANNOTATE_ACTIVATED_USAGE_LOG=0)"
        )
        print(
            f"\n[dry-run] {len(pairs)} pairs, ~{total_batches_est} LLM calls. "
            f"Ephemeral cache multipart: {'on' if EPHEMERAL_CACHE else 'off'}. "
            f"Usage log → {_ul}. No API calls made."
        )
        return

    errors: list[dict] = []

    if age_filter is not None:
        errors_lock = threading.Lock()
        pbar = tqdm(total=len(pairs), desc="Annotating scenarios") if HAS_TQDM else None

        def _process_pair(char: dict, scenario: dict):
            cid, sid = char["id"], scenario["id"]
            try:
                result = annotate_one_scenario(char, scenario, workers=workers, age_filter=age_filter)
                save_annotation(cid, sid, result)
                n_activated = result.get("activated_count", 0)
                eligible = result.get("eligible_memory_count", 0)
                print(f"  OK: {cid} × {sid} -> {n_activated}/{eligible} memories activated")
            except Exception as e:
                print(f"  FAIL {cid} × {sid}: {e}")
                with errors_lock:
                    errors.append({"char_id": cid, "scenario_id": sid, "error": str(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

        with ThreadPoolExecutor(max_workers=scenario_workers) as executor:
            futures = [executor.submit(_process_pair, char, scenario) for char, scenario in pairs]
            for future in as_completed(futures):
                future.result()

        if pbar is not None:
            pbar.close()
    else:
        for char, scenario in fast_pairs_list:
            cid, sid = char["id"], scenario["id"]
            try:
                result = build_fast_path_annotation(char, scenario)
                save_annotation(cid, sid, result)
                print(
                    f"  OK: {cid} × {sid} -> [fast] {result['activated_count']}/"
                    f"{result['eligible_memory_count']} memories"
                )
            except Exception as e:
                print(f"  FAIL {cid} × {sid}: {e}")
                errors.append({"char_id": cid, "scenario_id": sid, "error": str(e)})

        pbar = tqdm(total=len(units), desc="Annotating (cache schedule)") if HAS_TQDM else None

        def _process_unit(unit: tuple[dict, int, list[dict]]):
            char_u, mem_age, sc_list = unit
            return process_activation_char_age_unit(char_u, mem_age, sc_list)

        for cid_serial, char_units in group_units_by_character_sequential_order(units):
            print(f"  [serial by character] {cid_serial}: {len(char_units)} unit(s)", flush=True)
            with ThreadPoolExecutor(max_workers=pool_workers) as executor:
                futures = [executor.submit(_process_unit, u) for u in char_units]
                for future in as_completed(futures):
                    pair_errors, finalized = future.result()
                    errors.extend(pair_errors)
                    for cid, sid, merged in finalized:
                        save_annotation(cid, sid, merged)
                        n_activated = merged.get("activated_count", 0)
                        eligible = merged.get("eligible_memory_count", 0)
                        fp = " [fast]" if merged.get("fast_path") else ""
                        print(
                            f"  OK: {cid} × {sid} ->{fp} {n_activated}/{eligible} memories activated"
                        )
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

    if errors:
        err_path = OUTPUT_PATH.parent / f"binary_errors_{int(time.time())}.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"{len(errors)} errors saved to: {err_path}")

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

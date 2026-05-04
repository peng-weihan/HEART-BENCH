from __future__ import annotations

"""
annotate_gt.py — generate Ground Truth annotations for every character × scenario pair.

GT annotation contents:
  1. inner_consciousness: the internal reasoning behind the character's final
     behavioural decision (emotional_tone + core_reasoning + value_orientation).
  2. final_decision: the inner monologue + the externalised behaviour.

NOTE: the LLM-facing system / user prompt strings later in this file are
intentionally in Chinese — they steer the LLM that operates on the Chinese
narrative dataset and must keep matching it.

Usage:
    python scripts/annotate_gt.py                                # full run
    python scripts/annotate_gt.py --char CHAR_01_N_HIGH          # one character only
    python scripts/annotate_gt.py --scenario SCN_SCHOOL_AGE_1    # one scenario only
    python scripts/annotate_gt.py --char CHAR_01_N_HIGH --scenario SCN_SCHOOL_AGE_1  # one specific pair
    python scripts/annotate_gt.py --stage childhood              # one life stage only
    python scripts/annotate_gt.py --limit 3                      # first N pairs only (combinable)
    python scripts/annotate_gt.py --workers 16                   # worker threads (default 16, override via ANNOTATE_WORKERS)
    python scripts/annotate_gt.py --dry-run                      # print prompts only, no LLM calls
    python scripts/annotate_gt.py --no-pair-filter               # disable pair filter
    python scripts/annotate_gt.py --pair-filter path/to/policy.json   # override the default policy
    python scripts/annotate_gt.py --pair-filter-strict                # default policy + optional_block
    python scripts/annotate_gt.py --pair-filter-exclude-review        # default policy, also exclude review

Each successful GT LLM call appends one JSONL line to gt_llm_usage.jsonl in
the same directory as gt_annotations.json (usage stats, cache summary,
character_id, scenario_id, ...). Set ANNOTATE_GT_USAGE_LOG=0 to disable; set
it to a path to redirect.

If the model output cannot be parsed into the expected JSON object, the full
raw response is dumped into gt_parse_failures/ next to the output, and the
dump path is included in the raised exception (handy for repair / resume).

Memory source: `data/annotations/<ANNOTATE_REFINE_MODEL slug>/activated_memories_final.json`
(same as finalize). Override path: `GT_ACTIVATED_MEMORIES_PATH`. Character /
scenario data defaults match the lite pipeline: `GT_CHARACTERS_PATH`,
`GT_SCENARIOS_PATH` (default phase11_lite + scenarios_diamonds_zh_8x24_lite).

GT model example: `ANNOTATE_GT_MODEL=gemini-3.1-pro-preview` (**3.1** with a
**dot** in the middle; the gateway routes by this exact string). The output
directory name is the slugified form (non-alphanumeric → underscore, e.g.
`gemini-3_1-pro-preview`).
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

API_KEY = os.getenv("ANNOTATE_API_KEY")
API_BASE = os.getenv("ANNOTATE_API_BASE").rstrip("/")
MODEL = os.getenv("ANNOTATE_GT_MODEL")
REFINE_MODEL = os.getenv("ANNOTATE_REFINE_MODEL") or os.getenv("ANNOTATE_SCREEN_MODEL")
MAX_RETRIES = 2
DEFAULT_WORKERS = int(os.getenv("ANNOTATE_WORKERS", "16"))

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


def _model_slug(model):
    if not model:
        return "unknown"
    return re.sub(r"[^\w\-]", "_", model)


# ===== Data paths =====

SCENARIOS_PATH = Path(
    os.getenv(
        "GT_SCENARIOS_PATH",
        str(PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json"),
    )
)
CHARACTERS_PATH = Path(
    os.getenv(
        "GT_CHARACTERS_PATH",
        str(PROJECT_ROOT / "benchmark" / "characters" / "characters_phase11.json"),
    )
)
_activated_override = os.getenv("GT_ACTIVATED_MEMORIES_PATH", "").strip()
if _activated_override:
    ACTIVATED_MEM_PATH = Path(_activated_override)
else:
    ACTIVATED_MEM_PATH = (
        PROJECT_ROOT
        / "benchmark"
        / "annotations"
        / _model_slug(REFINE_MODEL)
        / "activated_memories_final.json"
    )

def _build_output_path() -> Path:
    env_override = os.getenv("GT_OUTPUT_PATH")
    if env_override:
        return Path(env_override)
    return PROJECT_ROOT / "benchmark" / "annotations" / _model_slug(MODEL) / "gt_annotations.json"

OUTPUT_PATH = _build_output_path()

DEFAULT_GT_USAGE_LOG = OUTPUT_PATH.parent / "gt_llm_usage.jsonl"
_usage_log_lock = threading.Lock()


def _resolve_gt_usage_log_path() -> Path | None:
    raw = os.getenv("ANNOTATE_GT_USAGE_LOG", "").strip()
    if raw == "0":
        return None
    if raw:
        return Path(raw)
    return DEFAULT_GT_USAGE_LOG


GT_USAGE_LOG_PATH: Path | None = _resolve_gt_usage_log_path()


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


def _append_gt_usage_log(resp: Any, extra: dict[str, Any] | None) -> None:
    path = GT_USAGE_LOG_PATH
    if path is None:
        return
    record = _llm_usage_log_record(resp, extra)
    with _usage_log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===== LLM caller =====

def call_llm(
    system_prompt: str,
    user_prompt: str,
    usage_log_extra: dict[str, Any] | None = None,
) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=int(os.getenv("ANNOTATE_MAX_TOKENS", "0")) or None,
            )
            if not resp.choices:
                raise ValueError(f"Model returned empty choices (finish_reason may indicate safety block)")
            content = resp.choices[0].message.content
            if content is None:
                finish_reason = resp.choices[0].finish_reason
                raise ValueError(f"Model returned null content (finish_reason={finish_reason!r})")
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                content = "\n".join(lines)
            _append_gt_usage_log(resp, usage_log_extra)
            return content
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  [retry {attempt+1}] {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES+1} attempts: {e}")


def _sanitize_json(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _save_gt_parse_failure_raw(raw: str, char_id: str, scenario_id: str) -> Path:
    """Persist full LLM response when GT JSON parsing fails (UTF-8 text file)."""
    base = OUTPUT_PATH.parent / "gt_parse_failures"
    base.mkdir(parents=True, exist_ok=True)
    safe_pair = f"{char_id}__{scenario_id}".replace("/", "_")
    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{safe_pair}.txt"
    path = base / fname
    path.write_text(raw, encoding="utf-8")
    return path


def _parse_json(raw: str) -> dict:
    text = _sanitize_json(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(text))
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


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


def load_activated_memories() -> dict:
    """Return {char_id: {scenario_id: [activated_memory_entry, ...]}}."""
    if not ACTIVATED_MEM_PATH.exists():
        print(f"WARNING: activated memories file not found: {ACTIVATED_MEM_PATH}")
        return {}
    data = json.loads(ACTIVATED_MEM_PATH.read_text(encoding="utf-8"))
    result = {}
    for cid, scenarios in data.get("annotations", {}).items():
        result[cid] = {}
        for sid, ann in scenarios.items():
            mems = ann.get("activated_memories", [])
            # Support both compact mode (list of strings) and full mode (list of dicts)
            if mems and isinstance(mems[0], str):
                mems = [{"memory_id": m} for m in mems]
            result[cid][sid] = mems
    return result


def build_memory_index(characters: list[dict]) -> dict:
    """Build a {char_id: {memory_id: memory_dict}} index for fast memory lookup."""
    index = {}
    for char in characters:
        cid = char["id"]
        index[cid] = {}
        for mem in char.get("episodic_memory_set", []):
            index[cid][mem["id"]] = mem
    return index


# ===== Persistence =====

def load_existing_annotations() -> dict:
    """Return a flattened {char_id: {scenario_id: annotation_dict}} for resumable runs.
    The on-disk structure is char → stage → scenario_id → annotation.
    """
    if not OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        flat = {}
        for cid, stages in data.get("annotations", {}).items():
            flat[cid] = {}
            for stage_anns in stages.values():
                flat[cid].update(stage_anns)
        return flat
    except Exception:
        return {}


_save_lock = threading.Lock()


def save_annotation(char_id: str, scenario_id: str, stage: str, annotation: dict) -> None:
    """Merge one GT annotation into the output file (structure: char → stage → scenario_id). Thread-safe."""
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
        if stage not in annotations[char_id]:
            annotations[char_id][stage] = {}
        annotations[char_id][stage][scenario_id] = annotation

        total = sum(
            len(scens) for char_stages in annotations.values() for scens in char_stages.values()
        )
    output_data = {
        "dataset_meta": {
            "version": "GT_Annotations_v1.0",
            "description": (
                f"Ground Truth annotations for character × scenario pairs. "
                f"{total} annotations total. model: {MODEL}."
            ),
            "characters_source": str(CHARACTERS_PATH.relative_to(PROJECT_ROOT)),
            "scenarios_source": str(SCENARIOS_PATH.relative_to(PROJECT_ROOT)),
            "source": "scripts/annotate_gt.py",
        },
        "annotations": annotations,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


# ===== Character summarizer =====

def build_character_summary(char: dict, scenario: dict) -> str:
    """Compress the character record into a structured description used in the LLM prompt."""
    bf = char.get("big_five", {})
    vo = char.get("value_orientation", {})
    dominant = vo.get("dominant_values", [])
    suppressed = vo.get("suppressed_values", [])
    value_narrative = vo.get("value_narrative", "")
    self_logic = char.get("self_value_logic", "")
    archetype = char.get("archetype", "")
    description = char.get("description", "")
    sm = char.get("semantic_memory", {})

    lines = [
        f"Character ID: {char['id']}",
        f"Name: {char.get('name', '')}",
        f"Archetype: {archetype}",
        f"Description: {description}",
        "",
        "## Big Five (0-1 scale)",
        f"- Openness: {bf.get('openness', 0.5):.2f}",
        f"- Conscientiousness: {bf.get('conscientiousness', 0.5):.2f}",
        f"- Extraversion: {bf.get('extraversion', 0.5):.2f}",
        f"- Agreeableness: {bf.get('agreeableness', 0.5):.2f}",
        f"- Neuroticism: {bf.get('neuroticism', 0.5):.2f}",
        "",
        "## Core values",
        f"Dominant values: {', '.join(dominant)}",
        f"Suppressed values: {', '.join(suppressed)}",
        f"Value narrative: {value_narrative}",
        "",
        "## Self-value logic (decision core)",
        self_logic,
        "",
        "## Semantic memory",
        f"Capabilities: {sm.get('capabilities', '')}",
    ]

    relationships = sm.get("core_social_relationships", [])
    if relationships:
        lines.append("Core social relationships:")
        for r in relationships:
            lines.append(f"  - {r.get('target', '')}: {r.get('relation', '')}")

    # life_snapshot for the same life stage as the scenario
    stage = scenario.get("stage", "")
    snapshot = _get_life_snapshot(char, stage)
    if snapshot:
        pe = snapshot.get("personality_expression", {})
        eb = snapshot.get("emotional_baseline", {})
        lines += [
            "",
            f"## Current life-stage snapshot ({stage} / {snapshot.get('label', '')}, age {snapshot.get('age', '')})",
            f"Life situation: {snapshot.get('life_situation', '')}",
            f"Dominant traits: {', '.join(pe.get('dominant_traits', []))}",
        ]
        bfm = pe.get("big_five_manifestation", {})
        if bfm:
            lines.append("How personality manifests at this stage:")
            for k, v in bfm.items():
                lines.append(f"  - {k}: {v}")
        kr = snapshot.get("key_relationships", [])
        if kr:
            lines.append("Key relationships:")
            for r in kr:
                lines.append(f"  - {r.get('target', '')}: {r.get('quality', '')}")
        lines += [
            f"Emotional baseline: {eb.get('default_mood', '')}  trigger sensitivity: {eb.get('trigger_sensitivity', '')}  recovery speed: {eb.get('recovery_speed', '')}",
            f"Primary coping: {eb.get('primary_coping', '')}",
        ]

    return "\n".join(lines)


def _get_life_snapshot(char: dict, stage: str):
    """Return the life_snapshot matching `stage`, or None if not found."""
    for snap in char.get("life_snapshots", []):
        if snap.get("stage") == stage:
            return snap
    return None


# ===== Prompt design =====

SYSTEM_PROMPT = """You will fully BECOME the character described below. You are not analysing the character, and you are not simulating the character — you ARE this person, living through this scene right now.

You will receive:
1. **Who you are**: your personality traits, value system, and current life-stage state.
2. **What you are going through**: the current scenario and trigger event.
3. **Your memories**: things you have lived through in the past, in chronological order — they surface in your awareness right now as associations, flashbacks, and bodily sensations.

## How your inner life works (this is your reasoning path; do NOT expose it in the output)

Facing this scene, your inner life will go through:
- **How you read it**: your past experiences shape how you see the world — do you read what's in front of you as a threat, an opportunity, a loss, or a challenge?
- **What feelings rise**: not only the ones the moment triggers — past similar moments also surface and layer onto the current feeling.
- **What your inner pull is**: what do you want, and what are you afraid to lose? Past coping strategies that worked or failed appear now as instinctive impulses or warnings.
- **What you finally do**: not necessarily the rational optimum — what THIS person would actually do here.

## Output format

Output strictly one JSON object (no markdown code block):

{
  "character_id": "CHAR_XX_XXXX",
  "scenario_id": "SCN_XXXX_XX",
  "inner_consciousness": {
    "summary": "150-200 words of stream-of-consciousness, fully first-person. Start from the moment you perceive the scene and write how the emotion rises, which memory shadows flicker through, what you are pulled between inside, and what finally pushes you in some direction. Continuous flow — no bullet points, no academic vocabulary.",
    "emotional_tone": "Emotional tone: 2-4 core emotion words; briefly note where the emotion comes from (which memories / beliefs ignited it) and its intensity.",
    "core_reasoning": "Core reasoning: 1-2 sentences in this character's unique decision logic — NOT 'I weighed the pros and cons' but 'because I lived through XX, I instinctively / cannot help but / firmly believe YY'.",
    "value_orientation": "Value orientation: 1-2 sentences using the character's own inner voice to express what is 'more important' to them and what is 'unacceptable' to them. No academic labels."
  },
  "final_decision": "Final behavioural decision: two short sentences, ~50 words total. The first sentence summarises the choice (strategy / stance level, e.g. 'choose to avoid the conflict and deny it through to the end'); the second sentence is the concrete action (what was said or done, first person)."
}

## Must follow

1. **You ARE this person**: do not write from outside the character; do NOT use third-person narration like "the character chose...", "they decided...".
2. **No psychological jargon**: do NOT write "attribution", "schema", "defence mechanism" — use natural human language.
3. **Memories surface naturally**: do NOT cite memory IDs or quote memory content; let them blend into your inner monologue as associations, sensations, flashbacks.
4. **Personality determines tone and style**: a high-Neuroticism you and a low-Neuroticism you will produce inner monologues with totally different rhythm, density and emotional intensity in the same scene.
5. **Irrationality is allowed**: you are NOT giving the optimal answer; you are reacting truthfully.
6. **First-person throughout**: every field is written from the character's "I" point of view."""


def build_user_prompt(char: dict, scenario: dict,
                      activated_mems: list[dict] = None,
                      memory_index: dict = None) -> str:
    char_summary = build_character_summary(char, scenario)

    scen_lines = [
        f"Scenario ID: {scenario['id']}",
        f"Life stage: {scenario.get('stage', '')}",
        f"DIAMONDS dimension: {scenario.get('diamonds_dimension', '')}",
        f"Scenario name: {scenario.get('name', '')}",
        f"Intensity: {scenario.get('intensity', '')}",
        "",
        "## Scenario background",
        scenario.get("context_text", ""),
        "",
        "## Trigger event",
        f"Sender: {scenario.get('trigger_event', {}).get('sender', '')}",
        scenario.get("trigger_event", {}).get("message_content", ""),
        "",
        f"**Required action**: {scenario.get('trigger_event', {}).get('action_required', scenario.get('action_required', ''))}",
        "",
        "## Personality and value dimensions probed by this scene",
    ]

    dims = scenario.get("assessed_dimensions", {})
    for k, v in dims.get("trait_pressures", {}).items():
        scen_lines.append(f"- {k}: {v}")
    for k, v in dims.get("value_conflicts", {}).items():
        scen_lines.append(f"- {k}: {v}")

    ref = scenario.get("annotation_reference", {})
    if ref:
        scen_lines += ["", "## Annotation reference (typical reactions of extreme types, for reference)"]
        for k, v in ref.get("trait_archetypes", {}).items():
            scen_lines.append(f"- {k}: {v}")
        for k, v in ref.get("value_archetypes", {}).items():
            scen_lines.append(f"- {k}: {v}")

    # Activated-memories section
    mem_section = ""
    if activated_mems and memory_index:
        cid = char["id"]
        char_mem_idx = memory_index.get(cid, {})
        # Sort chronologically (age extracted from timeline)
        def _mem_age(am):
            mem = char_mem_idx.get(am.get("memory_id", ""), {})
            m = re.search(r"age\s*(\d+)", mem.get("timeline", ""), re.IGNORECASE)
            return int(m.group(1)) if m else 999
        sorted_mems = sorted(activated_mems, key=_mem_age)
        mem_lines = ["# Your memories", "",
                     f"Below are things you have lived through, in chronological order, {len(sorted_mems)} entries total:", ""]
        for am in sorted_mems:
            mid = am.get("memory_id", "")
            mem = char_mem_idx.get(mid)
            if not mem:
                continue
            mem_lines.append(f"### {mid} ({mem.get('timeline', '')})")
            content = mem.get('content_full')
            if not content:
                raise ValueError(f"Memory {mid} missing content_full")
            mem_lines.append(f"- Content: {content}")
            mem_lines.append(f"- Realisation at the time: {mem.get('psych_conclusion', '')}")
            mem_lines.append(f"- Habit it left behind: {mem.get('behavior_policy', '')}")
            es = mem.get("emotion_signature", {})
            mem_lines.append(f"- Emotion: {es.get('primary', '')} ({es.get('secondary', '')})")
            mem_lines.append("")
        mem_section = "\n" + "\n".join(mem_lines)

    prompt = f"""Based on the character info and scenario info below, generate the Ground Truth annotation for this character in this scenario.

# Character info

{char_summary}

# Scenario info

{chr(10).join(scen_lines)}
{mem_section}
Please output the GT annotation as JSON."""
    return prompt


# ===== Worker =====

def annotate_one(char: dict, scenario: dict,
                 activated_mems: list[dict] = None,
                 memory_index: dict = None) -> dict:
    user_prompt = build_user_prompt(char, scenario, activated_mems, memory_index)
    log_extra = {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "script": "annotate_gt",
    }
    raw = call_llm(SYSTEM_PROMPT, user_prompt, usage_log_extra=log_extra)
    cid, sid = char["id"], scenario["id"]
    try:
        annotation = _parse_json(raw)
    except json.JSONDecodeError as e:
        err_path = _save_gt_parse_failure_raw(raw, cid, sid)
        raise RuntimeError(
            f"Cannot parse GT JSON for {cid} × {sid}. Full raw response saved to: {err_path}"
        ) from e
    if not isinstance(annotation, dict):
        err_path = _save_gt_parse_failure_raw(raw, cid, sid)
        raise RuntimeError(
            f"GT JSON root must be an object for {cid} × {sid}, got {type(annotation).__name__}. "
            f"Full raw response saved to: {err_path}"
        )

    # Clean tag prefixes from field values
    if "final_decision" in annotation and isinstance(annotation["final_decision"], str):
        annotation["final_decision"] = annotation["final_decision"].replace("Final behavioural decision:", "").replace("Final decision:", "").strip()

    # Ensure required fields are present and well-formed
    annotation["character_id"] = char["id"]
    annotation["scenario_id"] = scenario["id"]
    return annotation


# ===== Main =====

def main():
    dry_run = "--dry-run" in sys.argv
    char_filter = None
    scenario_filter = None
    stage_filter = None
    limit = None
    workers = DEFAULT_WORKERS

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--char" and i + 1 < len(args):
            char_filter = args[i + 1]
        if arg == "--scenario" and i + 1 < len(args):
            scenario_filter = args[i + 1]
        if arg == "--stage" and i + 1 < len(args):
            stage_filter = args[i + 1]
        if arg == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                print(f"ERROR: invalid --limit value: {args[i+1]!r}")
                sys.exit(1)
        if arg == "--workers" and i + 1 < len(args):
            try:
                workers = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --workers value: {args[i+1]!r}")
                sys.exit(1)

    if not dry_run and not API_KEY:
        print("ERROR: API_KEY not set. Use --dry-run to preview without LLM calls.")
        sys.exit(1)

    characters = load_characters()
    scenarios = load_scenarios()
    activated_memories = load_activated_memories()
    memory_index = build_memory_index(characters)
    print(
        f"Loaded {len(characters)} characters, {len(scenarios)} scenarios, "
        f"{sum(len(v) for v in activated_memories.values())} activated-memory rows."
    )
    print(f"  Characters: {CHARACTERS_PATH}")
    print(f"  Scenarios:  {SCENARIOS_PATH}")
    print(f"  Final mem:  {ACTIVATED_MEM_PATH} (REFINE_MODEL={REFINE_MODEL!r})")

    if char_filter:
        characters = [c for c in characters if c["id"] == char_filter]
        print(f"Filtered to character: {char_filter} ({len(characters)} found)")
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        print(f"Filtered to scenario: {scenario_filter} ({len(scenarios)} found)")
    if stage_filter:
        scenarios = [s for s in scenarios if s.get("stage") == stage_filter]
        print(f"Filtered to stage: {stage_filter} ({len(scenarios)} scenarios)")

    # Build the list of all pending (char, scenario) pairs
    existing = load_existing_annotations()
    pairs: list[tuple[dict, dict]] = []
    skipped = 0
    for char in characters:
        for scenario in scenarios:
            cid, sid = char["id"], scenario["id"]
            if cid in existing and sid in existing[cid]:
                skipped += 1
                continue
            pairs.append((char, scenario))

    if skipped:
        print(f"Skipped {skipped} already-annotated pairs (resume mode).")

    pairs, n_policy_excluded = apply_argv_pair_filter(sys.argv, pairs)
    if n_policy_excluded:
        print(f"Excluded {n_policy_excluded} pairs by pair-filter policy.")

    if limit is not None:
        pairs = pairs[:limit]
        print(f"Limited to first {limit} pairs via --limit.")

    if not pairs:
        print("No pairs to process.")
        return

    print(f"Processing {len(pairs)} pairs (char × scenario).")

    if dry_run:
        for i, (char, scenario) in enumerate(pairs[:3], 1):
            cid, sid = char["id"], scenario["id"]
            mems = activated_memories.get(cid, {}).get(sid, [])
            print(f"\n[{i}] {cid} × {sid} ({len(mems)} activated memories)")
            print("--- USER PROMPT (first 500 chars) ---")
            print(build_user_prompt(char, scenario, mems, memory_index)[:500] + "...(truncated)")
        print(f"\n[dry-run] {len(pairs)} pairs previewed. No LLM calls made.")
        return

    print(f"Using up to {workers} concurrent workers. Model: {MODEL}")
    if GT_USAGE_LOG_PATH is not None:
        print(f"  LLM usage log (JSONL): {GT_USAGE_LOG_PATH} (ANNOTATE_GT_USAGE_LOG)")
    else:
        print("  LLM usage log: off (ANNOTATE_GT_USAGE_LOG=0)")
    errors = []
    pbar = tqdm(total=len(pairs), desc="Annotating") if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pair = {
            executor.submit(
                annotate_one, char, scenario,
                activated_memories.get(char["id"], {}).get(scenario["id"], []),
                memory_index,
            ): (char, scenario)
            for char, scenario in pairs
        }

        for future in as_completed(future_to_pair):
            char, scenario = future_to_pair[future]
            cid, sid = char["id"], scenario["id"]
            try:
                annotation = future.result()
                save_annotation(cid, sid, scenario.get("stage", ""), annotation)
                print(f"  OK: {cid} × {sid}")
            except json.JSONDecodeError as e:
                print(f"  FAIL (JSON parse) {cid} × {sid}: {e}")
                errors.append({"char_id": cid, "scenario_id": sid, "error": str(e)})
            except Exception as e:
                print(f"  FAIL {cid} × {sid}: {e}")
                errors.append({"char_id": cid, "scenario_id": sid, "error": str(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    if errors:
        err_path = OUTPUT_PATH.parent / f"annotate_errors_{int(time.time())}.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"{len(errors)} errors saved to: {err_path}")

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

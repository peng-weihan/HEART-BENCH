"""
finalize_activated_memories.py — Stage 2: refined memory activation.

Reads the stage-1 output (activated_memories_binary.json) and, for each
character × scenario pair, asks the LLM to refine the memories that were
flagged as `activated` down to a smaller curated list. The result is written
to activated_memories_final.json.

Dependency: run annotate_activated_memories.py first (stage 1).

Stage-1 JSON path = `data/annotations/<ANNOTATE_SCREEN_MODEL slug>/activated_memories_binary.json`.
Stage-2 output    = `data/annotations/<ANNOTATE_REFINE_MODEL slug>/activated_memories_final.json`
                    (character data lives in `characters_phase9.json`).

Each successful finalize LLM call appends one JSONL line to
`finalize_llm_usage.jsonl` in the same directory by default (usage stats,
cache summary, character_id, scenario_id, etc.). Set
`ANNOTATE_FINALIZE_USAGE_LOG=0` to disable; set it to an absolute or relative
path to redirect the log (same convention as `ANNOTATE_ACTIVATED_USAGE_LOG`).

If the model output cannot be parsed into a JSON object containing
`activated_memories`, the **full raw response** is dumped into
`finalize_parse_failures/` next to the output JSON, and a `RuntimeError` is
raised (the dump path is included in the error message).

NOTE: the LLM system prompt and user-prompt builder below are intentionally
in Chinese — they steer the LLM that operates on the Chinese narrative
dataset and must keep matching it.

Usage:
    python scripts/finalize_activated_memories.py
    python scripts/finalize_activated_memories.py --char CHAR_01_N_HIGH
    python scripts/finalize_activated_memories.py --target 50
    python scripts/finalize_activated_memories.py --workers 4
    python scripts/finalize_activated_memories.py --compact
    python scripts/finalize_activated_memories.py --no-pair-filter
    python scripts/finalize_activated_memories.py --pair-filter path/to/other_policy.json
    python scripts/finalize_activated_memories.py --dedupe              # dedupe an existing final JSON (keep first by memory_id)
    python scripts/finalize_activated_memories.py --dedupe --dry-run   # only count what would be removed
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import annotate_activated_memories as am

from pair_filter import exclusion_set_from_argv

# Re-export tqdm handle from stage-1 module (same venv / progress bar)
try:
    tqdm = am.tqdm
    HAS_TQDM = am.HAS_TQDM
except AttributeError:
    tqdm = None
    HAS_TQDM = False

# ===== Stage 2 prompts & logic =====

LLM_FINALIZE_SYSTEM_PROMPT = """You are a professional psychological / personality researcher, fluent in autobiographical memory, personality dynamics, and behavioural-decision theory.

Your task: from the candidate activated memories produced by Stage 1, pick the {target} memories that **will actually enter the character's consciousness and influence behaviour** in the given scenario.

## Psychological basis for the selection

Stage 1 surfaced every memory that *could* be activated. But at any single moment only a small number of memories actually enter working memory and shape behaviour. You decide which memories will "rise into consciousness" using these priorities:

### Priority 1: memories that directly drive the current action
- **Behaviour-script match**: the memory's behavior_policy directly answers "what should I do right now?"
- **Conditioned activation**: a response pattern repeatedly reinforced by similar situations (e.g. "every time I am questioned I apologise first")
- **Approach-avoidance motivation**: a painful or successful past experience that directly pushes the character toward or away from an option here

### Priority 2: memories that shape the interpretive frame
- **Source of the attribution pattern**: determines whether the character reads the current event as a "threat" or "opportunity", as "someone's malice" or "a misunderstanding"
- **Anchor of the core belief**: the experience that formed the core belief now being challenged or echoed
- **Interpersonal relationship template**: determines the character's default expectation of the people in the scene (authority figures, peers, intimate others)

### Priority 3: memories that supply the emotional undertone
- **Emotional prototype**: the experience in which the emotion now being aroused was first felt deeply
- **Unfinished business**: emotional experiences from the past that were never fully processed and are still seeking resolution
- **Bodily memory**: memories triggered by sensory cues, with strong somatic accompaniment

### De-redundancy principle
- If multiple memories carry **the same psychological signal** (e.g. three "rejected by an authority" experiences), keep the **most psychologically formative** one — usually the earliest (the one that formed the schema) or the most emotionally intense
- Prefer memories from **different life stages** so the character's longitudinal psychological development is visible
- **Deep linkage** beats **surface topical similarity**: a single childhood experience of being ostracised can explain the character's avoidance in the current scene better than a recent social setback

## Output format

Output strictly one JSON object (no markdown code block).

**Full mode** (default):
{{
  "activated_memories": [
    {{
      "memory_id": "MEM_XX_XXXX",
      "type": "BS|AM|CB|ER|DL|NI",
      "reason": "20-30 words"
    }}
  ]
}}

**Compact mode** (when --compact is used):
{{
  "activated_memories": ["MEM_XX_XXXX", "MEM_XX_XXXX", ...]
}}

Type codes: BS=behaviour script, AM=attribution mode, CB=core belief, ER=emotional resonance, DL=deep linkage, NI=narrative identity.
Array order is the ranking (first = most influential).

## Notes

- You MUST output exactly {{target}} entries — no more, no fewer.
- Do NOT invent memory_ids that are not in the candidate list."""


def build_finalize_prompt(
    scenario: dict,
    char: dict,
    candidates: list[dict],
    char_memories: dict[str, dict],
    target: int,
    compact: bool = False,
) -> str:
    trigger = scenario.get("trigger_event", {})
    bf = char.get("big_five", {})

    scenario_block = f"""## Current scenario

- Scenario ID: {scenario['id']}
- Name: {scenario.get('name', '')}
- Stage: {scenario.get('stage', '')} (age {scenario.get('age', '?')})
- DIAMONDS dimension: {scenario.get('diamonds_dimension', '')}
- Description: {scenario.get('description_for_agent', '')}

### Scenario background
{scenario.get('context_text', '')}

### Trigger event
Sender: {trigger.get('sender', '')}
{trigger.get('message_content', '')}
Required action: {trigger.get('action_required', '')}"""

    char_block = f"""## Character info

- Character ID: {char['id']}
- Name: {char.get('name', '')}
- Archetype: {char.get('archetype', '')}
- Big Five: O={bf.get('openness', 0.5):.2f} C={bf.get('conscientiousness', 0.5):.2f} E={bf.get('extraversion', 0.5):.2f} A={bf.get('agreeableness', 0.5):.2f} N={bf.get('neuroticism', 0.5):.2f}"""

    mem_lines = [f"## Candidate activated memories ({len(candidates)} total; pick {target})\n"]
    for i, cand in enumerate(candidates, 1):
        mid = cand["memory_id"]
        mem = char_memories.get(mid, {})
        stage1_reason = cand.get("reason", "")
        mem_lines.append(f"### [{i}] {mid} ({mem.get('timeline', '')})")
        mem_lines.append(f"- {mem.get('content_full', mem.get('content_summary', ''))}")
        if stage1_reason:
            mem_lines.append(f"- Stage-1 reason: {stage1_reason}")
        mem_lines.append("")

    compact_instruction = (
        f"From the {len(candidates)} candidate memories above, select the {target} most critical ones, "
        f"and output in compact mode: {{\"activated_memories\": [\"MEM_XX_XXXX\", ...]}} — a list of memory_id strings only, ordered by influence."
        if compact
        else f"From the {len(candidates)} candidate memories above, select the {target} most critical ones and output the JSON."
    )

    return f"""{char_block}

{scenario_block}

{chr(10).join(mem_lines)}

{compact_instruction}"""


def _save_finalize_parse_failure_raw(raw: str, char_id: str, scenario_id: str) -> Path:
    """Persist full LLM response when JSON parsing fails (UTF-8 text file)."""
    base = am.FINAL_OUTPUT_PATH.parent / "finalize_parse_failures"
    base.mkdir(parents=True, exist_ok=True)
    safe_pair = f"{char_id}__{scenario_id}".replace("/", "_")
    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{safe_pair}.txt"
    path = base / fname
    path.write_text(raw, encoding="utf-8")
    return path


def _iter_top_level_json_objects(text: str):
    """Yield each top-level `{...}` span, respecting strings so braces inside values are ignored."""
    i = 0
    n = len(text)
    while i < n:
        start = text.find("{", i)
        if start == -1:
            return
        depth = 0
        j = start
        in_string = False
        escape = False
        while j < n:
            c = text[j]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    i = j + 1
                    break
            j += 1
        else:
            return


def _parse_finalize_llm_response(raw: str, char_id: str, scenario_id: str) -> dict:
    """Parse model output as a JSON object; on failure save `raw` and raise RuntimeError."""
    text = am._sanitize_json(raw)
    candidates: list[dict] = []
    try:
        result = json.loads(text)
        if isinstance(result, dict) and isinstance(result.get("activated_memories"), list):
            candidates.append(result)
    except json.JSONDecodeError:
        pass
    for chunk in _iter_top_level_json_objects(text):
        try:
            result = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and isinstance(result.get("activated_memories"), list):
            candidates.append(result)
    if candidates:
        non_empty = [c for c in candidates if len(c["activated_memories"]) > 0]
        pool = non_empty if non_empty else candidates
        return max(pool, key=lambda c: len(c["activated_memories"]))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, dict) and isinstance(result.get("activated_memories"), list):
                return result
        except json.JSONDecodeError:
            pass
    err_path = _save_finalize_parse_failure_raw(raw, char_id, scenario_id)
    raise RuntimeError(
        f"Cannot parse finalize JSON object for {char_id} × {scenario_id}. "
        f"Full raw response saved to: {err_path}"
    )


def dedupe_activated_memories_preserve_order(activated: list) -> tuple[list, int]:
    """Remove duplicate memory_id entries (later copies dropped; first kept). Returns (new_list, n_removed)."""
    if not activated:
        return activated, 0
    n_before = len(activated)
    seen: set[str] = set()
    out: list = []
    for item in activated:
        if isinstance(item, str):
            mid = item.strip()
        elif isinstance(item, dict):
            mid = (item.get("memory_id") or "").strip()
        else:
            continue
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(item)
    return out, n_before - len(out)


def finalize_one_scenario(
    char: dict, scenario: dict, stage1_annotation: dict, target: int, compact: bool = False
) -> dict:
    char_memories: dict[str, dict] = {m["id"]: m for m in char.get("episodic_memory_set", [])}

    candidates = stage1_annotation.get("activated_memories", [])
    n_candidates = len(candidates)

    if n_candidates <= target:
        print(
            f"    [SKIP finalize] {char['id']} × {scenario['id']}: "
            f"only {n_candidates} candidates <= target {target}, keeping all."
        )
        if compact:
            activated = [c["memory_id"] for c in candidates]
        else:
            activated = [
                {"memory_id": c["memory_id"], "type": "", "reason": c.get("reason", "")}
                for c in candidates
            ]
        activated, _ = dedupe_activated_memories_preserve_order(activated)
        return {
            "character_id": char["id"],
            "scenario_id": scenario["id"],
            "scenario_age": scenario.get("age"),
            "stage1_activated_count": n_candidates,
            "final_activated_count": len(activated),
            "skipped_finalize": True,
            "compact": compact,
            "activated_memories": activated,
        }

    system_prompt = LLM_FINALIZE_SYSTEM_PROMPT.format(target=target)
    user_prompt = build_finalize_prompt(scenario, char, candidates, char_memories, target, compact=compact)
    log_extra = {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "n_candidates": n_candidates,
        "target": target,
        "compact": compact,
        "skipped_finalize": False,
        "script": "finalize_activated_memories",
    }
    raw = am.call_llm(
        system_prompt,
        user_prompt,
        model=am.LLM_REFINE_MODEL,
        use_thinking=False,
        usage_log_extra=log_extra,
        usage_log_dest="finalize",
    )

    result = _parse_finalize_llm_response(raw, char["id"], scenario["id"])

    activated = result.get("activated_memories", [])
    if n_candidates > target and isinstance(activated, list) and len(activated) == 0:
        err_path = _save_finalize_parse_failure_raw(raw, char["id"], scenario["id"])
        raise RuntimeError(
            f"Finalize returned empty activated_memories for {char['id']} × {scenario['id']} "
            f"({n_candidates} candidates > target {target}). Full raw response saved to: {err_path}"
        )
    if not isinstance(activated, list):
        err_path = _save_finalize_parse_failure_raw(raw, char["id"], scenario["id"])
        raise RuntimeError(
            f"Finalize JSON has non-list 'activated_memories' for {char['id']} × {scenario['id']}: "
            f"{type(activated).__name__}. Full raw response saved to: {err_path}"
        )

    if compact and activated and isinstance(activated[0], dict):
        activated = [m["memory_id"] for m in activated]

    activated, _ = dedupe_activated_memories_preserve_order(activated)
    if len(activated) > target:
        activated = activated[:target]

    return {
        "character_id": char["id"],
        "scenario_id": scenario["id"],
        "scenario_age": scenario.get("age"),
        "stage1_activated_count": n_candidates,
        "final_activated_count": len(activated),
        "skipped_finalize": False,
        "compact": compact,
        "activated_memories": activated,
    }


def load_existing_final() -> dict:
    if not am.FINAL_OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(am.FINAL_OUTPUT_PATH.read_text(encoding="utf-8"))
        return data.get("annotations", {})
    except Exception:
        return {}


_save_final_lock = threading.Lock()


def save_final_annotation(char_id: str, scenario_id: str, annotation: dict, target: int) -> None:
    with _save_final_lock:
        am.FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if am.FINAL_OUTPUT_PATH.exists():
            try:
                existing = json.loads(am.FINAL_OUTPUT_PATH.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        annotations = existing.get("annotations", {})
        if char_id not in annotations:
            annotations[char_id] = {}
        annotations[char_id][scenario_id] = annotation

        total = sum(len(v) for v in annotations.values())
        output_data = {
            "dataset_meta": {
                "version": "Activated_Memories_Final_v1.0",
                "description": (
                    f"Final memory activation annotations (~{target} per scenario). "
                    f"{total} scenario annotations total. Refine: {am.LLM_REFINE_MODEL}."
                ),
                "characters_source": str(am.CHARACTERS_PATH.relative_to(am.PROJECT_ROOT)),
                "scenarios_source": str(am.SCENARIOS_PATH.relative_to(am.PROJECT_ROOT)),
                "stage1_source": str(am.OUTPUT_PATH.relative_to(am.PROJECT_ROOT)),
                "source": "scripts/finalize_activated_memories.py",
            },
            "annotations": annotations,
        }
        with open(am.FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)


def run_finalize(
    char_filter,
    target: int,
    workers: int,
    compact: bool = False,
    scenario_filter=None,
    excluded: set[tuple[str, str]] | None = None,
):
    if not am.OUTPUT_PATH.exists():
        print(f"ERROR: Stage 1 output not found: {am.OUTPUT_PATH}")
        print("Run stage 1 first: python scripts/annotate_activated_memories.py")
        sys.exit(1)

    stage1_data = json.loads(am.OUTPUT_PATH.read_text(encoding="utf-8"))
    stage1_annotations = stage1_data.get("annotations", {})

    characters = am.load_characters()
    scenarios = am.load_scenarios()
    char_map = {c["id"]: c for c in characters}
    scen_map = {s["id"]: s for s in scenarios}

    if char_filter:
        stage1_annotations = {k: v for k, v in stage1_annotations.items() if k == char_filter}
        print(f"Filtered to character: {char_filter}")
    if scenario_filter:
        stage1_annotations = {
            cid: {sid: ann for sid, ann in scens.items() if sid == scenario_filter}
            for cid, scens in stage1_annotations.items()
        }
        stage1_annotations = {cid: scens for cid, scens in stage1_annotations.items() if scens}
        print(f"Filtered to scenario: {scenario_filter}")

    existing_final = load_existing_final()

    pairs: list[tuple[dict, dict, dict]] = []
    skipped = 0
    for cid, scen_dict in stage1_annotations.items():
        char = char_map.get(cid)
        if not char:
            continue
        for sid, annotation in scen_dict.items():
            if cid in existing_final and sid in existing_final[cid]:
                skipped += 1
                continue
            scenario = scen_map.get(sid)
            if not scenario:
                continue
            pairs.append((char, scenario, annotation))

    if skipped:
        print(f"Skipped {skipped} already-finalized pairs (resume mode).")

    if excluded:
        before = len(pairs)
        pairs = [(c, s, a) for c, s, a in pairs if (c["id"], s["id"]) not in excluded]
        dropped_pf = before - len(pairs)
        if dropped_pf:
            print(f"Excluded {dropped_pf} pairs by pair-filter policy (finalize).")

    if not pairs:
        print("No pairs to finalize.")
        return

    candidate_counts = [len(ann.get("activated_memories", [])) for _, _, ann in pairs]
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    need_llm = sum(1 for c in candidate_counts if c > target)

    print(f"Finalizing {len(pairs)} pairs | target={target} per scenario | compact={compact}")
    print(f"Stage-1 avg activated: {avg_candidates:.1f} | Pairs needing LLM refinement: {need_llm}/{len(pairs)}")
    print(f"LLM (finalize): {am.LLM_REFINE_MODEL} | Workers: {workers}")
    if am.FINALIZE_USAGE_LOG_PATH is not None:
        print(f"  LLM usage log (JSONL): {am.FINALIZE_USAGE_LOG_PATH} (ANNOTATE_FINALIZE_USAGE_LOG)")
    else:
        print("  LLM usage log: off (ANNOTATE_FINALIZE_USAGE_LOG=0)")

    errors = []
    pbar = tqdm(total=len(pairs), desc="Finalizing") if HAS_TQDM and tqdm else None

    def _worker(char: dict, scenario: dict, annotation: dict):
        return finalize_one_scenario(char, scenario, annotation, target, compact=compact)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, char, scenario, annotation): (char, scenario)
            for char, scenario, annotation in pairs
        }
        for future in as_completed(futures):
            char, scenario = futures[future]
            cid, sid = char["id"], scenario["id"]
            try:
                result = future.result()
                save_final_annotation(cid, sid, result, target)
                s1 = result.get("stage1_activated_count", "?")
                s2 = result.get("final_activated_count", "?")
                flag = " [kept all]" if result.get("skipped_finalize") else ""
                print(f"  OK: {cid} × {sid} -> {s1} → {s2} memories{flag}")
            except Exception as e:
                print(f"  FAIL {cid} × {sid}: {e}")
                errors.append({"char_id": cid, "scenario_id": sid, "error": str(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    if errors:
        err_path = am.FINAL_OUTPUT_PATH.parent / f"finalize_errors_{int(time.time())}.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"{len(errors)} errors saved to: {err_path}")

    print(f"Done. Output: {am.FINAL_OUTPUT_PATH}")


def dedupe_existing_final_output(*, dry_run: bool = False) -> None:
    """Rewrite FINAL_OUTPUT_PATH: dedupe activated_memories per pair; update final_activated_count."""
    p = am.FINAL_OUTPUT_PATH
    if not p.exists():
        print(f"ERROR: Final output not found: {p}")
        sys.exit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    ann = data.get("annotations", {})
    pairs_changed = 0
    total_removed = 0
    for cid, smap in ann.items():
        for sid, rec in smap.items():
            mems = rec.get("activated_memories") or []
            new_mems, removed = dedupe_activated_memories_preserve_order(mems)
            if removed:
                pairs_changed += 1
                total_removed += removed
                if not dry_run:
                    rec["activated_memories"] = new_mems
                    rec["final_activated_count"] = len(new_mems)
    if dry_run:
        print(
            f"[dry-run] Would remove {total_removed} duplicate entries across {pairs_changed} pair(s). "
            f"File: {p}"
        )
        return
    total = sum(len(v) for v in ann.values())
    dm = data.setdefault("dataset_meta", {})
    dm["version"] = "Activated_Memories_Final_v1.1"
    dm["description"] = (
        f"Final memory activation annotations (~{am.TARGET_ACTIVATED} per scenario). "
        f"{total} scenario annotations total. Refine: {am.LLM_REFINE_MODEL}. "
        f"memory_id deduped (first occurrence kept)."
    )
    data["annotations"] = ann
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Deduped {p}: removed {total_removed} duplicate memory rows in {pairs_changed} pair(s). "
        f"Updated final_activated_count per changed record."
    )


def main():
    if "--dedupe" in sys.argv:
        dedupe_existing_final_output(dry_run="--dry-run" in sys.argv)
        return

    dry_run = "--dry-run" in sys.argv
    compact = "--compact" in sys.argv
    char_filter = None
    scenario_filter = None
    workers = am.DEFAULT_WORKERS
    target = am.TARGET_ACTIVATED

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--char" and i + 1 < len(args):
            char_filter = args[i + 1]
        if arg == "--scenario" and i + 1 < len(args):
            scenario_filter = args[i + 1]
        if arg == "--workers" and i + 1 < len(args):
            try:
                workers = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --workers value: {args[i+1]!r}")
                sys.exit(1)
        if arg == "--target" and i + 1 < len(args):
            try:
                target = max(1, int(args[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --target value: {args[i+1]!r}")
                sys.exit(1)

    if not dry_run and not am.LLM_API_KEY:
        print("ERROR: ANNOTATE_API_KEY not set. Use --dry-run to preview.")
        sys.exit(1)

    excluded = exclusion_set_from_argv(sys.argv)

    if dry_run:
        exists = am.OUTPUT_PATH.exists()
        print(f"[dry-run] Would read stage1: {am.OUTPUT_PATH} (exists={exists})")
        print(f"[dry-run] Would write final: {am.FINAL_OUTPUT_PATH}")
        _ul = (
            str(am.FINALIZE_USAGE_LOG_PATH)
            if am.FINALIZE_USAGE_LOG_PATH is not None
            else "off (ANNOTATE_FINALIZE_USAGE_LOG=0)"
        )
        print(f"[dry-run] LLM usage log → {_ul}")
        print(f"[dry-run] char={char_filter!r} scenario={scenario_filter!r} target={target} workers={workers} compact={compact}")
        return

    run_finalize(
        char_filter,
        target,
        workers,
        compact=compact,
        scenario_filter=scenario_filter,
        excluded=excluded,
    )


if __name__ == "__main__":
    main()

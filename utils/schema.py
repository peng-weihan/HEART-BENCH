"""Character / Scenario data classes and prompt builder.

Extracted from the legacy ``main.py`` so that downstream tools (experiments,
scripts) can depend on a small, stable surface without pulling in the full
benchmark runner.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional


class Character:
    """Lightweight wrapper around a character JSON record."""

    def __init__(self, data: Mapping[str, Any]):
        self.id: str = data["id"]
        self.name: str = data.get("name", "")
        self.description: str = data.get("description", "")
        self.big_five: Mapping[str, Any] = data.get("big_five", {})
        self.value_orientation: Mapping[str, Any] = data.get("value_orientation", {})
        self.self_value_logic: str = data.get("self_value_logic", "")
        self.semantic_memory: Mapping[str, Any] = data.get("semantic_memory", {})
        self.episodic_memory: list = data.get("episodic_memory_set", [])

    def retrieve_memories(
        self,
        context: str,
        top_k: int = 3,
        memory_index=None,
        stage: Optional[str] = None,
    ):
        if memory_index is None:
            raise RuntimeError(
                "memory_index is required. Set EMBEDDING_API_KEY in .env."
            )
        return memory_index.query(self.id, context, top_k=top_k, stage=stage)


class Scenario:
    """Lightweight wrapper around a scenario JSON record."""

    def __init__(self, data: Mapping[str, Any]):
        self.id: str = data["id"]
        self.name: str = data.get("name", "")
        self.category: str = data.get("category", "")
        self.stage: str = data.get("stage", "")
        self.age = data.get("age")
        self.context: str = data["context_text"]
        self.trigger: Mapping[str, Any] = data["trigger_event"]
        self.setting: Mapping[str, Any] = data.get("setting", {})
        self.assessed_dimensions = data.get(
            "assessed_dimensions", data.get("stress_factors", {})
        )


def build_prompt(
    char: Character,
    scenario: Scenario,
    retrieved_memories: Iterable[Mapping[str, Any]],
    options_text: Optional[str] = None,
) -> str:
    """Build a role-play prompt from raw memories and scenario facts.

    No personality summaries are injected; the LLM must infer the persona
    purely from the retrieved episodic memories and semantic facts.
    """
    mem_str = "\n".join(
        f"  - [{m.get('id', '?')}][{m.get('timeline', '?')}] "
        f"{m.get('content_full', m.get('content_summary', ''))}"
        for m in retrieved_memories
    )

    relationships = char.semantic_memory.get("core_social_relationships", [])
    rel_str = (
        "\n".join(f"  - {r['target']}: {r['relation']}" for r in relationships)
        if relationships
        else "  N/A"
    )

    prompt = f"""## Background
- Capabilities: {char.semantic_memory.get('capabilities', 'N/A')}

## Key Social Relationships
{rel_str}

## Past Experiences
Below are key episodes from this person's life. Use them to understand who they are:
{mem_str}

## Current Situation
Scenario: {scenario.name}
Location: {scenario.setting.get('location', 'unknown')} | Time: {scenario.setting.get('time', 'unknown')} | Atmosphere: {scenario.setting.get('atmosphere', 'unknown')}

Context: {scenario.context}

## Trigger Event
Sender: {scenario.trigger.get('sender', 'unknown')}
Message: {scenario.trigger.get('message_content', 'unknown')}
Action required: {scenario.trigger.get('action_required', 'unknown')}

## Task
Using the episodes above, infer this person's thought patterns, emotional tendencies, and behavioural habits, then simulate how they would actually react in the current situation.

Requirements:
1. System 1 (gut impulse): the person's first reaction; cite the memories that were activated.
2. System 2 (rational analysis): how the person would analyse and reason once calmed down.
3. Final Decision: two parts — `inner_consciousness` is the inner monologue of "what I am about to do/say" (the last layer of consciousness before the outward act, fusing emotional tone, core reasoning, and value orientation); `response_text` is what the person actually says or sends."""

    if options_text:
        prompt += f"""

## Behavioural Decision Options
Below are possible behavioural decisions different characters might make in this scenario. Pick the one that best fits you (as this character) and output the corresponding letter in the `decision_choice` field:

{options_text}"""

    return prompt


__all__ = ["Character", "Scenario", "build_prompt"]


# Pure event-age and activation-reference validation for new construction runs.

def parse_event_age(timeline: object) -> int | None:
    """Read one exact event age, never a range, upper bound, or approximation."""
    if not isinstance(timeline, str):
        return None
    text = timeline.strip()
    if re.search(
        r"\b(?:under|below|before|over|after|around|about|approximately|roughly|between|"
        r"younger|older|until|up\s+to|at\s+least|at\s+most|or\s+so|and\s+up)\b|"
        r"[<>≤≥~≈+]|未满|不满|超过|左右|以下|以上|之前|之后|约|(?<![A-Za-z])-\s*\d",
        text, re.IGNORECASE,
    ):
        return None
    # Multiple numbers also exclude ranges, dates, and fractional ages.
    if len(re.findall(r"[0-9]+", text)) != 1:
        return None
    patterns = (
        r"\bage\s*[:=]?\s*([0-9]+)\b",
        r"([0-9]+)\s*(?:years?\s*old\b|岁)",
        r"\bage\s*\(\s*([0-9]+)\s*\)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def require_scenario_age(value: object) -> int:
    """Require an explicit nonnegative integer scenario age."""
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f"Scenario age must be an exact nonnegative integer: {value!r}")


def _reference(item: object) -> dict | None:
    if isinstance(item, str):
        mid, record = item.strip(), {}
    elif isinstance(item, dict) and isinstance(item.get("memory_id"), str):
        mid, record = item["memory_id"].strip(), dict(item)
    else:
        return None
    if not mid:
        return None
    record["memory_id"] = mid
    return record


def sanitize_activation_candidates(
    references: list, character_memories: Mapping[str, dict], scenario_age: object,
) -> tuple[list[dict], list[dict]]:
    """Return eligible unique references plus explicit rejection diagnostics.

    ``character_memories`` must be the target character's own ID-to-record map.
    Inputs are never mutated; full and compact reference formats are accepted.
    """
    cutoff = require_scenario_age(scenario_age)
    if not isinstance(references, list):
        raise ValueError("Activated memory references must be a list")
    kept, rejected, seen = [], [], set()
    for position, item in enumerate(references):
        record = _reference(item)
        mid = record["memory_id"] if record else None
        memory = character_memories.get(mid) if mid is not None else None
        event_age = parse_event_age(memory.get("timeline")) if isinstance(memory, dict) else None
        if record is None:
            reason = "invalid_reference"
        elif not isinstance(memory, dict) or memory.get("id") != mid:
            reason = "unknown_or_other_character_memory"
        elif event_age is None:
            reason = "ambiguous_or_missing_event_age"
        elif event_age > cutoff:
            reason = "future_memory"
        elif mid in seen:
            reason = "duplicate_memory"
        else:
            kept.append(record)
            seen.add(mid)
            continue
        rejected.append({"position": position, "memory_id": mid, "reason": reason})
    return kept, rejected


def require_activation_references(
    references: list, character_memories: Mapping[str, dict], scenario_age: object,
) -> list[dict]:
    """Validate GT input without silently dropping or changing its evidence."""
    kept, rejected = sanitize_activation_candidates(references, character_memories, scenario_age)
    if rejected:
        details = "; ".join(f"{r['memory_id']!r}: {r['reason']}" for r in rejected)
        raise ValueError(f"Invalid activated memories: {details}")
    return kept


def validate_refined_selection(
    selected: list, candidates: list[dict], target: int, *, compact: bool = False,
) -> list:
    """Require candidate membership before deduplicating and applying an upper bound."""
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("Refinement target must be a positive integer")
    if not isinstance(selected, list):
        raise ValueError("Refined memory selection must be a list")
    allowed = {candidate["memory_id"] for candidate in candidates}
    normalized, seen = [], set()
    for item in selected:
        record = _reference(item)
        if record is None or record["memory_id"] not in allowed:
            raise ValueError(f"Refinement returned a noncandidate memory: {item!r}")
        mid = record["memory_id"]
        if mid not in seen:
            seen.add(mid)
            normalized.append(mid if compact else record)
    return normalized[:target]

"""Character / Scenario data classes and prompt builder.

Extracted from the legacy ``main.py`` so that downstream tools (experiments,
scripts) can depend on a small, stable surface without pulling in the full
benchmark runner.
"""
from __future__ import annotations

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

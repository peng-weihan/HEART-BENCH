"""Validate the current Big Five character/memory bundle without changing it."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.schema import parse_event_age

BIG_FIVE = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
PROFILE_TEXT = ("id", "name", "occupation", "description", "self_value_logic")
MEMORY_TEXT = ("id", "timeline", "context", "content_summary", "content_full", "psych_conclusion", "behavior_policy")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unit_score(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and 0 <= value <= 1 and math.isfinite(value)
    )


def validate_characters(data: object, *, expected_characters: int = 11,
                        memories_per_character: int = 1000) -> list[str]:
    """Return schema/count/range errors; drafting lengths and tag counts are guidance."""
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        return ["characters: expected an array in the top-level object"]
    characters = data["characters"]
    errors: list[str] = []
    if len(characters) != expected_characters:
        errors.append(f"characters: expected {expected_characters}, found {len(characters)}")
    character_ids: set[str] = set()
    memory_ids: set[str] = set()

    for index, char in enumerate(characters):
        where = f"characters[{index}]"
        if not isinstance(char, dict):
            errors.append(f"{where}: expected an object")
            continue
        for field in PROFILE_TEXT:
            if not _text(char.get(field)):
                errors.append(f"{where}.{field}: expected nonempty text")
        cid = char.get("id")
        if _text(cid):
            if cid in character_ids:
                errors.append(f"{where}.id: duplicate {cid!r}")
            character_ids.add(cid)
            where = cid
        scores = char.get("big_five")
        if not isinstance(scores, dict):
            errors.append(f"{where}.big_five: expected an object")
        else:
            if set(scores) != set(BIG_FIVE):
                errors.append(f"{where}.big_five: expected exactly the five OCEAN keys")
            for trait in BIG_FIVE:
                if not _unit_score(scores.get(trait)):
                    errors.append(f"{where}.big_five.{trait}: expected a finite number in [0, 1]")
        patterns = char.get("core_patterns")
        if not isinstance(patterns, list) or len(patterns) != 8 or not all(_text(p) for p in patterns):
            errors.append(f"{where}.core_patterns: expected eight nonempty strings")
        memories = char.get("episodic_memory_set")
        if not isinstance(memories, list):
            errors.append(f"{where}.episodic_memory_set: expected an array")
            continue
        if len(memories) != memories_per_character:
            errors.append(f"{where}.episodic_memory_set: expected {memories_per_character}, found {len(memories)}")
        for memory_index, memory in enumerate(memories):
            location = f"{where}.episodic_memory_set[{memory_index}]"
            if not isinstance(memory, dict):
                errors.append(f"{location}: expected an object")
                continue
            for field in MEMORY_TEXT:
                if not _text(memory.get(field)):
                    errors.append(f"{location}.{field}: expected nonempty text")
            mid = memory.get("id")
            if _text(mid):
                if mid in memory_ids:
                    errors.append(f"{location}.id: duplicate {mid!r}")
                memory_ids.add(mid)
            event_age = parse_event_age(memory.get("timeline"))
            if event_age is None or not 6 <= event_age <= 50:
                errors.append(f"{location}.timeline: expected one event age in [6, 50]")
            for field in ("triggers", "relevance_tags"):
                values = memory.get(field)
                if not isinstance(values, list) or not all(_text(v) for v in values):
                    errors.append(f"{location}.{field}: expected an array of nonempty strings")
            emotion = memory.get("emotion_signature")
            if not isinstance(emotion, dict):
                errors.append(f"{location}.emotion_signature: expected an object")
            else:
                for field in ("primary", "secondary"):
                    if not _text(emotion.get(field)):
                        errors.append(f"{location}.emotion_signature.{field}: expected nonempty text")
                if not _unit_score(emotion.get("intensity")):
                    errors.append(f"{location}.emotion_signature.intensity: expected a finite number in [0, 1]")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "benchmark" / "characters.json")
    parser.add_argument("--expected-characters", type=int, default=11)
    parser.add_argument("--memories-per-character", type=int, default=1000)
    args = parser.parse_args()
    if args.expected_characters < 1 or args.memories_per_character < 1:
        parser.error("expected counts must be positive")
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    errors = validate_characters(data, expected_characters=args.expected_characters,
                                 memories_per_character=args.memories_per_character)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print(f"Overall: {len(errors)} issue(s)")
        return 1
    print(f"PASS: {len(data['characters'])} characters, "
          f"{args.memories_per_character} memories each; schema, unique IDs, and ranges valid")
    print("Text lengths, trigger/tag counts, and the non-anchor 0.30-0.70 range are generation guidance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

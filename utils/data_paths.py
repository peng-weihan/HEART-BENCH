"""Resolve default paths for benchmark inputs (characters, scenarios, GT)."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_gt_annotations_path(data_dir: Path) -> Path | None:
    """
    Ground-truth JSON for multiple-choice options (final_decision per character).

    Order:
      1. GT_ANNOTATIONS_PATH env (absolute or relative file path)
      2. <data_dir>/ground_truth.json
    """
    env = os.getenv("GT_ANNOTATIONS_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    p = data_dir / "ground_truth.json"
    return p if p.is_file() else None


def resolve_mc_questions_path(data_dir: Path) -> Path | None:
    """
    Multiple-choice benchmark JSON.

    Order:
      1. MC_QUESTIONS_PATH env (absolute or relative file path)
      2. <data_dir>/mcq.json
    """
    env = os.getenv("MC_QUESTIONS_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    p = data_dir / "mcq.json"
    return p if p.is_file() else None


def resolve_activated_memories_path(data_dir: Path) -> Path | None:
    """
    Pre-annotated memory activation JSON (step-2 finalised candidates).

    Order:
      1. ACTIVATED_MEMORIES_PATH env (absolute or relative file path)
      2. <data_dir>/activated_memories_step2.json
      3. <data_dir>/activated_memories_step1.json
    """
    env = os.getenv("ACTIVATED_MEMORIES_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    for name in ("activated_memories_step2.json", "activated_memories_step1.json"):
        p = data_dir / name
        if p.is_file():
            return p
    return None


def activated_top5_memory_ids(annotation: dict | None) -> list[str]:
    """Return up to 5 memory ids for the benchmark prompt (preserves file order)."""
    if not annotation or not isinstance(annotation, dict):
        return []
    cands = annotation.get("candidate_memory_ids")
    if isinstance(cands, list) and cands and isinstance(cands[0], str):
        return [x for x in cands if isinstance(x, str)][:5]
    am = annotation.get("activated_memories") or []
    if not isinstance(am, list):
        return []
    out: list[str] = []
    for x in am:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            mid = x.get("memory_id")
            if isinstance(mid, str) and mid:
                out.append(mid)
        if len(out) >= 5:
            break
    return out


def resolve_characters_json(data_dir: Path) -> Path | None:
    """
    Character bundle (episodic_memory_set, semantic_memory, ...).

    Order:
      1. CHARACTERS_JSON_PATH env
      2. <data_dir>/characters.json
    """
    env = os.getenv("CHARACTERS_JSON_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    p = data_dir / "characters.json"
    return p if p.is_file() else None


def resolve_scenarios_json(data_dir: Path) -> Path | None:
    """
    Scenario list.

    Order:
      1. SCENARIOS_JSON_PATH env
      2. <data_dir>/scenarios.json
    """
    env = os.getenv("SCENARIOS_JSON_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    p = data_dir / "scenarios.json"
    return p if p.is_file() else None

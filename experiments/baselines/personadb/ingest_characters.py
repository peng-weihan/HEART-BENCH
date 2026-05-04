"""
ingest_characters.py — CLI to pre-ingest all characters into PersonaDB.

Usage:
    python -m personadb_integration.ingest_characters
    python -m personadb_integration.ingest_characters --reset
    PERSONADB_CONDENSE=1 python -m personadb_integration.ingest_characters

This only needs to be run once (or after the character data changes).
The store lives in-process; re-ingesting is fast (no LLM unless PERSONADB_CONDENSE=1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import os

_ENV_PATH = _REPO_ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

from utils.data_paths import resolve_characters_json  # type: ignore
from personadb_integration.benchmark import (
    make_personadb,
    ingest_character_episodes_personadb,
)


def main() -> None:
    reset = "--reset" in sys.argv

    data_dir = _REPO_ROOT / "data"
    chars_path = resolve_characters_json(data_dir)
    if chars_path is None:
        print("ERROR: characters JSON not found.", file=sys.stderr)
        sys.exit(1)

    with open(chars_path, "r", encoding="utf-8-sig") as f:
        chars_data = json.load(f)["characters"]

    cfg = make_personadb()
    condense_note = " (with L1 condensation)" if cfg.condense else " (L0 only)"
    print(f"PersonaDB ingest{condense_note}")
    print(f"Characters file: {chars_path}")
    print(f"Characters: {len(chars_data)}")
    print()

    total_ingested = 0
    for char_data in chars_data:
        cid = char_data.get("id", "")
        episodes = char_data.get("episodic_memory_set", []) or []
        n = ingest_character_episodes_personadb(
            cfg,
            cid,
            episodes,
            reset_user=reset,
            show_progress=True,
        )
        print(f"  {cid}: {n} entries ingested (store total: {cfg.store.count(cid)})")
        total_ingested += n

    print(f"\nDone. Total entries ingested: {total_ingested}")


if __name__ == "__main__":
    main()

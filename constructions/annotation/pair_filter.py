"""
Pair filter for annotation / benchmark runs.

By default, loads demo_pair_filter.json and excludes listed (character, scenario) pairs.
Use --no-pair-filter on CLI-invoked scripts to process the full cross product.

Usage:
  python scripts/pair_filter.py
  python scripts/pair_filter.py --pair-filter path/to/policy.json --strict --exclude-review
  # Scripts: default filter on; add --no-pair-filter to disable

Call exclusion_set_from_argv(sys.argv) or apply_argv_pair_filter(sys.argv, pairs) from other scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "benchmark" / "annotations" / "conflicts" / "demo_pair_filter.json"


def _pairs_from_entries(entries: list | None) -> set[tuple[str, str]]:
    if not entries:
        return set()
    out: set[tuple[str, str]] = set()
    for item in entries:
        cid = item.get("character_id")
        sid = item.get("scenario_id")
        if cid and sid:
            out.add((cid, sid))
    return out


def load_policy(path: Path | str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Pair filter policy not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_exclusion_set(
    policy_path: Path | str | None = None,
    *,
    include_optional_block: bool = False,
    exclude_review: bool = False,
) -> set[tuple[str, str]]:
    """
    Build the set of (character_id, scenario_id) to skip.

    - Always includes `block` from policy.
    - Adds `optional_block` if include_optional_block.
    - Adds `review` if exclude_review.
    """
    path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
    data = load_policy(path)
    excluded = _pairs_from_entries(data.get("block"))
    if include_optional_block:
        excluded |= _pairs_from_entries(data.get("optional_block"))
    if exclude_review:
        excluded |= _pairs_from_entries(data.get("review"))
    return excluded


def pair_is_allowed(
    character_id: str,
    scenario_id: str,
    excluded: set[tuple[str, str]],
) -> bool:
    return (character_id, scenario_id) not in excluded


def filter_cross_product_pairs(
    pairs: list[tuple[dict, dict]],
    excluded: set[tuple[str, str]],
) -> tuple[list[tuple[dict, dict]], int]:
    """Drop pairs whose (char_id, scenario_id) is in excluded. Returns (kept, n_dropped)."""
    kept: list[tuple[dict, dict]] = []
    dropped = 0
    for char, scenario in pairs:
        cid, sid = char["id"], scenario["id"]
        if (cid, sid) in excluded:
            dropped += 1
            continue
        kept.append((char, scenario))
    return kept, dropped


def count_cross_product(characters: list[dict], scenarios: list[dict]) -> int:
    return len(characters) * len(scenarios)


def parse_filter_cli_args(argv: list[str]) -> tuple[Path | None, bool, bool]:
    """
    Parse pair-filter flags from argv.

    Returns (explicit_policy_path_or_none, strict, exclude_review).
    explicit path is set only if --pair-filter PATH is present (not the default-on path).
    """
    policy_path: Path | None = None
    strict = False
    exclude_review = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--pair-filter" and i + 1 < len(argv):
            policy_path = Path(argv[i + 1])
            i += 2
            continue
        if arg == "--pair-filter-strict":
            strict = True
            i += 1
            continue
        if arg == "--pair-filter-exclude-review":
            exclude_review = True
            i += 1
            continue
        i += 1
    return policy_path, strict, exclude_review


def exclusion_set_from_argv(argv: list[str]) -> set[tuple[str, str]]:
    """
    Excluded (char_id, scenario_id) set for the current CLI invocation.

    - With --no-pair-filter: empty set (no exclusions).
    - Otherwise: load DEFAULT_POLICY_PATH, or --pair-filter PATH if given.
    - Modifiers: --pair-filter-strict, --pair-filter-exclude-review
    """
    if "--no-pair-filter" in argv:
        return set()
    explicit_path, strict, exclude_review = parse_filter_cli_args(argv)
    policy_path = explicit_path if explicit_path is not None else DEFAULT_POLICY_PATH
    return load_exclusion_set(
        policy_path,
        include_optional_block=strict,
        exclude_review=exclude_review,
    )


def apply_argv_pair_filter(
    argv: list[str],
    pairs: list[tuple[dict, dict]],
) -> tuple[list[tuple[dict, dict]], int]:
    """
    Apply pair filter from argv (default policy unless --no-pair-filter).
    Returns (pairs_out, n_excluded_by_policy).
    """
    excluded = exclusion_set_from_argv(argv)
    if not excluded:
        return pairs, 0
    kept, dropped = filter_cross_product_pairs(pairs, excluded)
    return kept, dropped


def _stats_main():
    parser = argparse.ArgumentParser(description="Print pair-filter coverage stats.")
    parser.add_argument(
        "--no-pair-filter",
        action="store_true",
        help="Show stats as if filtering were off (0 exclusions).",
    )
    parser.add_argument(
        "--pair-filter",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Policy JSON path (default: demo_pair_filter.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Include optional_block in exclusion set.",
    )
    parser.add_argument(
        "--exclude-review",
        action="store_true",
        help="Also exclude review pairs.",
    )
    parser.add_argument(
        "--characters",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "characters" / "characters_phase11.json",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "scenarios" / "scenarios_diamonds_zh_8x24_lite.json",
    )
    args = parser.parse_args()

    chars_data = json.loads(args.characters.read_text(encoding="utf-8"))
    characters = chars_data.get("characters", [])
    scen_raw = json.loads(args.scenarios.read_text(encoding="utf-8"))
    sc = scen_raw.get("scenarios", {})
    if isinstance(sc, dict):
        scenarios = [s for stage_list in sc.values() for s in stage_list]
    else:
        scenarios = sc

    total = count_cross_product(characters, scenarios)
    if args.no_pair_filter:
        ex = set()
        print("Policy: (disabled via --no-pair-filter)")
    else:
        ex = load_exclusion_set(
            args.pair_filter,
            include_optional_block=args.strict,
            exclude_review=args.exclude_review,
        )
        print(f"Policy: {args.pair_filter}")
    allowed = total - len(ex)
    print(f"Characters: {len(characters)} | Scenarios: {len(scenarios)} | Cross product: {total}")
    print(f"Excluded pairs: {len(ex)} | Allowed pairs: {allowed}")
    if args.strict:
        print("(strict: optional_block included)")
    if args.exclude_review:
        print("(exclude-review: review pairs also excluded)")


if __name__ == "__main__":
    _stats_main()

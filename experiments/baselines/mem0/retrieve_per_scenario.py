"""Retrieve top-K memories per (character, scenario) from age-matched snapshots.

For each scenario:
  - Determine target_age = scenario["age"] (exact match required).
  - For each character, load .mem0/snapshots/CHAR_XX/age_NN.jsonl into an
    isolated restored qdrant collection (cached: only loaded once per CHAR×age).
  - Build a full-context query string from the scenario fields.
  - Search top_k memories with the live mem0 search() and write one JSONL row.

Output:
  .mem0/retrieval/all.jsonl
  each line:
    {
      "char_id": "CHAR_01",
      "scenario_id": "SCN_SCHOOL_AGE_1",
      "stage": "school_age",
      "scenario_age": 9,
      "snapshot_age": 9,
      "query_text": "...",
      "top_k": 50,
      "results": [
        {
          "rank": 1,
          "score": 0.612,
          "memory": "...",
          "mem_id": "MEM_0123",
          "timeline": "...",
          "metadata": {...full payload metadata...}
        },
        ...
      ]
    }

CLI:
  python retrieve_per_scenario.py                       # all chars, all scenarios
  python retrieve_per_scenario.py --chars CHAR_01,CHAR_02
  python retrieve_per_scenario.py --top-k 50 --out .mem0/retrieval/all.jsonl
  python retrieve_per_scenario.py --dry-run             # print plan only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Disable mem0 telemetry BEFORE importing mem0 — avoids the global
# ~/.mem0/migrations_qdrant lock that blocks creating multiple Memory
# instances sequentially.
os.environ["MEM0_TELEMETRY"] = "False"

ROOT = Path(__file__).parent

# Reuse env loading + restore() helper from restore_snapshot.py
sys.path.insert(0, str(ROOT))
from restore_snapshot import restore  # noqa: E402

SCENARIO_FILE = Path(
    "/Users/raymone/Desktop/human-like/annotations/scenarios_diamonds_zh_8x24_lite.json"
)
DEFAULT_OUT = ROOT / ".mem0" / "retrieval" / "all.jsonl"
ALL_CHARS = [f"CHAR_{i:02d}" for i in range(1, 12)]


def build_query_text(scn: dict) -> str:
    """Concatenate ALL contextual fields for the embed query."""
    setting = scn.get("setting") or {}
    trig = scn.get("trigger_event") or {}
    parts = [
        f"[Scenario] {scn.get('name','')}",
        f"[Dimension] {scn.get('diamonds_dimension','')}  Category: {scn.get('category','')}  Intensity: {scn.get('intensity','')}",
        f"[Summary] {scn.get('description_for_agent','')}",
        f"[Setting] location: {setting.get('location','')}  time: {setting.get('time','')}  atmosphere: {setting.get('atmosphere','')}",
        f"[Context] {scn.get('context_text','')}",
        f"[Trigger] {trig.get('sender','')}: {trig.get('message_content','')}",
        f"[Action required] {trig.get('action_required','')}",
    ]
    return "\n".join(p for p in parts if p.strip())


def load_scenarios() -> list[dict]:
    data = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    out = []
    for stage, items in data["scenarios"].items():
        for it in items:
            out.append(
                {
                    "id": it["id"],
                    "stage": stage,
                    "age": int(it["age"]),
                    "raw": it,
                }
            )
    return out


def snapshot_exists(char: str, age: int) -> bool:
    return (ROOT / ".mem0" / "snapshots" / char / f"age_{age:02d}.jsonl").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chars", default=",".join(ALL_CHARS),
                    help="comma-separated char ids (default: all 11)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit, do not embed/search")
    args = ap.parse_args()

    chars = [c.strip() for c in args.chars.split(",") if c.strip()]
    scenarios = load_scenarios()
    print(f"chars       : {len(chars)}  ({chars[0]} .. {chars[-1]})")
    print(f"scenarios   : {len(scenarios)}")
    print(f"top_k       : {args.top_k}")
    print(f"out         : {args.out}")
    print(f"total tasks : {len(chars) * len(scenarios)}")

    # group scenarios by age so each (char, age) snapshot is loaded only once
    by_age: dict[int, list[dict]] = defaultdict(list)
    for s in scenarios:
        by_age[s["age"]].append(s)
    ages_sorted = sorted(by_age.keys())
    print(f"distinct ages: {len(ages_sorted)}  -> {ages_sorted}")

    # sanity: every char has every age
    missing = []
    for c in chars:
        for a in ages_sorted:
            if not snapshot_exists(c, a):
                missing.append((c, a))
    if missing:
        print(f"FATAL: missing snapshots: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        sys.exit(2)

    if args.dry_run:
        print("dry-run: plan looks good, exiting.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Wipe previous run output (we want a clean snapshot of this run)
    if args.out.exists():
        bak = args.out.with_suffix(args.out.suffix + ".bak")
        args.out.replace(bak)
        print(f"existing output backed up to: {bak}")

    f_out = args.out.open("w", encoding="utf-8")
    t0 = time.time()
    done = 0
    total = len(chars) * len(scenarios)
    fails: list[tuple[str, str, str]] = []

    try:
        for c in chars:
            for age in ages_sorted:
                # Load snapshot once per (char, age); restore() reuses if already populated
                try:
                    mem = restore(c, age)
                except SystemExit as e:
                    print(f"[{c} age {age}] restore failed: {e}")
                    for s in by_age[age]:
                        fails.append((c, s["id"], f"restore failed: {e}"))
                    continue

                for s in by_age[age]:
                    sid = s["id"]
                    q = build_query_text(s["raw"])
                    try:
                        r = mem.search(
                            query=q,
                            filters={"agent_id": c},
                            top_k=args.top_k,
                            threshold=0.0,
                        )
                    except Exception as e:
                        print(f"[{c} {sid}] search error: {e}")
                        fails.append((c, sid, f"search error: {e}"))
                        continue

                    results = []
                    for rank, item in enumerate(r.get("results", []), start=1):
                        md = item.get("metadata") or {}
                        results.append(
                            {
                                "rank": rank,
                                "score": item.get("score"),
                                "memory": item.get("memory"),
                                "mem_id": md.get("mem_id"),
                                "timeline": md.get("timeline"),
                            }
                        )

                    row = {
                        "char_id": c,
                        "scenario_id": sid,
                        "stage": s["stage"],
                        "scenario_age": s["age"],
                        "snapshot_age": age,
                        "query_text": q,
                        "top_k": args.top_k,
                        "n_returned": len(results),
                        "results": results,
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    done += 1
                    if done % 10 == 0 or done == total:
                        rate = done / max(time.time() - t0, 1e-3)
                        eta = (total - done) / max(rate, 1e-3)
                        print(f"  progress {done}/{total}  rate={rate:.2f}/s  eta={eta:.0f}s")

                # Release the qdrant file lock before moving to the next snapshot.
                # mem0 also opens an internal telemetry qdrant; closing the main
                # vector store + dropping the reference lets it be GC'd.
                try:
                    mem.vector_store.client.close()
                except Exception:
                    pass
                try:
                    if hasattr(mem, "_telemetry_vector_store") and mem._telemetry_vector_store:
                        mem._telemetry_vector_store.client.close()
                except Exception:
                    pass
                del mem
                import gc
                gc.collect()

    finally:
        f_out.close()

    elapsed = time.time() - t0
    print()
    print(f"done. {done}/{total} retrievals in {elapsed:.1f}s -> {args.out}")
    if fails:
        print(f"failures: {len(fails)}")
        for c, sid, why in fails[:20]:
            print(f"  {c} {sid}: {why}")


if __name__ == "__main__":
    main()

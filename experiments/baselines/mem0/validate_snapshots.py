"""Two-part validation:

1) Per-character isolation: a snapshot of CHAR_X contains ONLY CHAR_X memories.
   - Static: every payload.agent_id in age_NN.jsonl == CHAR_X
   - Dynamic: in the live main collection, filter by agent_id and confirm
     no cross-character contamination.

2) Per-age snapshot accuracy (CHAR_01 has 4 snapshots: 6,7,8,9):
   - id-set monotonically grows: ids(06) ⊆ ids(07) ⊆ ids(08) ⊆ ids(09)
   - max(payload.timeline parsed age) in age_NN.jsonl is ≤ NN
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SNAP_ROOT = ROOT / ".mem0" / "snapshots"

# load env so build_memory works
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Matches an age token like "age <N>" in dataset timeline strings.
AGE_RE = re.compile(r"age\s*(\d+)", re.IGNORECASE)


def parse_age(timeline: str) -> int | None:
    if not timeline:
        return None
    m = AGE_RE.search(timeline)
    return int(m.group(1)) if m else None


def read_snapshot(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# 1A) Static isolation: agent_id in payload matches folder
# --------------------------------------------------------------------------
def validate_static_isolation() -> bool:
    print("=" * 70)
    print("[1A] Static isolation — every payload.agent_id matches its folder")
    print("=" * 70)
    ok = True
    for char_dir in sorted(SNAP_ROOT.iterdir()):
        if not char_dir.is_dir():
            continue
        cid = char_dir.name
        for snap in sorted(char_dir.glob("age_*.jsonl")):
            rows = read_snapshot(snap)
            agents = {(r["payload"] or {}).get("agent_id") for r in rows}
            bad = {a for a in agents if a != cid}
            status = "OK " if not bad else "FAIL"
            if bad:
                ok = False
            print(f"  [{status}] {cid}/{snap.name:14s}  n={len(rows):4d}  "
                  f"agent_ids={agents}")
    print(f"\n  result: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------
# 1B) Dynamic isolation against the LIVE main collection
# --------------------------------------------------------------------------
def validate_live_isolation() -> bool:
    print("=" * 70)
    print("[1B] Live main-collection isolation — filter agent_id=X returns only X")
    print("=" * 70)
    from ingest_memories import build_memory, COLLECTION
    from qdrant_client.http import models as qm

    mem = build_memory()
    qclient = mem.vector_store.client

    chars = sorted([d.name for d in SNAP_ROOT.iterdir() if d.is_dir()])

    # per-char: count points filtered by agent_id, check they're all that agent
    ok = True
    counts: dict[str, int] = {}
    for cid in chars:
        flt = qm.Filter(must=[qm.FieldCondition(
            key="agent_id", match=qm.MatchValue(value=cid))])
        n = 0
        bad_agents: set[str] = set()
        offset = None
        while True:
            pts, offset = qclient.scroll(
                collection_name=COLLECTION, scroll_filter=flt,
                limit=512, offset=offset, with_payload=True, with_vectors=False)
            for p in pts:
                a = (p.payload or {}).get("agent_id")
                if a != cid:
                    bad_agents.add(a)
                n += 1
            if offset is None:
                break
        counts[cid] = n
        status = "OK " if not bad_agents else "FAIL"
        if bad_agents:
            ok = False
        print(f"  [{status}] agent_id={cid}  n={n:4d}  "
              f"contaminating_agents={bad_agents or '∅'}")

    # also check the union: does scrolling with agent_id == any partition
    # give the same total as scrolling unfiltered? this catches "orphan" points
    # whose payload has no agent_id at all.
    total_unfiltered = 0
    offset = None
    while True:
        pts, offset = qclient.scroll(
            collection_name=COLLECTION, limit=512, offset=offset,
            with_payload=False, with_vectors=False)
        total_unfiltered += len(pts)
        if offset is None:
            break
    sum_per_agent = sum(counts.values())
    print(f"\n  unfiltered total = {total_unfiltered}, "
          f"sum-of-per-agent = {sum_per_agent}")
    if total_unfiltered != sum_per_agent:
        print(f"  WARN: {total_unfiltered - sum_per_agent} points are not "
              f"covered by any agent_id filter")
        ok = False

    # close so the next step can grab the lock
    try:
        mem.vector_store.client.close()
    except Exception:
        pass
    try:
        mem._telemetry_vector_store.client.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    import gc; gc.collect()

    print(f"\n  result: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------
# 1C) Cross-character search: same query, 11 chars, top-1 should differ
# --------------------------------------------------------------------------
def validate_cross_char_search() -> bool:
    print("=" * 70)
    print("[1C] Cross-character search via filter — same query, different "
          "top-1 per char")
    print("=" * 70)
    from ingest_memories import build_memory

    mem = build_memory()
    chars = sorted([d.name for d in SNAP_ROOT.iterdir() if d.is_dir()])

    # Sample retrieval queries.
    queries = ["relationship with parents", "childhood fears"]
    ok = True
    for q in queries:
        print(f"\n  query: {q!r}")
        seen_top: dict[str, str] = {}
        for cid in chars:
            r = mem.search(query=q, filters={"agent_id": cid}, top_k=1)
            res = r["results"]
            if not res:
                print(f"    {cid}: <no results>")
                continue
            top = res[0]
            md = top.get("metadata") or {}
            mid = md.get("mem_id")
            txt = top["memory"][:60].replace("\n", " ")
            seen_top[cid] = mid or txt
            print(f"    {cid}: score={top['score']:.3f}  mem_id={mid}  ::  {txt}")
        # are the top-1 mem_ids unique-ish? (some queries may hit similar things,
        # but if we ever get all 11 the same that's a strong cross-talk signal)
        unique = len(set(seen_top.values()))
        print(f"    unique top-1 across {len(seen_top)} chars: {unique}")
        if unique <= 1:
            ok = False
            print("    FAIL: all chars returned same top-1 (looks like cross-talk)")

    try:
        mem.vector_store.client.close()
    except Exception:
        pass
    try:
        mem._telemetry_vector_store.client.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    import gc; gc.collect()
    print(f"\n  result: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------
# 2) Per-age snapshot accuracy for CHAR_01 (only char with >2 snapshots)
# --------------------------------------------------------------------------
def validate_age_progression(cid: str = "CHAR_01") -> bool:
    print("=" * 70)
    print(f"[2] Per-age snapshot monotonicity & age boundaries — {cid}")
    print("=" * 70)
    snaps = sorted((SNAP_ROOT / cid).glob("age_*.jsonl"))
    if len(snaps) < 2:
        print(f"  skip — only {len(snaps)} snapshot(s) for {cid}")
        return True

    by_age: dict[int, list[dict]] = {}
    for s in snaps:
        age = int(s.stem.split("_")[1])
        by_age[age] = read_snapshot(s)

    ages_sorted = sorted(by_age.keys())
    ok = True
    print(f"  ages found: {ages_sorted}")

    # 2a: id-set monotonicity
    print("\n  [2a] id-set monotonicity ids(N) ⊆ ids(N+1):")
    prev_age = None
    prev_ids: set[str] = set()
    for age in ages_sorted:
        ids = {r["id"] for r in by_age[age]}
        if prev_age is None:
            print(f"    age_{age:02d}: |ids|={len(ids)} (baseline)")
        else:
            missing = prev_ids - ids
            added = ids - prev_ids
            status = "OK " if not missing else "FAIL"
            if missing:
                ok = False
            print(f"    [{status}] age_{prev_age:02d}→age_{age:02d}: "
                  f"|prev|={len(prev_ids)} |cur|={len(ids)} "
                  f"added={len(added)} dropped={len(missing)}")
            if missing:
                print(f"           missing sample: {list(missing)[:3]}")
        prev_age, prev_ids = age, ids

    # 2b: max age in payload.timeline ≤ snapshot age label
    print("\n  [2b] max payload age ≤ snapshot label:")
    for age in ages_sorted:
        payload_ages = []
        for r in by_age[age]:
            md = r.get("payload") or {}
            tl = md.get("timeline") or md.get("data") or ""
            # try the timeline field first, fall back to scanning payload values
            a = parse_age(tl) if isinstance(tl, str) else None
            if a is None:
                # scan all string-valued payload fields for any age token
                for v in md.values():
                    if isinstance(v, str):
                        m = AGE_RE.search(v)
                        if m:
                            a = int(m.group(1))
                            break
            if a is not None:
                payload_ages.append(a)
        if not payload_ages:
            print(f"    age_{age:02d}: no parsable ages in payload (n={len(by_age[age])})")
            continue
        mx = max(payload_ages)
        mn = min(payload_ages)
        status = "OK " if mx <= age else "FAIL"
        if mx > age:
            ok = False
        print(f"    [{status}] age_{age:02d}: payload age range = [{mn}, {mx}]  "
              f"(samples: {payload_ages[:5]})")

    print(f"\n  result: {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> None:
    results = []
    results.append(("1A static isolation", validate_static_isolation()))
    results.append(("1B live-collection isolation", validate_live_isolation()))
    results.append(("1C cross-char filtered search", validate_cross_char_search()))
    results.append(("2  CHAR_01 age progression", validate_age_progression("CHAR_01")))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

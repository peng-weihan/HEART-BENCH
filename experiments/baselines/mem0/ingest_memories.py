"""Ingest episodic_memory_set into mem0 with per-character isolation.

11 characters share one qdrant collection; isolation is via agent_id=CHAR_XX.

Usage:
    source .venv/bin/activate

    # dry-run: print what would be sent, don't call APIs
    python ingest_memories.py --char CHAR_01 --limit 3 --dry-run

    # ingest first 20 memories of one character
    python ingest_memories.py --char CHAR_01 --limit 20

    # ingest all 1000 of one character with 4 parallel workers
    python ingest_memories.py --char CHAR_01 --workers 4

    # ingest everything (11 * 1000)
    python ingest_memories.py --workers 4

Resumable: each successfully-ingested mem_id is appended to
.mem0/ingested.jsonl; reruns skip them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---- env loader ------------------------------------------------------------
ROOT = Path(__file__).parent
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v or v.startswith("sk-your-"):
        sys.exit(f"{name} is not set in .env")
    return v


LLM_KEY = _require("API_KEY")
LLM_BASE = os.environ.get("API_BASE", "https://llm-sjtu.multiego.me/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")
EMBED_KEY = _require("AIHUBMIX_API_KEY")
EMBED_BASE = os.environ.get("AIHUBMIX_API_BASE", "https://aihubmix.com/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-4b")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "2560"))

REPO_ROOT = ROOT.resolve().parent.parent.parent  # experiments/baselines/mem0 → repo
CHARS_FILE = REPO_ROOT / "benchmark" / "characters.json"
# Per-process isolation: when MEM0_CHAR_TAG is set (e.g. CHAR_01), each
# process gets its own qdrant dir, ingested log, and SQLite history db.
# Empty tag => legacy shared paths (single-process mode).
_PER_CHAR = os.environ.get("MEM0_CHAR_TAG", "")
_TAG = f"_{_PER_CHAR}" if _PER_CHAR else ""
INGESTED_LOG = ROOT / ".mem0" / f"ingested{_TAG}.jsonl"
QDRANT_PATH = ROOT / ".mem0" / f"qdrant{_TAG}"
HISTORY_DB = ROOT / ".mem0" / f"history{_TAG}.db"
SNAPSHOT_ROOT = ROOT / ".mem0" / "snapshots"
COLLECTION = "mem0_qwen3_4b"


# Matches an age token like "age <N>" in dataset timeline strings.
_AGE_RE = re.compile(r"age\s*(\d+)", re.IGNORECASE)


def parse_age(timeline: str) -> int | None:
    m = _AGE_RE.search(timeline or "")
    return int(m.group(1)) if m else None


# ---- mem0 setup (lazy so --dry-run doesn't import) -------------------------
def build_memory():
    from mem0 import Memory

    config = {
        "history_db_path": str(HISTORY_DB),
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "openai_base_url": LLM_BASE,
                "api_key": LLM_KEY,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBED_MODEL,
                "openai_base_url": EMBED_BASE,
                "api_key": EMBED_KEY,
                # Do NOT set embedding_dims; aihubmix Qwen3 rejects `dimensions=`.
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": COLLECTION,
                "embedding_model_dims": EMBED_DIMS,
                "path": str(QDRANT_PATH),
            },
        },
    }
    return Memory.from_config(config)


# ---- ingest log (resume support) ------------------------------------------
def load_ingested() -> set[str]:
    if not INGESTED_LOG.exists():
        return set()
    seen: set[str] = set()
    for line in INGESTED_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            seen.add(row["mem_id"])
        except Exception:
            continue
    return seen


_log_lock = threading.Lock()
_dump_lock = threading.Lock()  # serialize qdrant scroll+write across char workers


def mark_ingested(mem_id: str, agent_id: str, mem0_result: Any) -> None:
    INGESTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"mem_id": mem_id, "agent_id": agent_id, "ts": time.time(), "result": mem0_result},
        ensure_ascii=False,
    )
    with _log_lock:
        with INGESTED_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---- snapshot dump --------------------------------------------------------
def dump_snapshot(memory_obj, agent_id: str, age: int) -> Path:
    """Dump current cumulative state of `agent_id` from the live qdrant collection
    into snapshots/<agent_id>/age_<NN>.jsonl, including vectors so restore can
    skip re-embedding."""
    qclient = memory_obj.vector_store.client  # qdrant local client
    out_dir = SNAPSHOT_ROOT / agent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"age_{age:02d}.jsonl"

    from qdrant_client.http import models as qm

    flt = qm.Filter(
        must=[qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
    )
    batch_size = 256
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        offset = None
        while True:
            points, offset = qclient.scroll(
                collection_name=COLLECTION,
                scroll_filter=flt,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                vec = p.vector
                # qdrant may return dict {name: [...]} for named vectors; flatten
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()))
                row = {
                    "id": str(p.id),
                    "vector": list(vec) if vec is not None else None,
                    "payload": p.payload,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            if offset is None:
                break
    print(f"  snapshot -> {out_path}  ({n} memories)")
    return out_path


def dump_snapshot_locked(memory_obj, agent_id: str, age: int) -> Path:
    """Thread-safe wrapper. snapshot dumps shouldn't race with each other on
    the shared qdrant client."""
    with _dump_lock:
        return dump_snapshot(memory_obj, agent_id, age)


# ---- snapshot restore (scenario D: continue ingest from a snapshot) -------
def restore_snapshot_into_main(memory_obj, agent_id: str, age: int) -> int:
    """Wipe agent_id's points in the main collection, then upsert the snapshot
    file age_<NN>.jsonl back. Returns number of points restored. Used to
    "rewind" the live store before continuing ingestion past the snapshot age."""
    snap = SNAPSHOT_ROOT / agent_id / f"age_{age:02d}.jsonl"
    if not snap.exists():
        sys.exit(f"snapshot file not found: {snap}")

    from qdrant_client.http import models as qm

    qclient = memory_obj.vector_store.client

    # 1) clear existing points for this agent in the main collection
    flt = qm.Filter(
        must=[qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
    )
    qclient.delete(collection_name=COLLECTION, points_selector=qm.FilterSelector(filter=flt))

    # 2) upsert snapshot points back
    BATCH = 256
    batch: list = []
    n = 0
    with snap.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch.append(qm.PointStruct(id=row["id"], vector=row["vector"], payload=row["payload"]))
            if len(batch) >= BATCH:
                qclient.upsert(collection_name=COLLECTION, points=batch)
                n += len(batch)
                batch = []
    if batch:
        qclient.upsert(collection_name=COLLECTION, points=batch)
        n += len(batch)
    print(f"  restored {n} points for {agent_id} from {snap.name}")
    return n


def rewrite_ingested_log_for_resume(resume_map: dict[str, int], chars_data: dict) -> int:
    """Drop entries from .mem0/ingested.jsonl whose mem belongs to (agent_id, age>cutoff).
    Keeps every other line. Returns number of entries dropped.

    Why: after restoring agent X to age 16, ingest must NOT skip 17+yo mem_ids
    just because they were marked ingested in a prior crashed run."""
    if not INGESTED_LOG.exists() or not resume_map:
        return 0

    # Build mem_id -> (agent_id, age) lookup from chars_data.
    mem_age: dict[str, tuple[str, int | None]] = {}
    for c in chars_data["characters"]:
        cid = c["id"]
        for m in c["episodic_memory_set"]:
            mem_age[m["id"]] = (cid, parse_age(m.get("timeline", "")))

    kept: list[str] = []
    dropped = 0
    for line in INGESTED_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            mid = row.get("mem_id")
            agent, age = mem_age.get(mid, (None, None))
            cutoff = resume_map.get(agent)
            if cutoff is not None and age is not None and age > cutoff:
                dropped += 1
                continue
        except Exception:
            pass
        kept.append(line)

    INGESTED_LOG.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return dropped


def memory_to_messages_and_meta(mem: dict) -> tuple[list[dict], dict]:
    """Build (messages, metadata) for one episodic memory.

    The LLM extractor in mem0 pulls facts from `content_full` (the long-form
    narrative). With more input mem0 may legitimately produce multiple facts
    from a single source memory, at the cost of more LLM tokens per add() and
    higher chance of hallucinated facts (see analysis: when the agent already
    has many memories, the extractor occasionally re-emits unrelated existing
    ones).

    `content_summary` is preserved in metadata for filter access / regression
    comparison; the rest of the structured fields ride along the same way.
    """
    full_text = mem.get("content_full") or mem["content_summary"]
    messages = [{"role": "user", "content": full_text}]
    metadata = {
        "mem_id": mem["id"],
        "timeline": mem.get("timeline"),
        "context": mem.get("context"),
        "triggers": mem.get("triggers"),
        "psych_conclusion": mem.get("psych_conclusion"),
        "behavior_policy": mem.get("behavior_policy"),
        "emotion_signature": mem.get("emotion_signature"),
        "relevance_tags": mem.get("relevance_tags"),
        "content_summary": mem.get("content_summary"),
        "content_full": mem.get("content_full"),
    }
    return messages, metadata


# ---- runner ---------------------------------------------------------------
def ingest_one(memory_obj, agent_id: str, mem: dict) -> tuple[str, str, Any]:
    msgs, meta = memory_to_messages_and_meta(mem)
    res = memory_obj.add(msgs, agent_id=agent_id, metadata=meta, infer=True)
    return mem["id"], "ok", res


def select_characters(data: dict, only: str | None) -> list[dict]:
    if only is None:
        return data["characters"]
    return [c for c in data["characters"] if c["id"] == only]


def run_character_worker(memory_obj, cid: str, mems: list[dict], seen: set[str],
                          snapshot_by_age: bool, start_age: int | None,
                          progress: dict, progress_lock: threading.Lock) -> tuple[int, int]:
    """Process one character serially in age order. Returns (ok, fail) counts.

    `progress` is shared across workers; we update progress[cid] = (done, fail, last_age)
    so the main thread can print rolled-up status."""
    current_age: int | None = start_age
    ok = fail = 0
    for m in mems:
        if m["id"] in seen:
            continue
        age = parse_age(m.get("timeline", ""))
        if snapshot_by_age and current_age is not None and age is not None and age != current_age:
            if age < current_age:
                print(f"WARN: age went backwards for {cid}: {current_age} -> {age}")
            else:
                dump_snapshot_locked(memory_obj, cid, current_age)
        try:
            mid, _, res = ingest_one(memory_obj, cid, m)
            mark_ingested(mid, cid, res)
            ok += 1
            current_age = age
            with progress_lock:
                progress[cid] = (ok, fail, age)
        except Exception as e:
            fail += 1
            print(f"FAIL agent={cid} mem={m['id']} :: {repr(e)[:200]}")
            with progress_lock:
                progress[cid] = (ok, fail, current_age)

    # Final dump for this character.
    if snapshot_by_age and current_age is not None and ok > 0:
        dump_snapshot_locked(memory_obj, cid, current_age)
    return ok, fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--char", default=None, help="ingest only this CHAR_XX id (default: all 11)")
    ap.add_argument("--start", type=int, default=0, help="skip this many memories per character")
    ap.add_argument("--limit", type=int, default=None, help="cap per character (default: all)")
    ap.add_argument("--workers", type=int, default=1, help="parallel ingest threads (1 = serial)")
    ap.add_argument("--char-parallel", type=int, default=0, metavar="N",
                    help="run up to N characters in parallel; each character is processed serially "
                         "internally so age order (and snapshot triggers) stay correct. "
                         "Mutually exclusive with --workers >1. Use this for multi-character ingest.")
    ap.add_argument("--snapshot-by-age", action="store_true",
                    help="dump per-character cumulative snapshot each time the timeline age advances")
    ap.add_argument("--resume-from-snapshot", action="append", default=[], metavar="CHAR_XX:AGE",
                    help="restore CHAR_XX's main-collection state from snapshot age_<AGE>.jsonl, "
                         "drop ingested-log entries past that age, then continue ingest. "
                         "Repeatable for multiple chars, e.g. "
                         "--resume-from-snapshot CHAR_01:16 --resume-from-snapshot CHAR_02:20")
    ap.add_argument("--dry-run", action="store_true", help="print plan, no API calls")
    args = ap.parse_args()

    if args.snapshot_by_age and args.workers != 1:
        sys.exit("--snapshot-by-age requires --workers 1 (or use --char-parallel which preserves per-character order)")
    if args.char_parallel and args.workers > 1:
        sys.exit("--char-parallel is mutually exclusive with --workers >1")

    # Parse --resume-from-snapshot CHAR_XX:AGE specs
    resume_map: dict[str, int] = {}
    for spec in args.resume_from_snapshot:
        if ":" not in spec:
            sys.exit(f"bad --resume-from-snapshot value '{spec}'; expected CHAR_XX:AGE")
        cid, _, age_s = spec.partition(":")
        try:
            resume_map[cid] = int(age_s)
        except ValueError:
            sys.exit(f"bad age in --resume-from-snapshot '{spec}'")

    data = json.loads(CHARS_FILE.read_text())
    chars = select_characters(data, args.char)
    if not chars:
        sys.exit(f"no character matched --char={args.char}")

    # If resuming from snapshot, prune ingested.jsonl FIRST so the resume cutoff
    # determines which mem_ids count as "already ingested".
    if resume_map:
        dropped = rewrite_ingested_log_for_resume(resume_map, data)
        print(f"resume: pruned {dropped} stale ingested-log entries past cutoffs {resume_map}")

    seen = load_ingested()
    print(f"resume log: {len(seen)} mem_ids already ingested")

    # build job list (preserves original per-character order, hence age order)
    jobs: list[tuple[str, dict]] = []
    for c in chars:
        cid = c["id"]
        mems = c["episodic_memory_set"][args.start:]
        if args.limit is not None:
            mems = mems[: args.limit]
        for m in mems:
            if m["id"] in seen:
                continue
            jobs.append((cid, m))

    print(f"chars: {[c['id'] for c in chars]}")
    print(f"planned ingestions: {len(jobs)}  (start={args.start} limit={args.limit} "
          f"workers={args.workers} snapshot_by_age={args.snapshot_by_age})")

    if args.dry_run:
        for cid, m in jobs[:5]:
            print(f"  - {cid} :: {m['id']} :: age={parse_age(m.get('timeline',''))} "
                  f":: {m['content_summary'][:50]}")
        if len(jobs) > 5:
            print(f"  ... and {len(jobs)-5} more")
        return

    if not jobs:
        print("nothing to do")
        return

    print("building Memory ...")
    memory_obj = build_memory()

    # Apply snapshot restore to the live store BEFORE ingest so the next add()
    # operates against the rewound state. Also re-establish current_age so the
    # very next age advance triggers a fresh snapshot dump.
    resumed_current_age: dict[str, int] = {}
    for cid, age in resume_map.items():
        restore_snapshot_into_main(memory_obj, cid, age)
        resumed_current_age[cid] = age

    print("starting ingest ...")

    t0 = time.time()
    done, fail = 0, 0

    if args.char_parallel and args.char_parallel > 0:
        # Per-character workers: each char serial internally (age order preserved),
        # up to N chars active concurrently (concurrency = parallelism for LLM/embed calls).
        # Build per-char job list.
        per_char_jobs: dict[str, list[dict]] = {}
        for cid, m in jobs:
            per_char_jobs.setdefault(cid, []).append(m)

        progress: dict[str, tuple[int, int, int | None]] = {}
        progress_lock = threading.Lock()
        n_workers = min(args.char_parallel, len(per_char_jobs))
        print(f"char-parallel mode: {len(per_char_jobs)} chars, {n_workers} concurrent workers")

        def _runner(cid_: str):
            return cid_, run_character_worker(
                memory_obj, cid_, per_char_jobs[cid_], seen,
                args.snapshot_by_age, resumed_current_age.get(cid_),
                progress, progress_lock,
            )

        # Light progress reporter — every 5s prints rolled-up status.
        stop_evt = threading.Event()

        def _reporter():
            while not stop_evt.wait(5.0):
                with progress_lock:
                    snap = dict(progress)
                if not snap:
                    continue
                total_done = sum(o for o, _, _ in snap.values())
                total_fail = sum(f for _, f, _ in snap.values())
                total_planned = len(jobs)
                ages = " ".join(f"{c}@{a}" for c, (_, _, a) in sorted(snap.items()))
                print(f"[{total_done+total_fail}/{total_planned}] ok={total_done} fail={total_fail} "
                      f"elapsed={time.time()-t0:.1f}s :: {ages}")

        rep = threading.Thread(target=_reporter, daemon=True)
        rep.start()

        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_runner, cid) for cid in per_char_jobs]
                for fut in as_completed(futures):
                    cid_done, (o, f) = fut.result()
                    done += o
                    fail += f
                    print(f"  char done: {cid_done}  ok={o} fail={f}")
        finally:
            stop_evt.set()
            rep.join(timeout=1)

    elif args.workers <= 1:
        # Track current age per character to detect advances.
        # Pre-seed with snapshot-resume cutoffs so the next age tick fires a dump.
        current_age: dict[str, int | None] = dict(resumed_current_age)
        for cid, m in jobs:
            age = parse_age(m.get("timeline", ""))
            prev = current_age.get(cid)
            # If snapshotting and age advanced, dump the *previous* age first.
            if args.snapshot_by_age and prev is not None and age is not None and age != prev:
                if age < prev:
                    print(f"WARN: age went backwards for {cid}: {prev} -> {age}; snapshot may be incorrect")
                else:
                    dump_snapshot(memory_obj, cid, prev)
            try:
                mid, status, res = ingest_one(memory_obj, cid, m)
                mark_ingested(mid, cid, res)
                done += 1
                current_age[cid] = age
                if done % 5 == 0 or done == len(jobs):
                    print(f"[{done}/{len(jobs)}] ok  agent={cid}  age={age}  mem={mid}  "
                          f"elapsed={time.time()-t0:.1f}s")
            except Exception as e:
                fail += 1
                print(f"FAIL agent={cid} mem={m['id']} :: {repr(e)[:200]}")

        # After all jobs, dump the final age for each character touched.
        if args.snapshot_by_age:
            for cid, age in current_age.items():
                if age is not None:
                    dump_snapshot(memory_obj, cid, age)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ingest_one, memory_obj, cid, m): (cid, m["id"]) for cid, m in jobs}
            for fut in as_completed(futures):
                cid, mid = futures[fut]
                try:
                    _, _, res = fut.result()
                    mark_ingested(mid, cid, res)
                    done += 1
                    if done % 5 == 0 or done == len(jobs):
                        print(f"[{done}/{len(jobs)}] ok  agent={cid}  mem={mid}  elapsed={time.time()-t0:.1f}s")
                except Exception as e:
                    fail += 1
                    print(f"FAIL agent={cid} mem={mid} :: {repr(e)[:200]}")

    print(f"\ndone. ok={done}  fail={fail}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

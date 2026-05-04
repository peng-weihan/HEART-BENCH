"""Restore a per-character snapshot into a fresh queryable mem0 Memory.

A snapshot is the cumulative state of one character at a given age, dumped to
.mem0/snapshots/CHAR_XX/age_NN.jsonl by ingest_memories.py --snapshot-by-age.

Each line in a snapshot file is:
    {"id": "<uuid>", "vector": [2560 floats], "payload": {...}}

This restores it into an *isolated* qdrant collection (so it does not collide
with the live ingestion collection or other restored snapshots) and returns a
ready-to-search Memory instance.

CLI usage:
    python restore_snapshot.py --char CHAR_01 --age 25
    # then it runs an interactive search loop

Programmatic usage:
    from restore_snapshot import restore
    mem = restore("CHAR_01", 25)
    print(mem.search(query="committee meeting", filters={"agent_id":"CHAR_01"}, top_k=5))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
LLM_BASE = os.environ.get("API_BASE", "")
if not LLM_BASE:
    sys.exit("API_BASE is not set in .env")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")
EMBED_KEY = _require("AIHUBMIX_API_KEY")
EMBED_BASE = os.environ.get("AIHUBMIX_API_BASE", "https://aihubmix.com/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-4b")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "2560"))

SNAPSHOT_ROOT = ROOT / ".mem0" / "snapshots"
RESTORE_ROOT = ROOT / ".mem0" / "restored"


def snapshot_path(char: str, age: int) -> Path:
    return SNAPSHOT_ROOT / char / f"age_{age:02d}.jsonl"


def list_snapshots(char: str) -> list[int]:
    d = SNAPSHOT_ROOT / char
    if not d.exists():
        return []
    out = []
    for p in d.glob("age_*.jsonl"):
        try:
            out.append(int(p.stem.split("_")[1]))
        except Exception:
            pass
    return sorted(out)


def restore(char: str, age: int, force_recreate: bool = False):
    """Build a Memory whose vector store is the snapshot of `char` at `age`.

    The collection is named snap_<char>_age_<NN> and lives under
    .mem0/restored/<char>_age_<NN>/ so live ingestion is untouched.
    """
    snap = snapshot_path(char, age)
    if not snap.exists():
        avail = list_snapshots(char)
        sys.exit(f"snapshot not found: {snap}\n  available ages for {char}: {avail}")

    from mem0 import Memory
    from qdrant_client.http import models as qm

    collection = f"snap_{char}_age_{age:02d}"
    qpath = RESTORE_ROOT / f"{char}_age_{age:02d}"
    qpath.mkdir(parents=True, exist_ok=True)

    config = {
        "llm": {
            "provider": "openai",
            "config": {"model": LLM_MODEL, "openai_base_url": LLM_BASE, "api_key": LLM_KEY},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": EMBED_MODEL, "openai_base_url": EMBED_BASE, "api_key": EMBED_KEY},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": EMBED_DIMS,
                "path": str(qpath),
            },
        },
    }
    mem = Memory.from_config(config)
    qclient = mem.vector_store.client

    # If collection already populated, optionally wipe and re-load.
    info = qclient.get_collection(collection_name=collection)
    if info.points_count and info.points_count > 0 and not force_recreate:
        print(f"collection '{collection}' already has {info.points_count} points; reusing")
        return mem

    if force_recreate and info.points_count:
        # Clear by deleting all points (collection schema kept).
        qclient.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qm.Filter()),
        )

    # Stream snapshot file -> upsert in batches.
    batch_points: list = []
    BATCH = 256
    total = 0
    with snap.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch_points.append(
                qm.PointStruct(id=row["id"], vector=row["vector"], payload=row["payload"])
            )
            if len(batch_points) >= BATCH:
                qclient.upsert(collection_name=collection, points=batch_points)
                total += len(batch_points)
                batch_points = []
    if batch_points:
        qclient.upsert(collection_name=collection, points=batch_points)
        total += len(batch_points)

    print(f"restored {total} memories -> collection '{collection}' (path={qpath})")
    return mem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--char", required=True, help="CHAR_XX id")
    ap.add_argument("--age", type=int, required=True, help="snapshot age, e.g. 25")
    ap.add_argument("--force-recreate", action="store_true",
                    help="if the restore collection already has data, wipe and re-load from jsonl")
    ap.add_argument("--query", default=None,
                    help="if given, run a single search and print results, then exit")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    mem = restore(args.char, args.age, force_recreate=args.force_recreate)

    def show(query: str) -> None:
        r = mem.search(query=query, filters={"agent_id": args.char}, top_k=args.top_k)
        for x in r["results"]:
            md = x.get("metadata") or {}
            print(f"  score={x['score']:.3f}  age_at_event={md.get('timeline')}  "
                  f"mem_id={md.get('mem_id')}")
            print(f"    text: {x['memory'][:120]}")

    if args.query:
        print(f"--- search: {args.query} ---")
        show(args.query)
        return

    print(f"\nInteractive search ({args.char} @ age {args.age}). Empty line to quit.")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(q)


if __name__ == "__main__":
    main()

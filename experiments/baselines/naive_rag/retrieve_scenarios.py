"""
retrieve_scenarios.py — Pre-compute top-K memory retrieval per (scenario, character).

For every scenario in ``benchmark/scenarios.json``, retrieve the top-K most
relevant memories from each of the 11 characters' ``episodic_memory_set``.

- Query text  : description_for_agent + context_text + trigger_event.message_content
                + trigger_event.action_required
- Candidate   : only memories with ``mem.age <= scenario.age``
- Similarity  : cosine (L2-normalized dot product)
- Ranking     : sorted by score desc, ``rank`` starts at 1
- If fewer than K memories pass the age filter, return them all.

Output (single compact JSON file, consumed by
``experiments/scripts/run_naive_rag.py``):

  cache/retrieval/scenario_topk.json
  {
    "meta": {...},
    "results": {
      "<scenario_id>": {
        "scenario_id": ..., "stage": ..., "age": ...,
        "query_text_preview": "...",   # first 200 chars, for human inspection
        "by_character": {
          "<char_id>": {
            "candidate_pool_size": int,
            "topk": [
              {"rank": 1, "mem_id": "MEM_...", "score": 0.83, "mem_age": 7}, ...
            ]
          }
        }
      }
    }
  }

Only ``mem_id`` / ``score`` / ``rank`` / ``mem_age`` are stored — the downstream
runner looks ``mem_id`` up in ``characters.json`` to recover ``content_full``.

Per-character embeddings are read from ``cache/embeddings/<char_id>.json``
(override with ``--embeddings-dir`` or ``EMBEDDINGS_DIR``). Each file is expected
to follow the schema:

  {
    "memories": [
      {"id": "MEM_...", "embedding": [float, ...]},
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths (aligned with HEART-BENCH layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
CACHE_DIR = PROJECT_ROOT / "cache"

SCENARIOS_FILE = BENCHMARK_DIR / "scenarios.json"
CHARACTERS_FILE = BENCHMARK_DIR / "characters.json"

DEFAULT_EMB_DIR = CACHE_DIR / "embeddings"
DEFAULT_OUT_DIR = CACHE_DIR / "retrieval"
DEFAULT_OUT_FILE = DEFAULT_OUT_DIR / "scenario_topk.json"


# ---------------------------------------------------------------------------
# Embedding API configuration (env-overridable)
# ---------------------------------------------------------------------------
def _load_dotenv(p: Path) -> None:
    """Lightweight .env loader (mirrors run_naive_rag.py)."""
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv(PROJECT_ROOT / ".env")

# Prefer dedicated embedding-side env vars, fall back to the generic API_KEY /
# API_BASE used by the chat runner so a single .env can drive both.
DEFAULT_API_KEY = os.environ.get(
    "EMBEDDING_API_KEY",
    os.environ.get("AIHUBMIX_API_KEY", os.environ.get("API_KEY", "")),
)
DEFAULT_API_BASE = os.environ.get(
    "EMBEDDING_API_BASE",
    os.environ.get("AIHUBMIX_API_BASE", os.environ.get("API_BASE", "https://api.openai.com/v1")),
)
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-4b")
DEFAULT_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "30"))

MAX_RETRIES = 8
RETRY_BASE_DELAY = 5.0
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Match both English ("age 6", "Age 12") and Chinese ("6岁") forms so the
# parser stays robust if the timeline field is ever localized again.
_AGE_RE = re.compile(r"(?:age\s*|Age\s*)?(\d{1,3})\s*(?:岁)?", re.IGNORECASE)
_AGE_RE_EN = re.compile(r"age\s*(\d{1,3})", re.IGNORECASE)
_AGE_RE_ZH = re.compile(r"(\d{1,3})\s*岁")


def parse_mem_age(timeline: str) -> int | None:
    """Extract an integer age from a timeline string.

    Examples handled:
      "Childhood (age 6)"  -> 6
      "Adolescence (Age 14)" -> 14
      "童年(6岁)" -> 6
    """
    if not timeline:
        return None
    m = _AGE_RE_EN.search(timeline)
    if m:
        return int(m.group(1))
    m = _AGE_RE_ZH.search(timeline)
    if m:
        return int(m.group(1))
    return None


def build_scenario_query(scenario: dict) -> str:
    """Concatenate scenario fields into a single query string."""
    parts: list[str] = []
    desc = scenario.get("description_for_agent", "")
    if desc:
        parts.append(desc)
    ctx = scenario.get("context_text", "")
    if ctx:
        parts.append(ctx)
    trig = scenario.get("trigger_event") or {}
    msg = trig.get("message_content", "")
    if msg:
        parts.append(msg)
    act = trig.get("action_required", "")
    if act:
        parts.append(act)
    return "\n".join(parts).strip()


def embed_with_retry(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"  [retry {attempt}/{MAX_RETRIES}] embed error: {e}; sleeping {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"embed failed after {MAX_RETRIES} retries: {last_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--scenarios-file",
        type=Path,
        default=SCENARIOS_FILE,
        help=f"Path to scenarios.json (default: {SCENARIOS_FILE})",
    )
    ap.add_argument(
        "--characters-file",
        type=Path,
        default=CHARACTERS_FILE,
        help=f"Path to characters.json (default: {CHARACTERS_FILE})",
    )
    ap.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(os.environ.get("EMBEDDINGS_DIR", str(DEFAULT_EMB_DIR))),
        help=f"Directory containing <CHAR_ID>.json embedding files (default: {DEFAULT_EMB_DIR})",
    )
    ap.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help=f"Output JSON path (default: {DEFAULT_OUT_FILE})",
    )
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    args = ap.parse_args()

    scenarios_file: Path = args.scenarios_file
    characters_file: Path = args.characters_file
    emb_dir: Path = args.embeddings_dir
    out_file: Path = args.out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        print(
            "ERROR: no embedding API key found. Set EMBEDDING_API_KEY / API_KEY / "
            "AIHUBMIX_API_KEY in env or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- load scenarios ----
    scen_data = json.loads(scenarios_file.read_text(encoding="utf-8"))
    scenarios_by_stage: dict[str, list[dict]] = scen_data["scenarios"]
    all_scenarios: list[dict] = []
    for _stage, sc_list in scenarios_by_stage.items():
        for sc in sc_list:
            all_scenarios.append(sc)
    print(f"loaded {len(all_scenarios)} scenarios from {scenarios_file}")

    # ---- load characters (for memory ages and the canonical id list) ----
    char_data = json.loads(characters_file.read_text(encoding="utf-8"))
    characters: list[dict] = char_data["characters"]
    char_ids = [c["id"] for c in characters]
    mem_age_by_id: dict[str, int | None] = {}
    for c in characters:
        for mem in c.get("episodic_memory_set", []) or []:
            mid = mem.get("id")
            if mid:
                mem_age_by_id[mid] = parse_mem_age(mem.get("timeline", ""))
    n_with_age = sum(1 for v in mem_age_by_id.values() if v is not None)
    print(
        f"loaded {len(characters)} characters, "
        f"{len(mem_age_by_id)} memory ids ({n_with_age} with parseable age)"
    )
    if n_with_age == 0:
        print(
            "ERROR: no memory ages could be parsed from timelines. "
            "Check parse_mem_age() and the timeline format.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- load all per-character embeddings ----
    if not emb_dir.exists():
        print(
            f"ERROR: embeddings dir {emb_dir} does not exist. "
            f"Generate per-character embeddings first (one <CHAR_ID>.json each).",
            file=sys.stderr,
        )
        sys.exit(2)

    char_index: dict[str, dict[str, Any]] = {}
    for cid in char_ids:
        emb_path = emb_dir / f"{cid}.json"
        if not emb_path.exists():
            print(f"  [warn] no embeddings for {cid} at {emb_path}", file=sys.stderr)
            continue
        emb = json.loads(emb_path.read_text(encoding="utf-8"))
        mems = emb["memories"]
        ids = [m["id"] for m in mems]
        ages = [mem_age_by_id.get(mid) for mid in ids]
        vecs = np.asarray([m["embedding"] for m in mems], dtype=np.float32)
        # L2-normalize so cosine == dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        char_index[cid] = {"ids": ids, "ages": ages, "vecs": vecs}
        with_age = sum(1 for a in ages if a is not None)
        print(
            f"  [{cid}] loaded {len(ids)} embeddings "
            f"(dim={vecs.shape[1]}, with_age={with_age})"
        )

    if not char_index:
        print("ERROR: no per-character embedding files were loaded.", file=sys.stderr)
        sys.exit(2)

    # ---- embed all scenario queries (batched) ----
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    queries = [build_scenario_query(sc) for sc in all_scenarios]
    print(f"\nembedding {len(queries)} scenario queries (batched, batch={BATCH_SIZE})...")
    query_vecs: list[list[float]] = []
    for start in range(0, len(queries), BATCH_SIZE):
        batch = queries[start : start + BATCH_SIZE]
        vecs = embed_with_retry(client, args.model, batch)
        query_vecs.extend(vecs)
        print(f"  embedded {min(start + BATCH_SIZE, len(queries))}/{len(queries)}")
    qv = np.asarray(query_vecs, dtype=np.float32)
    qnorms = np.linalg.norm(qv, axis=1, keepdims=True)
    qnorms[qnorms == 0] = 1.0
    qv = qv / qnorms
    print(f"query vectors shape: {qv.shape}")

    # ---- retrieve top-K for each (scenario, character) ----
    results: dict[str, Any] = {}
    for si, sc in enumerate(all_scenarios):
        scen_id = sc["id"]
        scen_age = sc.get("age")
        if not isinstance(scen_age, int):
            print(f"  [warn] scenario {scen_id} missing int age={scen_age}; using ∞")
            scen_age_eff = 10**9
        else:
            scen_age_eff = scen_age

        q = qv[si]
        per_char: dict[str, Any] = {}
        for cid in char_ids:
            entry = char_index.get(cid)
            if entry is None:
                continue
            ids = entry["ids"]
            ages = entry["ages"]
            vecs = entry["vecs"]
            # Memories without a parseable age are excluded — they cannot be
            # safely placed on the timeline for this comparison.
            mask = np.array(
                [a is not None and a <= scen_age_eff for a in ages],
                dtype=bool,
            )
            if not mask.any():
                per_char[cid] = {"candidate_pool_size": 0, "topk": []}
                continue
            sub_vecs = vecs[mask]
            sub_ids = [ids[i] for i, m in enumerate(mask) if m]
            sub_ages = [ages[i] for i, m in enumerate(mask) if m]
            scores = sub_vecs @ q
            n = scores.shape[0]
            k = min(args.top_k, n)
            if k < n:
                part = np.argpartition(-scores, k - 1)[:k]
                top_idx = part[np.argsort(-scores[part])]
            else:
                top_idx = np.argsort(-scores)
            topk_records = []
            for rank, idx in enumerate(top_idx, start=1):
                topk_records.append(
                    {
                        "rank": rank,
                        "mem_id": sub_ids[int(idx)],
                        "score": float(scores[int(idx)]),
                        "mem_age": sub_ages[int(idx)],
                    }
                )
            per_char[cid] = {
                "candidate_pool_size": int(n),
                "topk": topk_records,
            }

        results[scen_id] = {
            "scenario_id": scen_id,
            "stage": sc.get("stage"),
            "age": scen_age,
            "diamonds_dimension": sc.get("diamonds_dimension"),
            "name": sc.get("name"),
            "query_text_preview": queries[si][:200],
            "by_character": per_char,
        }

    out = {
        "meta": {
            "embedding_model": args.model,
            "top_k": args.top_k,
            "scenarios_file": str(scenarios_file.relative_to(PROJECT_ROOT)),
            "characters_file": str(characters_file.relative_to(PROJECT_ROOT)),
            "embeddings_dir": str(emb_dir.relative_to(PROJECT_ROOT)) if emb_dir.is_relative_to(PROJECT_ROOT) else str(emb_dir),
            "query_fields": [
                "description_for_agent",
                "context_text",
                "trigger_event.message_content",
                "trigger_event.action_required",
            ],
            "age_filter_rule": "memory.age <= scenario.age (memory.age parsed from timeline 'age N' / 'N岁'; memories without parseable age are excluded)",
            "similarity": "cosine (L2-normalized dot product)",
            "ranking": "rank starts at 1, sorted by score desc",
            "characters": char_ids,
            "num_scenarios": len(all_scenarios),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }

    out_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"\nSAVED -> {out_file}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

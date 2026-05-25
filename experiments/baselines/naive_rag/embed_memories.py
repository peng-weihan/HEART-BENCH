"""
embed_memories.py — Pre-compute per-character memory embeddings for naive RAG.

For every character in ``benchmark/characters.json``, embed every memory in
``episodic_memory_set`` and persist the result to
``cache/embeddings/<char_id>.json``. The output schema matches what
``retrieve_scenarios.py`` expects:

  {
    "char_id": "CHAR_01",
    "model": "qwen3-embedding-4b",
    "dim": 2560,
    "count": 1000,
    "memories": [
      {"id": "MEM_...", "text": "<text fed to the embedder>", "embedding": [...]}
    ]
  }

Only ``mem.content_full`` is fed into the embedder — no metadata is mixed in.

Features:
  * Per-character idempotency: if ``<char_id>.json`` already exists, skip.
  * In-character batched API calls with intra-character concurrency.
  * Cross-character concurrency (process several characters in parallel).
  * Resumable via ``<char_id>.partial.json`` checkpoint files.
  * Exponential-backoff retries on transient API failures.

Environment / config:
  EMBEDDING_API_KEY  -> falls back to AIHUBMIX_API_KEY / API_KEY
  EMBEDDING_API_BASE -> falls back to AIHUBMIX_API_BASE / API_BASE
                       (default: https://api.openai.com/v1)
  EMBEDDING_MODEL    (default: qwen3-embedding-4b)
  EMBEDDING_BATCH_SIZE        (default: 16)
  EMBEDDING_INTRA_CONCURRENCY (default: 16)  # batches in flight per character
  EMBEDDING_INTER_CONCURRENCY (default:  4)  # characters in flight
  EMBEDDINGS_DIR     -> overrides the output directory

Usage:
  python embed_memories.py                       # all characters
  python embed_memories.py CHAR_01 CHAR_02       # only these ids
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths (aligned with the HEART-BENCH layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
CACHE_DIR = PROJECT_ROOT / "cache"

DEFAULT_CHARACTERS_FILE = BENCHMARK_DIR / "characters.json"
DEFAULT_OUTPUT_DIR = CACHE_DIR / "embeddings"


# ---------------------------------------------------------------------------
# .env loader (mirrors retrieve_scenarios.py / run_naive_rag.py)
# ---------------------------------------------------------------------------
def _load_dotenv(p: Path) -> None:
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


# ---------------------------------------------------------------------------
# Embedding API configuration (env-overridable)
# ---------------------------------------------------------------------------
DEFAULT_API_KEY = os.environ.get(
    "EMBEDDING_API_KEY",
    os.environ.get("AIHUBMIX_API_KEY", os.environ.get("API_KEY", "")),
)
DEFAULT_API_BASE = os.environ.get(
    "EMBEDDING_API_BASE",
    os.environ.get("AIHUBMIX_API_BASE", os.environ.get("API_BASE", "https://api.openai.com/v1")),
)
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-4b")

BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
INTRA_CONCURRENCY = int(os.environ.get("EMBEDDING_INTRA_CONCURRENCY", "16"))
INTER_CONCURRENCY = int(os.environ.get("EMBEDDING_INTER_CONCURRENCY", "4"))

MAX_RETRIES = 8
# Many embedding endpoints rate-limit on a ~60s window, so back off generously.
RETRY_BASE_DELAY = 5.0


# ---------------------------------------------------------------------------
# Text builder: how a memory becomes one embedding input
# ---------------------------------------------------------------------------
def build_memory_text(mem: dict) -> str:
    """Use only ``content_full`` — no metadata is concatenated."""
    return mem.get("content_full", "") or ""


# ---------------------------------------------------------------------------
# Embedding call with retry
# ---------------------------------------------------------------------------
def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
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
                f"    [retry {attempt}/{MAX_RETRIES}] embed error: {e}; "
                f"sleeping {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"embed_batch failed after {MAX_RETRIES} retries: {last_err}")


# ---------------------------------------------------------------------------
# Per-character processing
# ---------------------------------------------------------------------------
def process_character(
    client: OpenAI,
    character: dict,
    output_dir: Path,
    model: str,
) -> None:
    char_id: str = character["id"]
    memories: list[dict] = character.get("episodic_memory_set", []) or []
    final_path = output_dir / f"{char_id}.json"
    partial_path = output_dir / f"{char_id}.partial.json"

    if final_path.exists():
        print(f"[{char_id}] already done -> {final_path.name} (skip)")
        return

    if not memories:
        print(f"[{char_id}] no memories, skip")
        return

    print(
        f"[{char_id}] {len(memories)} memories, batch_size={BATCH_SIZE}, "
        f"intra_concurrency={INTRA_CONCURRENCY}"
    )

    texts = [build_memory_text(m) for m in memories]
    ids = [m.get("id", f"{char_id}_idx{i}") for i, m in enumerate(memories)]

    # Restore from checkpoint if shape still matches.
    vectors: list[list[float] | None]
    if partial_path.exists():
        try:
            data = json.loads(partial_path.read_text(encoding="utf-8"))
            cached_ids = data.get("ids", [])
            cached_vecs = data.get("vectors", [])
            if cached_ids == ids and len(cached_vecs) == len(ids):
                vectors = list(cached_vecs)
                done = sum(1 for v in vectors if v is not None)
                print(
                    f"  [{char_id}] resuming from partial cache: "
                    f"{done}/{len(ids)} already done"
                )
            else:
                vectors = [None] * len(ids)
                print(f"  [{char_id}] partial cache mismatched, starting from scratch")
        except Exception as e:  # noqa: BLE001
            print(f"  [{char_id}] failed to read partial cache ({e}); starting fresh")
            vectors = [None] * len(ids)
    else:
        vectors = [None] * len(ids)

    state_lock = threading.Lock()
    save_lock = threading.Lock()
    save_every_n_batches = max(1, INTRA_CONCURRENCY)
    batches_since_save = {"n": 0}

    def save_partial() -> None:
        with save_lock:
            tmp = partial_path.with_suffix(".tmp")
            with state_lock:
                snapshot = list(vectors)
            tmp.write_text(
                json.dumps({"ids": ids, "vectors": snapshot}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(partial_path)

    pending_batches: list[list[int]] = []
    for batch_start in range(0, len(ids), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(ids))
        missing_idx = [i for i in range(batch_start, batch_end) if vectors[i] is None]
        if missing_idx:
            pending_batches.append(missing_idx)

    total = len(ids)
    if not pending_batches:
        print(f"  [{char_id}] all batches already cached, skipping API calls")
    else:
        start_time = time.time()
        completed_batches = 0

        def run_batch(missing_idx: list[int]) -> int:
            batch_texts = [texts[i] for i in missing_idx]
            batch_vecs = embed_batch(client, model, batch_texts)
            with state_lock:
                for i, v in zip(missing_idx, batch_vecs):
                    vectors[i] = v
            return len(missing_idx)

        with ThreadPoolExecutor(max_workers=INTRA_CONCURRENCY) as pool:
            futures = {pool.submit(run_batch, b): b for b in pending_batches}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [{char_id}] batch failed: {e}", file=sys.stderr)
                    raise
                completed_batches += 1
                batches_since_save["n"] += 1
                if batches_since_save["n"] >= save_every_n_batches:
                    batches_since_save["n"] = 0
                    save_partial()
                with state_lock:
                    done = sum(1 for v in vectors if v is not None)
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else float("inf")
                if completed_batches % 4 == 0 or done == total:
                    print(
                        f"  [{char_id}] {done}/{total}  ({100*done/total:5.1f}%)  "
                        f"elapsed={elapsed:6.1f}s  rate={rate:6.2f}/s  eta={eta:6.1f}s"
                    )

        save_partial()

    if any(v is None for v in vectors):
        raise RuntimeError(f"[{char_id}] some vectors missing after run")

    dim = len(vectors[0]) if vectors and vectors[0] is not None else 0
    output: dict[str, Any] = {
        "char_id": char_id,
        "model": model,
        "dim": dim,
        "count": total,
        "memories": [
            {"id": ids[i], "text": texts[i], "embedding": vectors[i]}
            for i in range(total)
        ],
    }
    final_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"[{char_id}] DONE -> {final_path}  (dim={dim}, count={total})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--characters-file",
        type=Path,
        default=DEFAULT_CHARACTERS_FILE,
        help=f"Path to characters.json (default: {DEFAULT_CHARACTERS_FILE})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("EMBEDDINGS_DIR", str(DEFAULT_OUTPUT_DIR))),
        help=f"Directory to write <CHAR_ID>.json files (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument(
        "char_ids",
        nargs="*",
        help="Optional subset of character ids to process (default: all).",
    )
    args = ap.parse_args()

    characters_file: Path = args.characters_file
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not characters_file.exists():
        print(f"ERROR: characters file not found: {characters_file}", file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        print(
            "ERROR: no embedding API key found. Set EMBEDDING_API_KEY / API_KEY / "
            "AIHUBMIX_API_KEY in env or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(2)

    raw = json.loads(characters_file.read_text(encoding="utf-8"))
    characters = raw.get("characters", [])
    print(f"loaded {len(characters)} characters from {characters_file}")
    print(
        f"model={args.model}  api_base={args.api_base}  output_dir={output_dir}"
    )

    target_ids = set(args.char_ids)
    if target_ids:
        characters = [c for c in characters if c["id"] in target_ids]
        print(f"filtered to {len(characters)} characters: {sorted(target_ids)}")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    print(
        f"inter_concurrency={INTER_CONCURRENCY}  "
        f"intra_concurrency={INTRA_CONCURRENCY}  batch_size={BATCH_SIZE}"
    )

    if INTER_CONCURRENCY <= 1:
        for ch in characters:
            try:
                process_character(client, ch, output_dir, args.model)
            except Exception as e:  # noqa: BLE001
                print(f"[{ch.get('id','?')}] FAILED: {e}", file=sys.stderr)
                continue
    else:
        with ThreadPoolExecutor(max_workers=INTER_CONCURRENCY) as pool:
            futures = {
                pool.submit(process_character, client, ch, output_dir, args.model): ch
                for ch in characters
            }
            for fut in as_completed(futures):
                ch = futures[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[{ch.get('id','?')}] FAILED: {e}", file=sys.stderr)

    print("ALL DONE")


if __name__ == "__main__":
    main()

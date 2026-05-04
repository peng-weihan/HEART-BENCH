"""
Vector embedding memory retrieval module.

Computes embeddings via an OpenAI-compatible API and retrieves a character's
episodic memories by cosine similarity. Embeddings are cached under
cache/embeddings/{char_id}.json to avoid repeated API calls.
"""

from __future__ import annotations

import json
import math
import os
import random as _random
import re
import threading
from pathlib import Path

from openai import OpenAI


class EmbeddingClient:
    """Wrapper around an OpenAI-compatible embedding endpoint."""

    def __init__(self, api_key: str, api_base: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of strings; returns vectors in input order."""
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # Sort by `index` so the response order matches the input order.
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_memory_text(mem: dict) -> str:
    """Compose the embedding text for a single memory record."""
    timeline = mem.get("timeline", "")
    summary = mem.get("content_summary", "")
    triggers = ", ".join(mem.get("triggers", []))
    tags = ", ".join(mem.get("relevance_tags", []))
    return f"[{timeline}] {summary} | triggers: {triggers} | tags: {tags}"


# Mapping from the timeline prefix used in the dataset to a stage key.
_TIMELINE_PREFIX_TO_STAGE = {
    "childhood": "childhood",
    "adolescence": "adolescence",
    "college": "college",
    "early_career": "early_career",
    "growth": "growth",
    "recent": "recent",
}

# Chronological order of stages (lower index = earlier in life).
STAGE_ORDER = ["childhood", "adolescence", "college", "early_career", "growth", "recent"]


def _memory_stage(mem: dict) -> str | None:
    """Resolve the stage key from a memory's timeline field, e.g. 'childhood(age 6)' → 'childhood'."""
    timeline = mem.get("timeline", "")
    for prefix, stage in _TIMELINE_PREFIX_TO_STAGE.items():
        if timeline.startswith(prefix):
            return stage
    return None


# Match age tokens like "age 10" or "(age 10)" in timeline strings.
_EPISODE_AGE_RE = re.compile(r"age\s*(\d{1,3})", re.IGNORECASE)


def parse_episode_age_from_timeline(timeline: str) -> int | None:
    """Parse narrative age in years from a timeline string, e.g. 'childhood(age 10)' → 10."""
    if not timeline:
        return None
    m = _EPISODE_AGE_RE.search(timeline)
    if not m:
        return None
    return int(m.group(1))


def _allowed_stages(current_stage: str) -> set[str]:
    """Return all stages a character could plausibly recall at ``current_stage`` (self + earlier)."""
    if current_stage not in STAGE_ORDER:
        return set(STAGE_ORDER)  # unknown stage → no truncation
    cutoff = STAGE_ORDER.index(current_stage)
    return set(STAGE_ORDER[: cutoff + 1])


class MemoryIndex:
    """Per-character vector index supporting build, cache, and query."""

    CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "embeddings"

    def __init__(self, embedding_client: EmbeddingClient):
        self.client = embedding_client
        # char_id -> list of (memory_dict, embedding_vector)
        self._indexes: dict[str, list[tuple[dict, list[float]]]] = {}
        self._lock = threading.Lock()

    def build(self, character) -> None:
        """Build the embedding index for one character (loads cache if available)."""
        char_id = character.id
        memories = character.episodic_memory

        if not memories:
            with self._lock:
                self._indexes[char_id] = []
            return

        # Try the disk cache first.
        cache_path = self.CACHE_DIR / f"{char_id}.json"
        cached = self._load_cache(cache_path, memories)
        if cached is not None:
            with self._lock:
                self._indexes[char_id] = cached
            print(f"  [MemoryIndex] {char_id}: loaded {len(cached)} embeddings from cache")
            return

        # Need to call the API (with batched, resumable checkpoints).
        texts = [_build_memory_text(m) for m in memories]
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))
        partial_path = cache_path.with_suffix(".partial.json")
        memory_ids = [m.get("id", "") for m in memories]
        vectors = self._load_partial_cache(partial_path, memory_ids)
        resumed = sum(1 for v in vectors if v is not None)
        if resumed > 0:
            print(
                f"  [MemoryIndex] {char_id}: resume from partial checkpoint "
                f"({resumed}/{len(texts)})..."
            )
        else:
            print(
                f"  [MemoryIndex] {char_id}: computing {len(texts)} embeddings via API "
                f"(batch={batch_size})..."
            )

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            missing_idx = [i for i in range(start, end) if vectors[i] is None]
            if not missing_idx:
                continue
            batch_texts = [texts[i] for i in missing_idx]
            batch_vectors = self.client.embed(batch_texts)
            for i, vec in zip(missing_idx, batch_vectors):
                vectors[i] = vec
            self._save_partial_cache(partial_path, memory_ids, vectors)

        if any(v is None for v in vectors):
            raise RuntimeError(f"Incomplete embedding vectors for character: {char_id}")

        finalized_vectors = [v for v in vectors if v is not None]

        index = list(zip(memories, finalized_vectors))
        with self._lock:
            self._indexes[char_id] = index

        # Write the final cache.
        self._save_cache(cache_path, memories, finalized_vectors)
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass

    def query(
        self, char_id: str, text: str, top_k: int = 3, stage: str | None = None
    ) -> list[dict]:
        """Embed ``text``, optionally truncate by life stage, and return top-k by cosine.

        Args:
            char_id: character ID.
            text: query text (typically scenario context + trigger).
            top_k: number of memories to return.
            stage: current scenario stage; when set, only memories from this stage or earlier are considered.
        """
        index = self._indexes.get(char_id, [])
        if not index:
            return []

        # Stage truncation: keep only memories from the current stage or earlier.
        if stage is not None:
            allowed = _allowed_stages(stage)
            index = [
                (mem, vec) for mem, vec in index
                if _memory_stage(mem) in allowed
            ]
            if not index:
                return []

        query_vec = self.client.embed_single(text)
        scored = [
            (_cosine_similarity(query_vec, vec), mem)
            for mem, vec in index
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    # ---- cache helpers ----

    def _load_cache(
        self, cache_path: Path, memories: list[dict]
    ) -> list[tuple[dict, list[float]]] | None:
        """Load cache iff the memory ID list matches the on-disk record; otherwise return None."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Sanity-check via the memory ID list.
        cached_ids = cache_data.get("memory_ids", [])
        current_ids = [m.get("id", "") for m in memories]
        if cached_ids != current_ids:
            return None

        vectors = cache_data.get("vectors", [])
        if len(vectors) != len(memories):
            return None

        return list(zip(memories, vectors))

    def _save_cache(
        self, cache_path: Path, memories: list[dict], vectors: list[list[float]]
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "memory_ids": [m.get("id", "") for m in memories],
            "vectors": vectors,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def _load_partial_cache(
        self, partial_path: Path, memory_ids: list[str]
    ) -> list[list[float] | None]:
        """Load the in-progress checkpoint; return an all-None list if it doesn't match."""
        empty: list[list[float] | None] = [None] * len(memory_ids)
        if not partial_path.exists():
            return empty
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                partial_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return empty

        if partial_data.get("memory_ids", []) != memory_ids:
            return empty
        vectors = partial_data.get("vectors", [])
        if not isinstance(vectors, list) or len(vectors) != len(memory_ids):
            return empty
        return vectors

    def _save_partial_cache(
        self, partial_path: Path, memory_ids: list[str], vectors: list[list[float] | None]
    ) -> None:
        """Persist in-progress batches so embedding can resume after an interruption."""
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "memory_ids": memory_ids,
            "vectors": vectors,
        }
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)


def create_memory_index() -> MemoryIndex:
    """Create ``MemoryIndex`` from environment variables. Embedding is mandatory; no silent fallback."""
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    api_base = os.getenv("EMBEDDING_API_BASE", "https://aihubmix.com/v1")
    model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

    if not api_key:
        raise RuntimeError(
            "EMBEDDING_API_KEY is not set. Default benchmark retrieval uses embedding only; "
            "set EMBEDDING_API_KEY (and EMBEDDING_API_BASE if needed) in .env, or pass "
            "--random / --mem0 / --pre-annotated for an alternate retrieval mode."
        )

    client = EmbeddingClient(api_key=api_key, api_base=api_base, model=model)
    return MemoryIndex(client)


class RandomMemoryRetriever:
    """Control retriever: stage-truncate then sample uniformly at random (no embedding)."""

    def __init__(self):
        # char_id -> list of memory dicts
        self._memories: dict[str, list[dict]] = {}

    def build(self, character) -> None:
        self._memories[character.id] = list(character.episodic_memory)
        print(f"  [RandomRetriever] {character.id}: {len(character.episodic_memory)} memories loaded")

    def query(
        self, char_id: str, text: str, top_k: int = 3, stage: str | None = None
    ) -> list[dict]:
        memories = self._memories.get(char_id, [])
        if not memories:
            return []

        # Stage truncation: keep only memories from the current stage or earlier.
        if stage is not None:
            allowed = _allowed_stages(stage)
            memories = [m for m in memories if _memory_stage(m) in allowed]

        if not memories:
            return []

        return _random.sample(memories, min(top_k, len(memories)))

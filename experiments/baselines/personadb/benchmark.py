"""
Persona-DB integration for the personality benchmark.
Paper: "Persona-DB: Efficient Large Language Model Personalization for Response
Customization with Retrieved Augmentation" (COLING 2025)
Repo:  https://github.com/chenkaisun/Persona-DB

Persona-DB core idea: two-level persona hierarchy per user.
  L0 (raw)     -- individual episodic content_full embedded directly
  L1 (refined) -- LLM-condensed persona trait statements (optional)

Retrieval: embedding cosine top-k over the L0+L1 pool, filtered by the
benchmark life-stage causal constraint (no-future-memory semantics).

We vendor the retrieval logic in-process using sentence-transformers --
no external service required.

Env vars:
  PERSONADB_EMBED_MODEL  -- sentence-transformers model (default all-MiniLM-L6-v2)
  PERSONADB_STORE_DIR    -- directory placeholder for the in-process store
  PERSONADB_CONDENSE     -- set 1 to enable LLM L1 condensation at ingest (default off)
  PERSONADB_LLM_MODEL    -- LLM model for condensation
  PERSONADB_LLM_API_KEY  -- API key for condensation LLM
  PERSONADB_LLM_API_BASE -- API base for condensation LLM
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.memory_retrieval import _memory_stage  # type: ignore
from mem0_integration.benchmark import allowed_memory_stages_for_scenario  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORE_DIR = str(_REPO_ROOT / "cache" / ".personadb_store")

_store_cache: dict[str, "PersonaDBStore"] = {}
_store_lock = threading.Lock()


# ---------------------------------------------------------------------------
# In-process vector store
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    doc_id: str
    char_id: str
    mem_stage: str
    mem_id: str
    timeline: str
    content: str
    level: str        # "l0" | "l1"
    vector: list[float] = field(default_factory=list)


class PersonaDBStore:
    """
    In-process persona store backed by sentence-transformers embeddings.

    L0 -- raw episodic content_full (one entry per benchmark memory)
    L1 -- LLM-condensed persona statements (optional; PERSONADB_CONDENSE=1)
    """

    def __init__(self, embed_model: str, persist_dir: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(embed_model)
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()

    def add(self, entry: _Entry) -> None:
        vec = self._model.encode(entry.content, normalize_embeddings=True).tolist()
        entry.vector = vec
        with self._lock:
            self._entries.append(entry)

    def delete_char(self, char_id: str) -> None:
        with self._lock:
            self._entries = [e for e in self._entries if e.char_id != char_id]

    def query(
        self,
        char_id: str,
        query_text: str,
        allowed_stages: frozenset,
        n_results: int,
    ) -> list[tuple[float, _Entry]]:
        q_vec = self._model.encode(query_text, normalize_embeddings=True).tolist()
        candidates: list[tuple[float, _Entry]] = []
        with self._lock:
            snapshot = list(self._entries)
        for e in snapshot:
            if e.char_id != char_id:
                continue
            if e.mem_stage != "unknown" and e.mem_stage not in allowed_stages:
                continue
            dot = sum(a * b for a, b in zip(q_vec, e.vector))
            candidates.append((dot, e))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:n_results]

    def count(self, char_id: str) -> int:
        with self._lock:
            return sum(1 for e in self._entries if e.char_id == char_id)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PersonaDBConfig:
    store: PersonaDBStore
    condense: bool
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = ""


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def make_personadb() -> PersonaDBConfig:
    """Construct a PersonaDB in-process store from environment variables."""
    embed_model = os.getenv("PERSONADB_EMBED_MODEL", "all-MiniLM-L6-v2").strip()
    store_dir = os.getenv("PERSONADB_STORE_DIR", _DEFAULT_STORE_DIR).strip()
    condense = _parse_bool_env("PERSONADB_CONDENSE", False)

    key = embed_model + "\x00" + store_dir
    with _store_lock:
        if key not in _store_cache:
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            _store_cache[key] = PersonaDBStore(embed_model, store_dir)
        store = _store_cache[key]

    llm_key = (
        os.getenv("PERSONADB_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
        or ""
    )
    llm_base = (
        os.getenv("PERSONADB_LLM_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("API_BASE")
        or ""
    ).rstrip("/")
    llm_model = (
        os.getenv("PERSONADB_LLM_MODEL")
        or os.getenv("MODEL_NAME")
        or "gpt-4.1-mini"
    )
    return PersonaDBConfig(
        store=store,
        condense=condense,
        llm_api_key=llm_key,
        llm_api_base=llm_base,
        llm_model=llm_model,
    )


# ---------------------------------------------------------------------------
# L1 condensation (optional LLM step)
# ---------------------------------------------------------------------------

def _condense_to_persona_statements(
    content: str,
    cfg: PersonaDBConfig,
) -> list[str]:
    """
    Persona-DB L1: distill raw episodic memory into persona trait statements.
    Mirrors the paper's personality-based condensation step.
    """
    import json
    import urllib.request

    prompt = (
        "Extract 2-3 personality trait statements from this first-person episodic memory. "
        "Each statement should start with 'View toward' or 'Tends to', under 20 words. "
        'Output only a JSON array, e.g.: ["View toward criticism: very sensitive", '
        '"Tends to self-attack"]\n\n'
        "Memory:\n" + content[:800]
    )
    payload = json.dumps({
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }, ensure_ascii=False).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + cfg.llm_api_key,
    }
    url = cfg.llm_api_base + "/chat/completions"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = "\n".join(line for line in text.split("\n") if not line.startswith("```"))
        statements = json.loads(text)
        if isinstance(statements, list):
            return [str(s) for s in statements if s]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_character_episodes_personadb(
    cfg: PersonaDBConfig,
    char_id: str,
    episodes: list[dict],
    *,
    reset_user: bool = False,
    show_progress: bool = True,
) -> int:
    """
    Load episodic_memory_set into PersonaDB for one character.

    L0: each memory's content_full is embedded directly.
    L1: if cfg.condense is True, additionally embed LLM-generated persona statements.
    """
    if reset_user:
        cfg.store.delete_char(char_id)

    total = len(episodes)
    step = max(1, total // 50) if total else 1
    iter_ep: Any = episodes
    simple_progress = False
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iter_ep = tqdm(episodes, desc="PersonaDB ingest " + char_id, unit="ep", leave=True)
        except ImportError:
            simple_progress = bool(total)

    n = 0
    for idx, mem in enumerate(iter_ep, 1):
        if simple_progress and (idx % step == 0 or idx == total):
            print("  " + char_id + ": " + str(idx) + "/" + str(total) + " episodes...", flush=True)

        text = (mem.get("content_full") or "").strip()
        if not text:
            continue

        mid = str(mem.get("id", "") or "")
        timeline = str(mem.get("timeline", "") or "")
        ms = _memory_stage(mem) or "unknown"

        cfg.store.add(_Entry(
            doc_id=char_id + "__l0__" + mid,
            char_id=char_id,
            mem_stage=ms,
            mem_id=mid,
            timeline=timeline,
            content=text,
            level="l0",
        ))
        n += 1

        if cfg.condense:
            statements = _condense_to_persona_statements(text, cfg)
            for i, stmt in enumerate(statements):
                cfg.store.add(_Entry(
                    doc_id=char_id + "__l1__" + mid + "__" + str(i),
                    char_id=char_id,
                    mem_stage=ms,
                    mem_id=mid,
                    timeline=timeline,
                    content=stmt,
                    level="l1",
                ))

    return n


def personadb_retrieve_for_prompt(
    cfg: PersonaDBConfig,
    *,
    char_id: str,
    query: str,
    scenario_stage: str | None,
    top_k: int = 15,
    search_limit: int = 100,
) -> list[dict]:
    """
    Retrieve top-k memories from PersonaDB with life-stage causal filtering.

    Returns dicts compatible with build_prompt():
      {"id": str, "timeline": str, "content_full": str, "content_summary": str}

    When both L0 and L1 entries point to the same source memory, keeps L0 (full content).
    """
    allowed = allowed_memory_stages_for_scenario(scenario_stage)
    raw = cfg.store.query(
        char_id=char_id,
        query_text=query,
        allowed_stages=allowed,
        n_results=search_limit,
    )

    dedup: list[dict] = []
    seen_mem_ids: set[str] = set()
    for _score, entry in raw:
        if entry.mem_id in seen_mem_ids:
            continue
        seen_mem_ids.add(entry.mem_id)
        dedup.append({
            "id": entry.mem_id,
            "timeline": entry.timeline,
            "content_full": entry.content,
            "content_summary": entry.content,
        })
        if len(dedup) >= top_k:
            break

    return dedup

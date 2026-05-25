"""Naive RAG baseline.

The Naive RAG retrieval logic (embedding-based cosine top-k) lives in
``utils.memory_retrieval`` because it is also the default retriever
used by ``experiments/scripts/main.py``. This module re-exports that
implementation so the baseline has its own importable surface.

Driver scripts:
  - ``experiments/baselines/naive_rag/retrieve_scenarios.py``
      Pre-computes per-scenario top-K memory retrieval and writes
      ``cache/retrieval/scenario_topk.json``.
  - ``experiments/scripts/run_naive_rag.py``
      Consumes that file and runs the MCQ benchmark with an LLM.
"""
from utils.memory_retrieval import (
    EmbeddingClient,
    RandomMemoryRetriever,
    create_memory_index,
)

__all__ = [
    "EmbeddingClient",
    "RandomMemoryRetriever",
    "create_memory_index",
]

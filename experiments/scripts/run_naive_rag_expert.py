"""Wrapper: run naive_rag against the *expert* MCQ subset (69 Qs).
Reuses experiments/run_naive_rag.py end-to-end; only swaps the MCQ file
and the results root so it never collides with the main benchmark output.

Usage example:
  python3 experiments/run_naive_rag_expert.py --model deepseek-v3.2 \
      --workers 8 --top-k 30 --temperature 0
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_naive_rag as base  # noqa: E402

# --- override file locations BEFORE main() runs ---------------------------
PROJECT_ROOT = HERE.parent
base.MCQ_PATH = PROJECT_ROOT.parent / "benchmark" / "mcq.json"  # flat layout: no expert subset
base.RESULTS_ROOT = PROJECT_ROOT.parent / "experiments" / "results" / "naive_rag_expert"
base.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    base.main()

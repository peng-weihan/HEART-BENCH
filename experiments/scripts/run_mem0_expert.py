"""Wrapper: run mem0 against the *expert* MCQ subset (69 Qs).
Reuses experiments/run_mem0.py end-to-end; only swaps the MCQ file
and the results root so it never collides with the main benchmark output.

Usage:
  python3 experiments/run_mem0_expert.py --model deepseek-v3.2 \
      --workers 24 --top-k 150 --temperature 0
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_mem0 as base  # noqa: E402

PROJECT_ROOT = HERE.parent
base.MCQ_PATH = PROJECT_ROOT.parent / "benchmark" / "mcq.json"  # flat layout: no expert subset
base.RESULTS_ROOT = PROJECT_ROOT.parent / "experiments" / "results" / "mem0_expert"
base.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    base.main()

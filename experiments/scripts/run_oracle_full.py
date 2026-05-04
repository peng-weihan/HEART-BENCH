"""run_oracle_full.py — naive_rag-style runner that uses oracle (annotated) memories as references.

Key differences:
  - No retrieval: uses the activated_memories from
    benchmark/activated_memories_step2.json
    as the reference set (~50 per scenario).
  - Looks up each memory_id in characters.json to recover the original content_full / timeline.
  - The prompt template is identical to run_naive_rag.py, with the same mem_id
    sanitisation (trait tags stripped).
  - --top-k limits the maximum number of oracle memories per question (None / 0 = use all).

Output: experiments/results/oracle_full/<model>/{predictions,summary}_top<k>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_naive_rag as base  # noqa: E402

PROJECT_ROOT = HERE.parent
ANN_PATH = (
    PROJECT_ROOT.parent
    / "benchmark"
    / "activated_memories"
    / "claude-sonnet-4-6"
    / "activated_memories_final.json"
)


# --- override ANN-driven memory selection -----------------------------------
import json  # noqa: E402

_ANN_CACHE: dict | None = None


def _load_ann() -> dict:
    """Returns nested dict: {char_id: {scenario_id: [mem_id, mem_id, ...]}}"""
    global _ANN_CACHE
    if _ANN_CACHE is not None:
        return _ANN_CACHE
    raw = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    for cid, by_scen in raw["annotations"].items():
        d: dict[str, list[str]] = {}
        for sid, sa in by_scen.items():
            ams = sa.get("activated_memories") or []
            d[sid] = [m["memory_id"] for m in ams if m.get("memory_id")]
        out[cid] = d
    _ANN_CACHE = out
    return out


def get_topk_memories_oracle(
    retrieval, mem_lookup, scenario_id, char_id, top_k
):
    """Replacement for base.get_topk_memories: pulls memory ids from annotation."""
    ann = _load_ann()
    ids = (ann.get(char_id) or {}).get(scenario_id) or []
    if top_k and top_k > 0:
        ids = ids[:top_k]
    out = []
    for mid in ids:
        m = mem_lookup.get(mid)
        if m is not None:
            out.append(m)
    return out


# Monkey-patch the base module so process_one() uses oracle source.
base.get_topk_memories = get_topk_memories_oracle
base.RESULTS_ROOT = PROJECT_ROOT.parent / "experiments" / "results" / "oracle_full"
base.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# We still call load_inputs to get characters/scenarios/questions/mem_lookup
# but the retrieval json is unused; just keep loading to avoid changing signatures.

if __name__ == "__main__":
    base.main()

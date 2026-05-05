# HEART-BENCH

HEART-BENCH is a benchmark for evaluating whether LLM-based agents can act in a **human-like** way: given only a fictional character's *raw episodic memories* (no pre-summarised personality labels), can a model infer that character's personality and produce in-character reactions to new scenarios?

## Overview

![Pipeline overview](assets/pipeline.svg)


## Data

The released benchmark data is hosted on Hugging Face:

**🤗 [https://huggingface.co/datasets/HEART-BENCH/HEART-BENCH](https://huggingface.co/datasets/HEART-BENCH/HEART-BENCH)**

A local copy of the released artefacts is also mirrored under `benchmark/` in this repository.

The benchmark covers:

- **Character grounding from raw memories** — no Big-Five scores, no value-scale labels, no personality descriptions are provided to the model under test.
- **Scenario-based behavioural probing** — characters are placed in standardised scenarios (designed under the DIAMONDS situation taxonomy) and asked to react.
- **Multiple-choice + open-ended evaluation** — both an MCQ track (single behavioural choice) and a consciousness-narrative track (integrated emotional / reasoning / value statements).
- **Memory-system baselines** — the same model is evaluated under several memory configurations (no memory, naive RAG, an intelligent-memory baseline, a persona-DB baseline, oracle full memory).

---

## Repository layout

```
benchmark/         Released benchmark artefacts (characters, scenarios, MCQs, GTs, activated memories)
constructions/    Construction-time scripts (character / scenario / annotation pipelines)
prompts/          Prompt templates used during construction and annotation
experiments/
    scripts/      Runners for each baseline (no-memory, naive RAG, intelligent memory, persona-DB, oracle)
    baselines/   Per-baseline working directories
    evaluation/  Evaluation pipeline and metric definitions
    results/      Aggregated result tables
docs/             Design notes (scenario design, gateway, evaluation rationale)
utils/            Shared helpers (HTTP client, schema, retrieval, paths)
```

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Per-baseline extras (only install what you need):

```bash
pip install -r experiments/baselines/mem0/requirements.txt        # intelligent-memory baseline
pip install -r experiments/baselines/personadb/requirements.txt   # persona-DB baseline
```

### 2. Configure the LLM gateway

Copy `.env.example` to `.env` and fill in your own credentials. **Do not commit `.env`.** The harness uses an OpenAI-compatible HTTP gateway; any provider that exposes `/v1/chat/completions` works.

```bash
cp .env.example .env
# then edit API_KEY, API_BASE, MODEL_NAME, EMBEDDING_* …
```

See `docs/llm_http_gateway.md` for the contract the gateway must satisfy.

### 3. Run a baseline

```bash
# No-memory baseline (model sees only the scenario)
python experiments/scripts/run_baseline_model_only.py

# Naive RAG over raw episodic memories
python experiments/scripts/run_naive_rag.py --top_k 30

# Intelligent-memory baseline
python experiments/scripts/run_mem0.py --top_k 150

# Persona-DB baseline
python experiments/scripts/run_personadb_mcq.py --top_k 30

# Oracle (full memory in context)
python experiments/scripts/run_oracle_full.py
```

Each runner writes a JSON file under `experiments/results/<baseline>/<model>/`. Runs are checkpointed and resumable.

### 4. Evaluate

```bash
python experiments/evaluation/evaluate_results.py --input experiments/results/<baseline>/<model>/<run>.json
```

See `experiments/evaluation/README.md` for the full set of metrics (behavioural human-likeness, consciousness human-likeness, combined score).

---

## Benchmark artefacts

The released benchmark (under `benchmark/`) contains:

| File | Description |
|---|---|
| `characters.json` | Fictional characters, each with raw episodic memories only |
| `scenarios.json` | DIAMONDS-balanced scenarios |
| `mcq.json` | Multiple-choice questions (one per character × scenario cell) |
| `ground_truth.json` | Expert-annotated reference behaviour and rationale |
| `activated_memories_step1.json` / `_step2.json` | Two-step memory activation annotations |

---

## Reproducing the result tables

`experiments/results/RESULTS.md` reports headline numbers. To reproduce:

1. Run all baselines for the target model (`experiments/scripts/run_baseline_all_models.sh` is a convenience wrapper).
2. Run the evaluator on each output file.
3. Aggregate using the evaluator's summary mode.

Determinism notes: temperature is non-zero by default; reported numbers are the average of *N* runs where *N* is documented per table.

---

## License

Released under the Apache License 2.0. See `LICENSE`.

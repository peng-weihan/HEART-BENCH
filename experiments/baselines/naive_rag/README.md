# Naive RAG baseline

A simple retrieval-augmented baseline: for every MCQ, embed the scenario,
retrieve the top-K most semantically similar episodic memories from the
target character's memory set (filtered to memories the character could
plausibly have at the scenario's age), and feed them to the LLM along with
the scenario and answer options.

This directory holds the two **one-off preprocessing scripts** that produce
the retrieval cache consumed by
[`experiments/scripts/run_naive_rag.py`](../../scripts/run_naive_rag.py):

| Script | Purpose | Output |
|---|---|---|
| `embed_memories.py` | Embed every memory in `episodic_memory_set` for every character | `cache/embeddings/<CHAR_ID>.json` |
| `retrieve_scenarios.py` | For each scenario, score every character's memories against the scenario query and keep the top-K | `cache/retrieval/scenario_topk.json` |

Both steps are **idempotent and resumable** — running them again after a
partial / failed run will skip work that is already on disk.

## Pipeline overview

```
benchmark/characters.json
        │
        │  embed_memories.py
        │  (qwen3-embedding-4b on mem.content_full only)
        ▼
cache/embeddings/<CHAR_ID>.json
        │
benchmark/scenarios.json ──┐
                           │  retrieve_scenarios.py
                           │  (cosine on L2-normalized vectors,
                           │   filtered by mem.age <= scenario.age)
                           ▼
                cache/retrieval/scenario_topk.json
                           │
benchmark/mcq.json ────────┤
                           │  experiments/scripts/run_naive_rag.py
                           │  (looks mem_id up in characters.json,
                           │   builds the prompt, calls the LLM)
                           ▼
experiments/results/naive_rag/<model>/predictions_topK.json
```

## Prerequisites

- Python deps from the project root: `pip install -r requirements.txt`
  (uses `openai`, `numpy`).
- An OpenAI-compatible embedding endpoint. Configure it in the project-root
  `.env` (copy `.env.example` first):

  ```bash
  # embedding side (used by embed_memories.py and retrieve_scenarios.py)
  EMBEDDING_API_KEY=sk-...
  EMBEDDING_API_BASE=https://...        # e.g. https://aihubmix.com/v1
  EMBEDDING_MODEL=qwen3-embedding-4b    # default

  # LLM side (used by run_naive_rag.py)
  API_KEY=sk-...
  API_BASE=https://...
  ```

  If `EMBEDDING_API_KEY` / `EMBEDDING_API_BASE` are unset, the scripts
  fall back to `AIHUBMIX_API_KEY` / `AIHUBMIX_API_BASE`, and finally to
  `API_KEY` / `API_BASE`, so a single key can drive both sides.

## Step 1 — embed memories

```bash
python experiments/baselines/naive_rag/embed_memories.py
```

Defaults (all overridable via flags or env vars):

| Flag / env | Default | Meaning |
|---|---|---|
| `--characters-file` | `benchmark/characters.json` | Source memories |
| `--output-dir` / `EMBEDDINGS_DIR` | `cache/embeddings/` | Where `<CHAR_ID>.json` files are written |
| `--model` / `EMBEDDING_MODEL` | `qwen3-embedding-4b` | Embedding model name |
| `EMBEDDING_BATCH_SIZE` | `16` | Memories per API call |
| `EMBEDDING_INTRA_CONCURRENCY` | `16` | Concurrent batches per character |
| `EMBEDDING_INTER_CONCURRENCY` | `4` | Characters processed in parallel |

Only `mem.content_full` is fed to the embedder — no metadata is concatenated.

Selective re-run (process only some characters):

```bash
python experiments/baselines/naive_rag/embed_memories.py CHAR_01 CHAR_02
```

Resumability:

- If `<CHAR_ID>.json` already exists, the character is skipped.
- Mid-character progress is checkpointed to `<CHAR_ID>.partial.json`; if a
  run is interrupted, the next invocation picks up from there.

Output schema (one file per character):

```json
{
  "char_id": "CHAR_01",
  "model": "qwen3-embedding-4b",
  "dim": 2560,
  "count": 1000,
  "memories": [
    {"id": "MEM_CHAR_01_...", "text": "<content_full>", "embedding": [/* dim floats */]}
  ]
}
```

## Step 2 — retrieve scenario top-K

```bash
python experiments/baselines/naive_rag/retrieve_scenarios.py --top-k 30
```

Defaults:

| Flag / env | Default | Meaning |
|---|---|---|
| `--scenarios-file` | `benchmark/scenarios.json` | Scenarios to retrieve against |
| `--characters-file` | `benchmark/characters.json` | Source for `mem.age` parsing |
| `--embeddings-dir` / `EMBEDDINGS_DIR` | `cache/embeddings/` | Per-character embedding files from step 1 |
| `--out-file` | `cache/retrieval/scenario_topk.json` | Single compact JSON output |
| `--top-k` / `RETRIEVAL_TOP_K` | `30` | Memories kept per (scenario, character) |
| `--model` / `EMBEDDING_MODEL` | `qwen3-embedding-4b` | Must match step 1's model |

Retrieval logic:

- **Query text**: `description_for_agent` + `context_text` +
  `trigger_event.message_content` + `trigger_event.action_required`,
  joined with newlines.
- **Candidate pool**: memories with `mem.age <= scenario.age` only.
  `mem.age` is parsed from the `timeline` string (`"Childhood (age 6)"`
  → `6`; Chinese `"6岁"` form also supported). Memories whose age cannot
  be parsed are excluded — they cannot be safely placed on the timeline.
- **Similarity**: cosine via L2-normalized dot product.
- **Ranking**: sorted by score desc, `rank` starts at 1.

Output schema (one compact JSON file):

```json
{
  "meta": {
    "embedding_model": "qwen3-embedding-4b",
    "top_k": 30,
    "scenarios_file": "benchmark/scenarios.json",
    "characters_file": "benchmark/characters.json",
    "embeddings_dir": "cache/embeddings",
    "query_fields": [
      "description_for_agent", "context_text",
      "trigger_event.message_content", "trigger_event.action_required"
    ],
    "age_filter_rule": "memory.age <= scenario.age (...)",
    "similarity": "cosine (L2-normalized dot product)",
    "characters": ["CHAR_01", ..., "CHAR_11"],
    "num_scenarios": 64,
    "generated_at_utc": "..."
  },
  "results": {
    "<scenario_id>": {
      "scenario_id": "...", "stage": "...", "age": 12,
      "query_text_preview": "...",
      "by_character": {
        "<char_id>": {
          "candidate_pool_size": 421,
          "topk": [
            {"rank": 1, "mem_id": "MEM_...", "score": 0.83, "mem_age": 7},
            ...
          ]
        }
      }
    }
  }
}
```

Only `mem_id` / `score` / `rank` / `mem_age` are stored — the runner looks
`mem_id` up in `benchmark/characters.json` to recover the full memory.

## Step 3 — run the MCQ benchmark

```bash
python experiments/scripts/run_naive_rag.py \
    --top-k 30 \
    --model gpt-5.4-mini \
    --workers 10
```

Defaults:

| Flag / env | Default | Meaning |
|---|---|---|
| `--top-k` | `30` | How many of the retrieved memories to feed to the LLM (must be ≤ what step 2 stored) |
| `--model` / `MCQ_MODEL` | `gpt-5.4-mini` | LLM that answers the MCQ |
| `--api-key` / `API_KEY` | _(from `.env`)_ | LLM API key |
| `--api-base` / `API_BASE` | `https://api.openai.com/v1` | LLM endpoint |
| `--timeout` / `TIMEOUT` | `120` | Per-request HTTP timeout (seconds) |
| `--temperature` | `0` | LLM sampling temperature |
| `--workers` | `10` | Concurrent LLM calls |
| `--limit` | `0` | Process only the first N questions (`0` = all) |
| `--resume` | _(none)_ | Path to a previous `predictions_top<K>.json`; questions already finished with `ok=true` are skipped |
| `--out-prefix` | _(timestamp)_ | Unused for the canonical filenames below, but kept for legacy callers |

Reads `cache/retrieval/scenario_topk.json`, builds a prompt that
mirrors `experiments/scripts/main.py` (Background / Past Experiences /
Current Situation / Trigger Event / Behavioural Decision Options), calls
the LLM, parses `decision_choice` (A/B/C/D), and writes:

- `experiments/results/naive_rag/<model>/predictions_top<K>.json`
- `experiments/results/naive_rag/<model>/summary_top<K>.json`

The runner is resumable; pass `--resume <previous predictions.json>` to
skip questions that already have `ok=true`. See
`python experiments/scripts/run_naive_rag.py --help` for the full flag
list.

## Sensitivity to K

Different `top-k` values trade off retrieval coverage against prompt
length. Step 2 always stores the maximum K you ever care about (the
default `30` is enough for most experiments); step 3's `--top-k` then
truncates to the desired prefix without re-running retrieval. To explore
beyond `30`, rerun step 2 with a larger `--top-k`.

## Files in this directory

| File | Role |
|---|---|
| `embed_memories.py` | Preprocessing step 1 (per-character memory embeddings) |
| `retrieve_scenarios.py` | Preprocessing step 2 (per-scenario top-K retrieval) |
| `__init__.py` | Re-exports the shared retriever from `utils.memory_retrieval` so the baseline has its own importable surface |

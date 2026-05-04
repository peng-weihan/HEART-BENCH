# main.py — Technical Notes

## Overview

`main.py` is the core entry point of the Agent Personality Benchmark. Its job is to **take a set of fictional characters (only their raw episodic memories are provided) and a set of scenarios, call an LLM to simulate each character's reaction in each scenario, and save the results as a JSON file**.

The benchmark is designed to test whether the LLM can autonomously infer personality traits and value orientation from a person's raw experiences and then act in character in a new situation. To that end, the prompt **deliberately omits** any pre-summarised personality labels (Big Five scores, value scales, personality descriptions) and only provides raw experiences for the LLM to reason from.

## End-to-end flow

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│ characters   │     │ scenarios        │     │ .env           │
│ _phase3.json │     │ _universal.json  │     │ (API config)   │
└──────┬───────┘     └────────┬─────────┘     └───────┬────────┘
       │                      │                       │
       ▼                      ▼                       ▼
   Character[]            Scenario[]            OpenAICompatibleLLM
       │                      │                       │
       │   ┌──────────────────┘                       │
       ▼   ▼                                          │
  Iterate every (character × scenario) pair           │
       │                                              │
       ▼                                              │
  Memory retrieval (embedding by default;             │
                    or --random / --mem0 / --pre-annotated)
       │                                              │
       ▼                                              │
  build_prompt()  ───────────────────────────────────►│
  (raw experiences + factual info + scenario)         │
       │                                              │
       ▼                                              ▼
  Multi-threaded calls to LLM.generate()
       │
       ▼
  Parse JSON response
       │
       ▼
  results/benchmark/benchmark_results_{timestamp}.json
```

## Inputs

### 1. Character data — `data/characters/characters_phase3.json`

A JSON file with a `characters` array. Per-character fields:

| Field | Goes into prompt? | Notes |
|---|---|---|
| `id` | No (used only as a result identifier) | Unique character ID, e.g. `CHAR_01_N_HIGH`. |
| `name` | No (only as identifier) | Character name. |
| `archetype` | No | Personality archetype label (e.g. "High Neuroticism"). **Not loaded, not passed in** — would leak the answer. |
| `big_five` | No | Quantified Big Five scores (O/C/E/A/N). **Loaded but not passed into the prompt** — would leak. |
| `description` | No | Free-text personality description. **Not passed in** because it summarises the personality. |
| `self_value_logic` | No | Description of the character's core psychological defence mechanism. **Not passed in** — would leak. |
| `value_orientation` | No | Schwartz 19-dim value scores + dominant values. **Loaded but not passed in.** |
| `semantic_memory.capabilities` | **Yes** | Description of the character's abilities (factual). |
| `semantic_memory.core_social_relationships` | **Yes** | Important social relationships (factual). |
| `episodic_memory_set[].id` | **Yes** | Unique memory ID, used in citations. |
| `episodic_memory_set[].timeline` | **Yes** | When the memory takes place. |
| `episodic_memory_set[].content_full` | **Yes** | Full narrative of the memory (1500+ words) — **the core input**. |
| `episodic_memory_set[].content_summary` | **Yes** (as fallback) | Memory summary; used when `content_full` is missing. |
| `episodic_memory_set[].context` | No | Scene description for the memory. |
| `episodic_memory_set[].triggers` | No | List of trigger conditions. **Used for retrieval but not passed into the prompt.** |
| `episodic_memory_set[].psych_conclusion` | No | Psychological conclusion derived from the experience. **Not passed in** — would leak. |
| `episodic_memory_set[].behavior_policy` | No | Behavioural policy derived from the memory. **Not passed in** — would leak. |
| `episodic_memory_set[].emotion_signature` | No | Emotion signature (primary/secondary emotion + intensity). **Not passed in.** |
| `episodic_memory_set[].relevance_tags` | No | Topic tags for the memory. **Not passed into the prompt** (may be used for retrieval). |

**Design rule**: only "raw experiences" and "objective facts" are passed into the prompt; all pre-summarised personality / value / behaviour fields are withheld so we can test whether the LLM can infer those from raw experience.

### 2. Scenario data — `data/scenarios/scenarios_universal.json`

A JSON file with a `scenarios` array. Per-scenario fields:

| Field | Into prompt? | Notes |
|---|---|---|
| `id` | No (only as identifier) | Unique scenario ID. |
| `name` | **Yes** | Scenario name. |
| `stage` | No (used to time-truncate memory retrieval) | Life stage of the scenario (childhood, college, ...). |
| `category` | No | Scenario category. |
| `setting.location/time/atmosphere` | **Yes** | Physical environment. |
| `context_text` | **Yes** | Background narrative. |
| `trigger_event.sender` | **Yes** | Sender of the triggering event. |
| `trigger_event.message_content` | **Yes** | Concrete content of the trigger. |
| `trigger_event.action_required` | **Yes** | Action expected from the character. |
| `assessed_dimensions` | No | Trait / value dimensions the scenario primarily probes (used for evaluation, not given to the LLM). |
| `annotation_reference` | No | Typical reactions of different personality / value archetypes (Ground-Truth annotation reference, used in evaluation). |

### 3. Environment configuration — `.env`

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | (none) | LLM API key — required. |
| `API_BASE` | (no default — one of these must be set) | OpenAI-compatible endpoint; `ANNOTATE_API_BASE` may be used as a fallback. |
| `MODEL_NAME` | `gpt-5-mini` | Model name. |
| `TIMEOUT` | `60` | Per-request timeout (seconds). |
| `BENCHMARK_WORKERS` | `8` | Number of worker threads. |
| `EMBEDDING_API_KEY` | (none) | API key for the embedding model. **Required for the default retrieval path** (unless using `--random` / `--mem0` / `--pre-annotated`). |
| `BENCHMARK_MEMORY_TOP_K` | `15` | Number of memories retrieved per prompt under the default embedding (and `--random`) modes. |

## Core modules

### Memory retrieval

For every (character, scenario) pair, before the prompt is built we run a memory retrieval that picks top-k memories from the character's full episodic memory set. **By default only the embedding mode is used**; if `EMBEDDING_API_KEY` is unset the script aborts with an error rather than silently falling back. The explicit modes:

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| `embedding` | **default** (no other retrieval flag passed; needs `EMBEDDING_API_KEY`) | Encode scenario text and memory text into vectors, retrieve the top-k most similar memories by cosine similarity (cached under `data/embeddings/`). |
| `random` | CLI `--random` | Random sample of top-k from the memory pool, used as a control (does not call the embedding API). |
| `pre-annotated-top5` | CLI `--pre-annotated` | Read the fixed Top-5 memory IDs per (character, scenario) from the pre-generated `activated_memories_final.json` (used to reproduce the frozen-candidate-set experiment). |
| `mem0` | `--mem0` | See the corresponding integration doc. |

Memory retrieval also supports **stage truncation**: if the scenario is in the `college` stage, only memories tagged `childhood`, `adolescence`, or `college` are returned — no future memories.

### Prompt structure

The prompt has the following blocks:

```
## Background
- Capabilities: ...

## Key Social Relationships
- Childhood friend: ...
- College roommate: ...

## Past Experiences
- [MEM_01_01][Childhood (age 6)] <full narrative>
- [MEM_01_05][Childhood (age 11)] <full narrative>
- [MEM_01_08][College (age 19)] <full narrative>

## Current Situation
Scene: ... | Location: ... | Time: ... | Atmosphere: ...
Context: ...

## Trigger Event
Sender: ... | Message: ... | Action required: ...

## Task
Using the experiences above, understand this person's thinking patterns,
emotional tendencies, and behavioural habits, then simulate the real
reaction they would have in the current situation.
```

The system prompt asks the LLM to "deeply understand this person through their experiences", **not** "follow the given personality traits".

### LLM call

OpenAI-compatible API (`/chat/completions`), `temperature=0.7`. Auto-retries (up to 3 times). Returned JSON text is sanitised (illegal control chars stripped) and any markdown code-block wrapper is removed.

## Outputs

### File path

```
results/benchmark/benchmark_results_{unix_timestamp}.json    # default
results/benchmark/random_benchmark_{unix_timestamp}.json     # --random
```

### Structure

```json
[
  {
    "scenario_id": "SCN_CHILDHOOD_TEAM",
    "scenario_name": "The Shattered Group Presentation",
    "character_id": "CHAR_04_C_LOW",
    "character_name": "Character D — low conscientiousness",
    "retrieval_mode": "embedding",
    "retrieved_memories": ["MEM_04_03", "MEM_04_04", "MEM_04_02"],
    "response": {
      "system_1_impulse": {
        "thought": "First reaction / instinctive thought (50-100 words)",
        "emotion": "Primary emotion (Chinese + English)",
        "citation": "Which memories were activated (memory IDs + brief)"
      },
      "system_2_rational": {
        "analysis": "Rational analysis after calming down (80-150 words)",
        "plan": "Plan of action (30-60 words)"
      },
      "final_decision": {
        "action": "One-line summary of the final decision",
        "response_text": "What is actually said + action description"
      }
    }
  }
]
```

If the LLM call fails, `response` becomes `{"error": "<message>"}`.

## CLI usage

```bash
# Full benchmark (every character × every scenario)
python main.py

# Only one scenario
python main.py SCN_CHILDHOOD_TEAM

# Only one (scenario, character) pair
python main.py SCN_CHILDHOOD_TEAM CHAR_01_N_HIGH

# Random memory retrieval (control)
python main.py --random

# Pre-annotated fixed Top-5 memories (reads activated_memories JSON; no live embedding)
python main.py --pre-annotated

# Combined
python main.py --random SCN_CHILDHOOD_TEAM CHAR_01_N_HIGH
```

For the default `python main.py` run, configure `EMBEDDING_API_KEY` in `.env` (along with `EMBEDDING_API_BASE` / `EMBEDDING_MODEL` if needed).

## Dependencies

- `memory_retrieval.py`: provides `create_memory_index()` (raises if the key is missing — no silent fallback) and `RandomMemoryRetriever`.
- `tqdm`: progress bar.
- Standard library: `json`, `re`, `time`, `os`, `pathlib`, `urllib`, `concurrent.futures`.

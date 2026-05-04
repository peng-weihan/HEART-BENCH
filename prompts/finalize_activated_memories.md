# finalize_activated_memories.py — Stage 2: Activated-Memory Refinement

**Source:** `scripts/finalize_activated_memories.py`
**Purpose:** Read the Stage-1 output (`activated_memories_binary.json`); for each character × scenario, refine the candidate activated memories down to a fixed count (default 50) via a second LLM pass. Output: `activated_memories_final.json`.

---

## 1. System Prompt (`LLM_FINALIZE_SYSTEM_PROMPT`)

Template string; before the call, formatted with `.format(target=target)` to inject the desired count.

```text
You are a professional research psychologist specializing in autobiographical memory, personality dynamics, and behavioral decision theory.

Your task: from the Stage-1 candidate activated memories, select the {target} memories that will **actually enter the character's consciousness and influence behavior** in the given situation.

## Psychological basis for refinement

Stage 1 filtered all "potentially activated" memories, but at any given moment only a small number actually enter working memory and shape behavior. You need to judge which memories will "surface to consciousness", using the following priority order:

### Priority 1: Memories that directly drive current behavior
- Behavioral-script match: the memory's behavior_policy directly answers "what to do right now"
- Conditioned-reflex activation: response patterns repeatedly reinforced in similar situations (e.g., "always apologizing first when challenged")
- Approach/avoidance motivation: pain or success in the memory directly pushes the character toward or away from a current option

### Priority 2: Memories that shape the interpretive frame
- Source of attribution patterns: determines whether the character reads the event as "threat" or "opportunity", "malice" or "misunderstanding"
- Core-belief anchor: the experience that formed the core belief currently being challenged or echoed
- Interpersonal templates: defines the character's default expectations toward specific people in the scene (authority, peers, intimate others)

### Priority 3: Memories that supply emotional undertone
- Emotion prototype: the experience in which the emotion now evoked by the scene was first deeply felt
- Unfinished business: emotional experiences that were not fully processed in the past and still seek resolution
- Body memory: memories activated through sensory cues, accompanied by strong somatic sensations

### De-redundancy principles
- If multiple memories carry the SAME psychological signal (e.g., three episodes all about "being denied by an authority"), keep the one with the GREATEST psychological formative power — usually the earliest (the one that formed the schema) or the most emotionally intense.
- Prefer memories from DIFFERENT life stages, to reflect the character's longitudinal psychological development.
- Deep linkage outweighs surface topical similarity: a childhood experience of being ostracized may explain the character's avoidance in the current scene better than a recent social failure.

## Output format

Strictly output a JSON object (no markdown code block).

Full mode (default):
{
  "activated_memories": [
    {
      "memory_id": "MEM_XX_XXXX",
      "type": "BS|AM|CB|ER|DL|NI",
      "reason": "20–30 chars"
    }
  ]
}

Compact mode (with --compact):
{
  "activated_memories": ["MEM_XX_XXXX", "MEM_XX_XXXX", ...]
}

Type codes: BS=behavioral script, AM=attribution mode, CB=core belief, ER=emotional resonance, DL=deep linkage, NI=narrative identity.
The array order is the ranking (item 1 = most influential).

## Cautions

- Output exactly {target} entries — no more, no fewer.
- Do not invent memory_id values not in the candidate list.
```

---

## 2. User Prompt (`build_finalize_prompt`)

Three blocks concatenated.

### Block A — Character info

```text
## Character info

- Character ID: {char.id}
- Name: {char.name}
- Archetype: {char.archetype}
- Big Five: O={openness:.2f} C={conscientiousness:.2f} E={extraversion:.2f} A={agreeableness:.2f} N={neuroticism:.2f}
```

### Block B — Current scenario

```text
## Current scenario

- Scenario ID: {scenario.id}
- Name: {scenario.name}
- Stage: {scenario.stage} (age {scenario.age})
- DIAMONDS dimension: {scenario.diamonds_dimension}
- Description: {scenario.description_for_agent}

### Background
{scenario.context_text}

### Trigger event
Sender: {trigger.sender}
{trigger.message_content}
Action required: {trigger.action_required}
```

### Block C — Candidate activated memories

```text
## Candidate activated memories (total {N}, refine to {target})

### [1] {memory_id} ({memory.timeline})
- {memory.content_full or content_summary}
- Stage-1 reason: {stage1_reason}    ← only if Stage 1 provided a reason

### [2] {memory_id} ({memory.timeline})
- {memory.content_full or content_summary}
- Stage-1 reason: {stage1_reason}

... (N entries)
```

### Closing instruction (one of two)

**Default (full mode):**
```text
From the {N} candidate memories above, select the most critical {target}, and output the JSON.
```

**With `--compact`:**
```text
From the {N} candidate memories above, select the most critical {target}, and output in compact mode: {"activated_memories": ["MEM_XX_XXXX", ...]}, a list of memory_id strings only, ordered by influence.
```

---

## 3. Fast skip (no LLM call)

When the candidate count ≤ `target`, all candidates are kept directly without an LLM call; the output is marked `skipped_finalize: true`.

---

## 4. Call parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | `0.2` | inherited from `annotate_activated_memories.call_llm` default |
| `max_tokens` | from `ANNOTATE_MAX_TOKENS`, default unbounded | env |
| `model` | `ANNOTATE_REFINE_MODEL` (falls back to `ANNOTATE_SCREEN_MODEL`) | env |
| `use_thinking` | `False` (thinking explicitly disabled) | hard-coded |
| `target` | `--target` CLI flag (default `TARGET_ACTIVATED = 50`) | CLI / hard-coded |
| `compact` | `--compact` CLI flag | CLI |
| `MAX_RETRIES` | `2` (excludes 429 rate-limit retries) | inherited from Stage 1 |

---

## 5. JSON-parse failure handling (no prompt-level retry)

Stage 2 does **not** retry the model. Instead:

1. Try `json.loads` directly on the text.
2. If that fails, scan the text for all top-level `{...}` spans and attempt to parse each.
3. As a last fallback, take `text[first '{' : last '}']` and try to parse.
4. If everything fails, the **full raw response** is written to `data/annotations/<refine_slug>/finalize_parse_failures/<ts>_<uid>_<char>__<scn>.txt`, and a `RuntimeError` referencing that file is raised.

If candidates > target but the model returns an empty `activated_memories` list, the same dump-and-raise path applies.

---

## 6. Call-chain structure

```
finalize_one_scenario
  ├─ if n_candidates <= target:
  │     skip LLM, copy all candidates → activated_memories
  └─ else:
        annotate_activated_memories.call_llm(
            system_prompt = LLM_FINALIZE_SYSTEM_PROMPT.format(target=target),
            user_prompt   = build_finalize_prompt(...),
            model         = LLM_REFINE_MODEL,
            use_thinking  = False,
            usage_log_dest = "finalize",
        )
        → _parse_finalize_llm_response → dedupe & truncate to target
```

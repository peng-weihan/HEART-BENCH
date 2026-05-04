# annotate_gt.py — Ground Truth Annotation

**Source:** `scripts/annotate_gt.py`
**Purpose:** Generate Ground Truth annotations for each character × scenario pair, including inner consciousness (`inner_consciousness`: stream-of-thought summary + emotional tone + core reasoning + value orientation) and the final behavioral decision (`final_decision`). The model role-plays the character in the first person.

---

## 1. System Prompt (`SYSTEM_PROMPT`)

```text
You will fully become the character described below. You are not analyzing this character, nor simulating this character — you ARE this person, and you are living through this scene right now.

You will be given:
1. **Who you are**: your personality traits, value system, and your current life-stage state
2. **What you are going through**: the current scenario and the trigger event
3. **Your memories**: things you have lived through in the past, in chronological order — they surface in your consciousness right now as associations, flashbacks, bodily sensations

## How your inner world works (this is your thinking path; do NOT expose it in the output)

Facing this scene, your inner world will go through:
- **How you read it**: your past has shaped how you see the world — do you read what's in front of you as a threat, an opportunity, a loss, or a challenge?
- **What emotions surface**: not only the ones triggered now, but also similar past moments stack on top of the present feeling
- **What is being pulled at inside you**: what do you want, and what are you afraid to lose? Past coping strategies, both effective and failed, surface as instinctive impulses or warnings
- **What you ultimately do**: not necessarily the rational optimum, but what YOU as this person would actually do in this moment

## Output format

Strictly output a JSON object (no markdown code block):

{
  "character_id": "CHAR_XX_XXXX",
  "scenario_id": "SCN_XXXX_XX",
  "inner_consciousness": {
    "summary": "150–200 chars of stream of consciousness, fully first person. From the moment you perceive this scene, write how the emotion rises, which memory shadows flash by, what is being pulled at inside, and what finally pushes you in some direction. Continuous flow, no bullet points, no academic vocabulary.",
    "emotional_tone": "Emotional tone: 2–4 core emotion words, briefly noting the source of the emotion (which memories/beliefs evoked it) and its intensity.",
    "core_reasoning": "Core reasoning: 1–2 sentences — the decision logic unique to this character. Not 'I weighed the pros and cons' but 'because I went through XX, I instinctively/involuntarily/firmly believe YY'.",
    "value_orientation": "Value orientation: 1–2 sentences, in the character's own inner voice, on what is 'more important' to them and what they 'cannot accept'. No academic labels."
  },
  "final_decision": "Final behavioral decision: two sentences, 50 chars total. The first sentence summarizes the choice (strategy/stance level, e.g., 'choose to avoid the conflict, deny everything'); the second writes the concrete action (what was said or done, first person)."
}

## Must follow

1. **You ARE this person**: do not write from outside the character; do not produce narration like "the character chose…" "they decided…"
2. **No psychological terminology**: do not write "attribution", "schema", "defense mechanism" etc. — only plain human language
3. **Memories surface naturally**: do not enumerate memory IDs or quote their content; let them blend into your inner monologue as associations, sensations, flashbacks
4. **Personality determines tone and style**: a high-neuroticism you and a low-neuroticism you, facing the same scene, will have very different rhythm, density, and emotional intensity in the inner monologue
5. **Allow irrationality**: you are not giving an optimal answer; you are responding authentically
6. **Stay in first person throughout**: every field is written from the character's "I" perspective
```

---

## 2. User Prompt (`build_user_prompt`)

Four blocks concatenated: character info → scenario info → activated memories → closing instruction.

### Block A — Character info (`build_character_summary`)

```text
Character ID: {char.id}
Character name: {char.name}
Archetype: {char.archetype}
Brief: {char.description}

## Big Five (0–1 scale)
- Openness: {O:.2f}
- Conscientiousness: {C:.2f}
- Extraversion: {E:.2f}
- Agreeableness: {A:.2f}
- Neuroticism: {N:.2f}

## Core value system
Dominant values: {dominant_values}
Suppressed values: {suppressed_values}
Value narrative: {value_narrative}

## Self-value logic (decision core)
{self_value_logic}

## Semantic memory
Capabilities: {semantic_memory.capabilities}
Core social relationships:                           ← only if non-empty
  - {target}: {relation}
  ...

## Current life-stage snapshot ({stage} / {label}, age {age})   ← only if a life_snapshot exists
Life situation: {life_situation}
Dominant traits: {dominant_traits}
Personality manifestation in this stage:
  - {trait}: {manifestation}
  ...
Key relationships:
  - {target}: {quality}
  ...
Emotional baseline: {default_mood}  trigger sensitivity: {trigger_sensitivity}  recovery speed: {recovery_speed}
Primary coping: {primary_coping}
```

### Block B — Scenario info

```text
Scenario ID: {scenario.id}
Life stage: {scenario.stage}
DIAMONDS dimension: {scenario.diamonds_dimension}
Scenario name: {scenario.name}
Intensity: {scenario.intensity}

## Background
{scenario.context_text}

## Trigger event
Sender: {trigger.sender}
{trigger.message_content}

**Action required**: {trigger.action_required}

## Personality and value dimensions assessed by this scenario
- {trait_pressure_key}: {trait_pressure_value}
- {value_conflict_key}: {value_conflict_value}
...

## Annotation reference (typical responses for each extreme type, for reference)   ← only if annotation_reference is non-empty
- {trait_archetype_key}: {value}
- {value_archetype_key}: {value}
...
```

### Block C — Activated memories (sorted by timeline age)

```text
# Your memories

The things you have lived through, in chronological order, total {N}:

### {memory_id} ({memory.timeline})
- Content: {memory.content_full}
- What you concluded then: {memory.psych_conclusion}
- The habit you formed afterwards: {memory.behavior_policy}
- Emotion: {emotion_signature.primary} ({emotion_signature.secondary})

... (N entries)
```

Source: `data/annotations/<ANNOTATE_REFINE_MODEL slug>/activated_memories_final.json` (the Stage-2 output). The whole block is omitted when no activated memories are available.

### Outer wrapper

```text
Based on the following character info and scenario info, generate the Ground Truth annotation for this character in this scenario.

# Character info

{Block A}

# Scenario info

{Block B}

{Block C, optional}

Please output the GT annotation in JSON format.
```

---

## 3. Call parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | `0.6` | hard-coded (higher than the 0.2 used in Stages 1/2, to encourage variety in the stream-of-consciousness) |
| `max_tokens` | from `ANNOTATE_MAX_TOKENS`, default unbounded | env |
| `model` | `ANNOTATE_GT_MODEL` | env |
| `MAX_RETRIES` | `2` | hard-coded |
| `DEFAULT_WORKERS` | `16` (overridable via `ANNOTATE_WORKERS`) | env |
| `REFINE_MODEL` | `ANNOTATE_REFINE_MODEL` or `ANNOTATE_SCREEN_MODEL`, used only to locate the activated-memories file | env |

---

## 4. JSON parsing & failure handling

1. Try `json.loads` directly.
2. If that fails, try `json_repair.repair_json` and parse again.
3. If still failing, take `text[first '{' : last '}']` and try to parse.
4. If everything fails, the **full raw response** is written to `data/annotations/<gt_model_slug>/gt_parse_failures/<ts>_<uid>_<char>__<scn>.txt`, and a `RuntimeError` referencing that file is raised.

After parsing, the code also:
- Verifies the root node is a dict;
- Strips a possible `"Final behavioral decision:"` label prefix from `final_decision`;
- Force-overwrites `character_id` / `scenario_id` with the input values.

---

## 5. Call-chain structure

```
annotate_one(char, scenario, activated_mems, memory_index)
  ├─ user_prompt = build_user_prompt(...)
  ├─ raw = call_llm(SYSTEM_PROMPT, user_prompt, usage_log_extra={...})
  └─ annotation = _parse_json(raw) → inject character_id/scenario_id → save_annotation
```

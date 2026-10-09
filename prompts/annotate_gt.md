# annotate_gt.py — Candidate Response Annotation

**Source:** `constructions/annotation/annotate_gt.py`
**Purpose:** Generate candidate responses for expert comparison and adjudication. The model role-plays the character in the first person and returns `inner_consciousness` and `final_decision`. The model output is a candidate, not the adjudicated answer key.

This is an abridged construction-prompt exhibit. Unpopulated character and scenario metadata blocks are omitted. Length and sentence-count instructions guide Chinese-source drafting; English translations preserve the complete meaning. The retained `inner_consciousness.value_orientation` key names an auxiliary natural-language rationale, not a scored personality profile. See the [construction guidance](../README.md#construction-guidance).

---

## 1. System Prompt Excerpt (`SYSTEM_PROMPT`)

```text
You will fully become the character described below. You are not analyzing this character, nor simulating this character — you ARE this person, and you are living through this scene right now.

You will be given:
1. **Who you are**: your supplied character specification [unpopulated metadata omitted]
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
  "final_decision": "Final behavioral decision: aim for two concise sentences, about 50 Chinese characters as drafting guidance. The first sentence summarizes the choice (strategy/stance level, e.g., 'choose to avoid the conflict, deny everything'); the second writes the concrete action (what was said or done, first person)."
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

## Self-value logic (decision core)
{self_value_logic}
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

## Translation and text repair

Translate each supplied field faithfully. Preserve amounts, currencies, units, dates, ages, relationships, chronology, negations, conditions, uncertainty, and the direction of the decision. For example, 198 yuan remains 198 yuan; do not convert the currency or change the amount. Keep field names, IDs, option labels, order, and answer keys unchanged.

Translate each MCQ option's **actual text**. Do not substitute another record's `final_decision` solely because its `source_character` and `scenario_id` match: edited or generated distractors can differ from that reference. Review the four translated options together to preserve their behavioral distinctions. Length and sentence-count hints guide drafting and do not justify truncating an English translation. Flag incomplete or ambiguous source text for review instead of inventing missing facts or changing the answer.

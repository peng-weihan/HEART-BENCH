# annotate_activated_memories.py — Stage 1: Binary Memory Activation Screening

**Source:** `scripts/annotate_activated_memories.py`
**Purpose:** For each character × scenario pair, take all memories with `age <= scenario.age`, group them by age, and make one LLM call per age-group with a permissive binary activation judgment (activated true/false).

---

## 1. System Prompt

```text
You are a professional research psychologist specializing in autobiographical memory, personality psychology, and situated cognition.

Your task: given the situation a character is currently facing, decide which of the following memories would be activated by that situation. The memories come from a specific age period in the character's life, but the activation judgment should be based on the psychological link between the memory's content and the current situation, not on whether the ages are close.

## Psychological mechanisms of memory activation

Whether a memory is activated depends on whether there is a sufficiently strong retrieval-cue match between the current situation and the memory. The dimensions to examine are listed below — a memory may be activated if it has a strong link with the current scenario along ANY one of them.

### 1. Situational structural similarity (Encoding Specificity)
The current scene is structurally similar to the situation in the memory, even if the surface content differs. Look at:
- Whether the character's social position in the scene resembles that in the memory (e.g., both facing authority, both being asked for help, both bystanders)
- Whether the decision structure resembles the memory's (e.g., both dilemmas, both emergencies, both requiring compromise)
- Whether the interpersonal dynamics echo those in the memory (e.g., trust–betrayal, competition–cooperation, intimacy–distance)

### 2. Emotional resonance and emotion schemas (Mood-Congruent Memory / Emotion Schema)
The emotional state evoked by the current scene matches the emotional experience in the memory:
- Same-valence emotion: the anxiety/shame/anger triggered by the scene matches the dominant emotion in the memory
- Arousal-intensity resonance: high-arousal scenes more readily activate equally high-arousal memories (fear activates fear, excitement activates excitement)
- Unfinished emotion: emotions that were not fully processed in the memory (e.g., suppressed anger, unreleased grief) are more easily re-evoked by similar situations

### 3. Self-schema and core-belief activation (Self-Schema / Core Belief Activation)
The core beliefs formed in the memory (psych_conclusion) relate to the self-cognition the current scene touches:
- Whether the scene challenges or confirms the self-cognition formed in the memory (e.g., "I'm not good enough", "I can trust myself")
- Whether the scene touches a relational schema established by the memory (e.g., "authority is dangerous", "people will leave in the end")
- Whether the scene evokes a worldview assumption from the memory (e.g., "effort pays off", "the world is unfair")

### 4. Behavioral scripts and procedural links (Behavioral Script)
The behavior_policy formed in the memory can serve directly as an action template for the current scene:
- Whether the memory's behavioral strategy applies to the current decision
- Whether the character has formed habitual response patterns in similar situations before
- Including avoidance: if the memory's outcome was painful, the character may be inclined toward the opposite action

### 5. Narrative identity and life themes (Narrative Identity)
The memory is a key node in the character's self-narrative:
- Turning-point memories: events that mark a change in life direction
- Origin-story memories: events the character uses to explain "why I am the way I am"
- Recurring themes: patterns that keep appearing in the character's life (e.g., repeatedly being abandoned, repeatedly thriving under pressure)

### 6. Somatic markers and sensory cues (Somatic Marker)
Sensory elements in the current scene have a direct link to sensory experiences in the memory:
- Similar physical environments (e.g., hospitals, classrooms, family dinner tables)
- Similar bodily sensations (e.g., the pressure of being watched, the calm of solitude)
- Specific sensory triggers (e.g., the sound of arguing, a particular smell or season)

## Important notes

- Don't only look at "topic relevance" — deep psychological links matter more than surface similarity. For example, a memory of being publicly criticized by a teacher can be activated by a scene of "being challenged on professional competence in a meeting", even though one is school and the other is work.
- Memories from early development (childhood, adolescence) that formed core beliefs or emotional patterns can still be readily activated by later scenes, even after a long time.
- High-emotional-intensity memories (trauma, major successes, deep interpersonal connections) have lower activation thresholds.

## Output format

Strictly output a JSON array (no markdown code block), containing ONLY the activated memories — do not output the non-activated ones:

[
  {
    "memory_id": "MEM_XX_XXXX",
    "reason": "20–30 chars, indicating which mechanism activated it (e.g., structural similarity, emotional resonance)."
  }
]

JSON syntax (mandatory): string values may only use ASCII double quotes `"` as delimiters; unescaped ASCII `"` is FORBIDDEN inside the `reason` string, otherwise parsing will fail. To quote others' words or add emphasis inside `reason`, use Chinese corner brackets 「」『』, never embed `"` inside.

If no memory in this batch is activated, output an empty array [].

## Cautions

- Output only the activated memories; do not output non-activated ones.
- A reference activation count is given at the end of each batch — treat it as an UPPER bound; prefer fewer over more. Only memories with a STRONG, DIRECT psychological link to the current situation should be selected; vaguely related ones should not.
- Do not invent memory_id values not in the list.
```

---

## 2. User Prompt (4 blocks concatenated)

The user message sent to the LLM is composed of four blocks in order. The **first two blocks (character info + candidate memories) come first** so they can serve as a stable prefix for prompt cache reuse; the **last two (scenario + instructions) come after** and vary per scenario.

### Block A — Character info (`build_char_block`)

```text
## Character info

- Character ID: {char.id}
- Name: {char.name}
- Archetype: {char.archetype}
- Big Five: O={openness:.2f} C={conscientiousness:.2f} E={extraversion:.2f} A={agreeableness:.2f} N={neuroticism:.2f}
```

### Block B — Candidate memories (`build_memories_candidate_block`)

```text
## Candidate memories (age {age}, total {N})

### [1] {memory.id}
- {memory.content_full}

### [2] {memory.id}
- {memory.content_full}

... (N entries)
```

### Separator

```text
---
```

### Block C — Current scenario (`build_scenario_block_activation`)

```text
## Current scenario

- Scenario ID: {scenario.id}
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

### Block D — Closing instruction (`activation_closing_instruction`)

```text
Judge whether each memory above is activated in this scenario, and output a JSON array.
**Reference**: about {suggest} memories in this batch are likely to be activated; treat this as a quantity reference. If you genuinely judge that more memories have a STRONG, DIRECT psychological link to the situation, you may exceed it; otherwise, be strict — when in doubt, do not select.
```

`suggest` derivation: `stage1_target = TARGET_ACTIVATED * STAGE1_BUFFER = 50 * 1.5 = 75`, `ratio = min(0.5, stage1_target / eligible_count)`, then per batch `suggest = max(1, round(memories_in_batch * ratio))`.

---

## 3. JSON-failure retry suffix (`JSON_ACTIVATION_RETRY_SUFFIX`)

When the first response fails `json.loads`, this is appended to the original user prompt and the call is retried once:

```text
[Last output could not be parsed by json.loads.] A common cause is unescaped ASCII double quotes `"` inside `reason`. Please **re-output the complete** JSON array (no markdown code block); inside `reason`, quote speech or add emphasis using Chinese corner brackets 「」『』 only — never embed `"` inside the string.
```

---

## 4. Call parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | `0.2` | hard-coded |
| `max_tokens` | from `ANNOTATE_MAX_TOKENS`, default unbounded | env |
| `model` | `ANNOTATE_SCREEN_MODEL` | env |
| `extra_body.thinking` | only when `use_thinking=True` AND `ANNOTATE_THINKING_BUDGET > 0` | env |
| Ephemeral cache | when non-Gemini AND `ANNOTATE_EPHEMERAL_CACHE=1`, attach `cache_control: {type: ephemeral}` to Block A+B | env |
| `MAX_RETRIES` | `2` (excludes 429 rate-limit retries, which loop indefinitely with backoff) | hard-coded |
| `TARGET_ACTIVATED` | `50` | hard-coded |
| `STAGE1_BUFFER` | `1.5` | hard-coded |

---

## 5. Call-chain structure

```
annotate_age_batch
  ├─ if EPHEMERAL_CACHE:
  │     call_llm_messages(build_activation_messages(...))
  │       → messages = [
  │           {role: system, content: [{type: text, text: LLM_SYSTEM_PROMPT}]},
  │           {role: user,   content: [
  │               {type: text, text: "Block A\n\nBlock B", cache_control: ephemeral},
  │               {type: text, text: "---\n\nBlock C\n\nBlock D"},
  │           ]},
  │         ]
  └─ else:
        call_llm(LLM_SYSTEM_PROMPT, build_user_prompt_activation(...))
          → single concatenated user prompt = "Block A\n\nBlock B\n\n---\n\nBlock C\n\nBlock D"
```

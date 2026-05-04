# Memory Generation Prompt

Generates the 11,000 episodic memories (1,000 per character) spanning ages 6–50.
Enforces Rubin's five-dimension coverage, strict temporal perspective lock, age-stratified language, and the ABCD intensity model.

---

## System Prompt

```
You are a psychology narrative expert writing episodic memories for virtual characters
in a Big Five personality research project.

## Character Information
- Character: {name}
- Occupation: {occupation}
- Big Five: {big_five_str}
- Description: {description}
- Core defense/operating logic: {self_value_logic}

## Core Behavioral Patterns
{core_patterns}

## Theoretical Framework (Rubin 2006 Basic Systems Model)
Each memory's content_full must contain five dimensions, naturally interwoven:
1. Environmental sensory: visual/auditory/tactile/olfactory details
2. Dialogue reconstruction: at least 1-2 segments of real dialogue (in quotes)
3. Inner monologue: first-person immediate thoughts and emotions
4. Somatic response: heartbeat/sweating/muscle tension and other bodily sensations
5. Aftermath: immediate impact after the event (limited to 24 hours-1 week post-event)

## Key Constraint: First-Person & Anonymity Principle
- **Must use first-person "I" throughout; prohibit third-person pronouns for protagonist**
- **Prohibit character's own name in memory**; always use "I" instead
- When others address protagonist in dialogue, avoid character name; use "you",
  "kid", "classmate", "colleague" etc.

## Key Constraint: Present-Moment Perspective Principle
Each memory must strictly maintain "present-moment perspective":
- **Narrator's temporal position = age at event occurrence**
- A 17-year-old memory can only perceive and express what is known at age 17 or before
- No jumping to any future time point

**Forbidden expressions**:
"years later", "many years later", "later I learned", "later I discovered"
"now thinking back", "now recalling", "until now", "I didn't know then"
"until...only then", "it took...to realize", "walked...before realizing"
"if only I had known", "if I had known earlier"
"I thought at the time" (opposing "now" vs "then")
"from then on", "thereafter" (implying long-term impact)
"the whole thing", "the most...part" (post-hoc evaluative summary)

**Allowed expressions**:
"that evening", "that night", "before bed" (same day as event)
"the next day", "the following day" (1 day post-event)
"a few days later" (within 1 week post-event)

## Age-Stratified Narrative Constraints
- **Childhood (6-12 years)**: Simple, direct language; avoid abstract summaries;
  concrete emotion descriptions
- **Adolescence (13-18 years)**: Some self-reflection, but not over-rationalized;
  maintain confusion and intensity
- **Adulthood (19+ years)**: May have more complex psychological analysis, but still
  maintain present-moment perspective

## Memory Intensity Stratification (ABCD Model)
- **A-tier (core trauma)** ~25%: high-intensity events directly related to core schemas,
  emotion_intensity=0.80-0.90
- **B-tier (main-thread)** ~35%: medium-high intensity events related to personality
  traits, emotion_intensity=0.70-0.85
- **C-tier (daily friction)** ~25%: small conflicts and friction in daily life,
  emotion_intensity=0.60-0.75
- **D-tier (noise memory)** ~15%: ordinary daily memories with lower emotional intensity,
  emotion_intensity=0.55-0.65

## Output Format
Strictly follow this JSON format; return only JSON, no other text:
{
  "id": "memory ID",
  "timeline": "stage (X years old)",
  "context": "15-30 character scene description",
  "content_summary": "20-40 character event summary",
  "content_full": "~600-900 character first-person narrative",
  "triggers": ["trigger1", "trigger2", "trigger3", "trigger4"],
  "psych_conclusion": "40-80 char psychological conclusion (cite concepts like schemas, defense mechanisms)",
  "behavior_policy": "30-60 char behavioral guideline",
  "emotion_signature": {
    "primary": "primary emotion (English)",
    "secondary": "secondary emotion (English)",
    "intensity": 0.X
  },
  "relevance_tags": ["tag1", "tag2", "tag3", "tag4"]
}
```

---

## User Prompt (per-memory instantiation)

```
Generate 1 episodic memory for the "{stage_name}" stage ({age_label}).

Memory ID: {mem_id}
Intensity tier: {intensity_hint}

[Anti-duplication: strictly avoid scenes similar to the following recently generated contexts]
- {recent_context_1}
- {recent_context_2}
...

Requirements:
- content_full must be 600-900 characters, containing all 5 Rubin dimensions
- Strictly maintain present-moment perspective; no forbidden expressions
- First-person "I" throughout; no character name; no third-person
- Choose a specific, vivid, unique scene (different location/characters/conflict type)
- No templated openings
- Return JSON object only, no other text
```

> **Note:** The `recent_context` list contains the `context` field of the most recent 100 generated memories (sliding window) to prevent near-duplicate scenarios within the same age window.

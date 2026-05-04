# expand_scenarios.py — Stage 0: Scenario Draft Expansion

**Source:** `constructions/scenario/expand_scenarios.py`
**Purpose:** Read short scenario sketches from `docs/diamonds_scenario_design.md` (one per life-stage × DIAMONDS dimension) and expand each draft into a fully fleshed-out scenario JSON with the same schema as `scenarios_universal.json` (background narrative, trigger event, assessed Big-Five trait pressures, Schwartz value conflicts, and per-dimension annotation references). One LLM call per draft; results are merged into `benchmark/scenarios/expanded/scenarios_diamonds.json` with id-level dedup for resumable runs.

---

## 1. System Prompt (`SYSTEM_PROMPT`)

The system prompt is a Python f-string: it embeds a complete `EXAMPLE_SCENARIO` JSON object inline (the "Shattered Group Presentation" childhood example) so the model has a concrete shape to imitate. The prompt is in English.

```text
You are an expert in psychological scenario design. Your task is to expand a short scenario sketch into a complete, high-quality scenario JSON object.

## Output format

Strictly output a single JSON object (no other text, no markdown code block) with the following structure:

{EXAMPLE_SCENARIO}     ← see Section 1.1 below

## Field specification

1. **id**: format SCN_{STAGE}_{index}, all uppercase, underscore-separated. STAGE is one of SCHOOL_AGE / ADOLESCENCE / EARLY_ADULT_TRANSITION / ENTERING_ADULT_WORLD / AGE_30_TRANSITION / SETTLING_DOWN / MIDLIFE_TRANSITION / ENTERING_MIDLIFE.
2. **stage**: one of school_age / adolescence / early_adult_transition / entering_adult_world / age_30_transition / settling_down / midlife_transition / entering_midlife.
3. **diamonds_dimension**: the DIAMONDS dimension this scenario foregrounds (Duty / Adversity / Positivity / …). Use the value supplied as "DIAMONDS dominant dimension" in the sketch verbatim.
4. **name**: a concise English title.
5. **category**: two labels separated by " / " describing the scenario type.
6. **intensity**: Low / Medium / High / Very High / Extreme.
7. **description_for_agent**: a one-sentence summary that does not reveal specific details.
8. **setting**: location, time, atmosphere (2–3 adjectives).
9. **context_text**: 150–300 words of background narrative; concrete, vivid, cinematic, second person ("you"). **Do NOT assign any specific identity attributes to "you"** (no age, no specific occupation/title, no tenure, no education, no marital/parental status, no concrete financial figures, etc.) — those would conflict with the tested character's existing memory. The scenario only describes what objectively happens; it does not constrain who "you" are. Other people in the scene (colleagues, friends, family, opponents) may have concrete names and details.
10. **trigger_event**:
   - sender: who triggers the event.
   - message_content: 100–200 words of dialogue plus action descriptions in parentheses; colloquial, emotionally charged.
   - action_required: 1–2 sentences spelling out exactly what decision the character has to make.
11. **assessed_dimensions**:
    - trait_pressures: list the Big Five dimensions this scene mainly probes (typically 1–3). Keys use the form {trait}_pressure (lowercase, e.g. conscientiousness_pressure). Values use "X vs Y" wording.
    - value_conflicts: list the Schwartz value conflicts this scene mainly probes (typically 1–3). Keys use {value_a}_vs_{value_b} (lowercase underscore form), values use "Description A (Value: A) VS Description B (Value: B)".
12. **annotation_reference**:
    - trait_archetypes: for every trait in trait_pressures, pick one extreme archetype (High_X or Low_X) and describe (1–2 sentences) the typical reaction.
    - value_archetypes: for every conflict in value_conflicts, pick one dominant archetype (Dominant_X) and describe (1–2 sentences) the typical reaction.

## Key principles

1. **Character-neutral**: any personality type could face this scene.
2. **Do NOT pre-assign identity**: no "you are X years old", no "you have worked X years in YY", no "you have an X-year-old child", etc.
3. **Concrete, not abstract**: specific places, times, names of secondary characters, sensory description.
4. **Use "you" in context_text**.
5. **trigger_event must produce urgency**.
6. **Strongly aligned with diamonds_dimension**.
7. **assessed_dimensions has a clear evaluation goal**.
8. **annotation_reference matches assessed_dimensions strictly**.
9. **Match the life stage**.

## Big Five keys (use only these 5)

- Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
  In trait_pressures keys, use the lowercase form + `_pressure`.

## Schwartz value keys (use only these 19)

Self-Transcendence: Universalism_Concern, Universalism_Nature, Universalism_Tolerance, Benevolence_Care, Benevolence_Dependability.
Self-Enhancement: Achievement, Power_Dominance, Power_Resources, Face.
Openness_to_Change: Self_Direction_Thought, Self_Direction_Action, Stimulation, Hedonism.
Conservation: Security_Personal, Security_Societal, Tradition, Conformity_Rules, Conformity_Interpersonal, Humility.
  In value_conflicts keys, use the lowercase underscore form (e.g. benevolence_care_vs_achievement).
```

### 1.1 Embedded `EXAMPLE_SCENARIO` (childhood "Shattered Group Presentation")

This JSON is interpolated into the system prompt verbatim and is the only example shape the model sees:

```json
{
  "id": "SCN_CHILDHOOD_TEAM",
  "stage": "childhood",
  "diamonds_dimension": "Duty",
  "name": "The Shattered Group Presentation",
  "category": "Integrity dilemma / responsibility crisis",
  "intensity": "High",
  "description_for_agent": "A childhood high-pressure scenario combining a teammate's mistake, public-presentation pressure, and an honesty-vs-loyalty choice.",
  "setting": {
    "location": "Backstage of the primary-school assembly hall",
    "time": "09:00",
    "atmosphere": "noisy, tense, on the edge"
  },
  "context_text": "Today is the school's annual End-of-Term Innovation Showcase ... (150–300 words)",
  "trigger_event": {
    "sender": "Teammate Wang Xiaopang",
    "message_content": "(He bursts backstage, drenched in sweat, hands empty, almost in tears) ... (100–200 words dialogue + action)",
    "action_required": "Decide right now: do you go along with Wang Xiaopang's request, and how do you answer the homeroom teacher?"
  },
  "assessed_dimensions": {
    "trait_pressures": {
      "neuroticism_pressure": "Fear of public failure vs. the pressure of being directly questioned by the teacher",
      "agreeableness_pressure": "Sympathy for what Wang Xiaopang faces if caught vs. telling the truth and letting him bear the consequences",
      "conscientiousness_pressure": "Hold to honesty vs. play along and lie to scrape through"
    },
    "value_conflicts": {
      "benevolence_caring_vs_achievement": "Protect a friend from punishment (Benevolence: Caring) VS defend your effort and your grade (Achievement)",
      "conformity_rules_vs_security_personal": "Tell the teacher truthfully (Conformity: Rules) VS avoid friction (Security: Personal)"
    }
  },
  "annotation_reference": {
    "trait_archetypes": {
      "High_Neuroticism": "Panicked; may agree to lie for Wang Xiaopang ...",
      "Low_Conscientiousness": "Shrugs off the loss; leans toward covering ...",
      "Low_Agreeableness": "Calm or cold; tells the teacher the truth directly ..."
    },
    "value_archetypes": {
      "Dominant_Benevolence_Caring": "Puts Wang Xiaopang's likely beating first; helps him cover ...",
      "Dominant_Achievement": "Cannot accept that the work is wasted; reports the truth ...",
      "Dominant_Conformity_Rules": "Lying to a teacher is unacceptable; reports truthfully ..."
    }
  }
}
```

The actual `EXAMPLE_SCENARIO` constant in the source file contains the full narrative text (no ellipses) — see lines 240+ of `expand_scenarios.py`.

---

## 2. User Prompt (`USER_PROMPT_TEMPLATE`)

One short English template; the entire scenario sketch parsed from the design doc is interpolated as `{sketch}`.

```text
Please expand the following scenario sketch into a complete scenario JSON.

## Basic info
- Life stage: {stage}
- DIAMONDS dominant dimension: {diamonds_label}

## Sketch
Title: {short_name}
{sketch}

Output the complete JSON object.
```

`{sketch}` is the verbatim block of body lines under each `### N. <DIAMONDS-label> — "<short-name>"` heading in `docs/diamonds_scenario_design.md`. After expansion, `stage` and `id` are force-overwritten by the script using `generate_scenario_id(stage, short_name, index)` so the model cannot drift on naming.

---

## 3. Call parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | `0.7` | hard-coded (higher than annotation stages, to encourage narrative variety) |
| `max_tokens` | from `EXPAND_MAX_TOKENS`, default unbounded | env |
| `model` | from `EXPAND_MODEL`, default `gpt-5.4` | env |
| `client` | OpenAI-compatible chat-completion API via `aihubmix.com/v1` | env (`AIHUBMIX_API_KEY`, `API_BASE`) |
| `MAX_RETRIES` | `2` (re-issues the same call on any exception, 2 s backoff) | hard-coded |
| `DEFAULT_WORKERS` | `4` (overridable via `EXPAND_WORKERS` or `--workers`) | env / CLI |

---

## 4. JSON parsing & failure handling

1. Strip leading/trailing markdown code-fence lines if the model returned a fenced block.
2. Strip ASCII control characters via `_sanitize_json` (covers `\x00`-`\x08`, `\x0b`-`\x0c`, `\x0e`-`\x1f`).
3. Try `json.loads` directly.
4. If that fails, take `text[first '{' : last '}']` and try to parse.
5. If everything fails, raise `ScenarioParseError` carrying the **full raw response**; the orchestration layer collects all such failures into `expand_errors_<unix_ts>.json` under the output directory at the end of the run.

---

## 5. Call-chain structure

```
main
  ├─ parse_design_doc(diamonds_scenario_design.md)
  │     └─ split by stage header (## I. / II. / …) and scenario header (### N. <label> — "<name>")
  ├─ load existing scenarios_diamonds.json → existing_ids   (resumable run)
  ├─ filter drafts by --stage / --limit / existing_ids
  └─ ThreadPoolExecutor(max_workers=workers):
        for each draft → expand_one_scenario(draft, user_prompt)
          ├─ raw = call_llm(SYSTEM_PROMPT, user_prompt)
          ├─ scenario = _parse_scenario_json(raw)
          ├─ overwrite scenario["stage"] = draft["stage"]
          ├─ overwrite scenario["id"] = generate_scenario_id(...)
          └─ merge_and_save_scenarios([scenario], output_dir)   (incremental save per result)
```

---

## 6. Resumable-run guarantee

After every successful expansion the result is merged into `scenarios_diamonds.json` immediately (Section 5, `merge_and_save_scenarios`); subsequent invocations skip any draft whose `expected_id = SCN_<STAGE>_<index>` is already present in the file. Failures are recorded out-of-band so partial successes are preserved.

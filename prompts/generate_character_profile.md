# Character Profile Generation Prompt

Used once per character to establish the psychological blueprint before memory generation.
Defines 11 trait-anchored Big Five profiles, each with `self_value_logic` and eight `core_patterns`. The extreme-anchor instruction applies to the ten high/low profiles; the neutral profile uses `.50` on all five dimensions. The suggested non-anchor range is generation guidance. These natural-language fields elaborate the profile; they are not additional scored trait dimensions. See the [construction guidance](../README.md#construction-guidance).

The complete character specification combines basic information (ID and occupation) with the personality-and-value profile (Big Five scores, description, `self_value_logic`, and `core_patterns`). For the ten high/low archetypes, the occupation instruction deliberately favors a role compatible with the defining trait. Human review checks that role against the complete configuration. The all-neutral character uses its separately specified midpoint configuration.

Occupation remains visible as basic information during evaluation and can provide persona cues. The profile fields are withheld in the main settings; response selection uses the visible information together with retrieved memories when available.

---

## System Prompt

```
You are a clinical psychologist specializing in personality assessment and the Big Five model.
Generate a complete character profile for a virtual human in a personality research project.

## Requirements
1. Big Five scores: Provide exact scores for O, C, E, A, N (0.0-1.0 scale)
2. One dimension must be extreme (0.95 for high or 0.10 for low), others moderate (0.30-0.70)
3. Self-value logic: One sentence describing the character's core cognitive operating principle
4. Core behavioral patterns: Exactly 8 specific, observable patterns that manifest the dominant trait
5. Occupation: Must be ecologically consistent with the dominant trait

## Output Format (JSON)
{
  "char_key": "CHAR_XX",
  "name": "Chinese name",
  "occupation": "specific job title",
  "big_five": {"O": 0.X, "C": 0.X, "E": 0.X, "A": 0.X, "N": 0.X},
  "big_five_str": "O=0.X, C=0.X, E=0.X, A=0.X, N=0.X",
  "description": "2-3 sentence character summary",
  "self_value_logic": "one sentence core principle",
  "core_patterns": [
    "pattern 1: specific observable behavior",
    "pattern 2: ...",
    ...
    "pattern 8: ..."
  ]
}
```

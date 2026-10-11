# Character Profile Generation Prompt

The 11 character specifications are fixed before memory generation, each with `self_value_logic` and eight `core_patterns`. The quoted prompt drafts the ten high/low archetypes; CHAR_11 uses a separately specified vector with `.50` on all five dimensions. The suggested non-anchor range is generation guidance. These natural-language fields elaborate the profile; they are not additional scored trait dimensions. See the [construction guidance](../README.md#construction-guidance).

The complete character specification contains basic information (ID and occupation), a descriptive construction label (`name`), and the personality-and-value profile (Big Five scores, description, `self_value_logic`, and `core_patterns`). The released English `name` field encodes the archetype and is used during construction. For the ten high/low archetypes, the occupation instruction deliberately favors a role compatible with the defining trait. The prompt also emphasizes the defining trait in the initial behavioral-pattern draft. Human review checks both the occupation and patterns against the complete configuration, including non-anchor traits.

Occupation remains visible as basic information during evaluation and can provide persona cues. The descriptive label and profile fields are withheld in the main settings; response selection uses the visible information together with retrieved memories when available.

---

## System Prompt (High/Low Drafts)

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

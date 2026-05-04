### 5.1 Consciousness–Decision Human-likeness

**Goal**: at the single-sample level, evaluate whether the Agent's integrated-consciousness narrative and its final behavioural decision — given a (character × scenario × retrieved memories) input — are close, on the whole, to the human-annotated ground truth and to the reactions expected from the character profile.

- **Label design**
  - For each sample, an expert familiar with the character archetype and the world setting annotates:
    - the ideal integrated-consciousness statement (which may be a structured combination of "emotional tone + core reasoning + value orientation");
    - the ideal final behavioural choice (which may be one option from the multiple-choice set, or a set of equivalent strategies);
    - a scoring rubric for "does this match the character profile?".

- **Example metrics**
  - **Behavioural human-likeness**
    - Agreement between the final behavioural choice and the expert-annotated ground truth (accuracy / F1 / top-k etc.).
    - For scenarios where multiple reasonable interpretations are allowed, the similarity to the human choice distribution (e.g. KL divergence or correlation).
  - **Consciousness human-likeness**
    - Extract emotion tags, key reasoning, and value claims from the integrated consciousness, and match them against the expert-annotated tags.
    - A 1–5 or 1–7 subjective rating, given by experts or by LLM-as-a-judge, of "does this read like a self-narrative a real person would produce in this scenario?".
  - **Combined score**
    - Combine the "consciousness human-likeness rating" and the "behavioural human-likeness rating" with appropriate weights into a unified Consciousness–Decision Human-likeness Score — one of the core headline metrics for this benchmark.

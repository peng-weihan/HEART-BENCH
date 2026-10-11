# Vibe Research: Memory Diversity — Design Motivation for Everyday Memories and Intensity Stratification

> Task: upgrade the generation strategy from "all-high-pressure scenarios" to "a complete life of memories"
> Date: 2026-03-06 | Status: **DRAFT**
> Historical design discussion; the figures and proposed checks below do not report an audit of the current release.

## 1. Problem statement

In the motivating setup discussed in this draft, all 1,000 memories revolve around the character's core personality patterns, 98% of `emotion_intensity` values fall in 0.5–0.85, and almost every scene is high-conflict / high-emotion. Industry feedback: **a real person's memory store is not made of nothing but high-pressure events** — daily trivia, calm moments, and positive experiences are equally part of the personality.

**Need**: find solid cognitive-science / psychology grounding for "introducing daily / noise memories into the memory set + stratifying emotional intensity".

---

## 2. Why we need "noise memories" — the ecological validity of autobiographical memory

### 2.1 Conway & Pleydell-Pearce (2000) — Self-Memory System

**Key reference:**
- Conway, M.A. & Pleydell-Pearce, C.W. (2000). The construction of autobiographical memories in the self-memory system. *Psychological Review*, 107(2), 261-288.

**Takeaways:**
- Autobiographical memory is organised in three layers: Lifetime Periods → General Events → Event-Specific Knowledge (ESK).
- The General Events layer contains a large amount of **everyday events** (repeated events, routine activities), not only self-defining moments.
- **Design implication**: include routine events to broaden coverage beyond high-conflict episodes.

### 2.2 Pillemer (1998) — Momentous Events vs Personal Event Memories

**Key reference:**
- Pillemer, D.B. (1998). *Momentous Events, Vivid Memories*. Cambridge, MA: Harvard University Press. (**800+ citations**)

**Takeaways:**
- Explicitly distinguishes **momentous events** (life-trajectory-changing events) from **personal event memories** (everyday small events).
- Daily memories may not be "important" but they form the **background texture** of personality.
- They give life its sense of continuity — they are the "filler" between high-intensity memories.
- **Application**: our Layer C/D memories correspond to personal event memories.

### 2.3 Waters & Fivush (2015) — three functions of autobiographical memory

**Key references:**
- Waters, T.E. & Fivush, R. (2015). Relations between narrative coherence, identity, and psychological well-being in emerging adulthood. *Journal of Personality*, 83, 441-451.
- Bluck, S., Alea, N., Habermas, T., & Rubin, D.C. (2005). A tale of three functions: The self-reported uses of autobiographical memory. *Social Cognition*, 23, 91-117.

**The three functions:**

| Function | Description | Corresponding memory tier |
|----------|-------------|---------------------------|
| **Self** | Maintains continuity of self and identity | High-intensity main-thread memories (Layer A) |
| **Social** | Social bonding, shared experience, building intimacy | Daily social memories (Layer B/C) |
| **Directive** | Guides future behaviour, problem-solving | Daily-experience memories (Layer C/D) |

- Missing daily memories = missing the memory base for Social and Directive functions.
- These functions motivate including social and practical episodes alongside self-defining events.

---

## 3. Why we need intensity stratification — emotional granularity and psychological resilience

### 3.1 Emotional Granularity (Barrett, 2004)

**Key references:**
- Barrett, L.F. (2004). Feelings or words? Understanding the content in self-report ratings of experienced emotion. *Journal of Personality and Social Psychology*, 87, 266-281.
- Tugade, M.M., Fredrickson, B.L., & Barrett, L.F. (2004). Psychological resilience and positive emotional granularity. *Journal of Personality*, 72, 1161-1190.

**Takeaways:**
- **Emotional granularity** concerns differentiation among emotional experiences.
- Even people high on N (neuroticism) are not anxious / sad every minute of every day.
- Varying intensity from calm to intense broadens the synthetic episode set. Intensity scores alone do not measure emotional granularity or establish a clinical condition.

### 3.2 Broaden-and-Build Theory (Fredrickson, 2001)

**Key reference:**
- Fredrickson, B.L. (2001). The role of positive emotions in positive psychology: The broaden-and-build theory of positive emotions. *American Psychologist*, 56, 218-226. (**12,000+ citations**)

**Takeaways:**
- Positive emotions (joy, interest, contentment, serenity) broaden the cognitive-behavioural repertoire and accumulate psychological resources.
- Positive episodes for the high-N archetype can cover daily activities, creative work (e.g. CHAR_01 is a content creator), and social relationships.
- **Design implication**: include positive and low-intensity episodes for high-N archetypes to diversify their synthetic histories.

### 3.3 Berntsen & Rubin (2002) — emotional valence distribution of autobiographical memory

**Key reference:**
- Berntsen, D. & Rubin, D.C. (2002). Emotionally charged autobiographical memories across the life span: The recall of happy, sad, traumatic, and involuntary memories. *Psychology and Aging*, 17, 636-652.

**Design interpretation:** This reference motivates attention to emotional valence and retrieval context. Valence, intensity, and thematic relevance are distinct properties. The four-tier proportions below are author-chosen sampling targets, not estimates of a population distribution.

---

## 4. Memory stratification scheme

This historical proposal uses the following four-tier sampling targets. The numerical hints describe the proposed design; they are not acceptance thresholds for the released memories.

| Tier | Share | Count | emotion_intensity | Related concept | Description |
|------|-------|-------|-------------------|--------------------|-------------|
| **A. Core thread** | 25% | 250 | 0.70-0.90 | Singer (1993) Self-Defining Memories | High conflict; directly expresses life_threads |
| **B. Thread-related** | 35% | 350 | 0.45-0.70 | Conway (2005) Working Self | Micro-expressions of personality in everyday life |
| **C. Daily positive** | 25% | 250 | 0.20-0.50 | Fredrickson (2001) Broaden-and-Build | Calm / pleasant / contented moments |
| **D. Noise** | 15% | 150 | 0.15-0.45 | Pillemer (1998) Personal Events | Random life fragments unrelated to the main threads |

---

## 5. Suggested paper-level citation paragraphs

### English

> To diversify synthetic autobiographical histories, we use an authored mixture of personality-related episodes and everyday experiences. Research on autobiographical memory, positive emotion, and the social and directive functions of remembering motivates this breadth (Pillemer, 1998; Fredrickson, 2001; Berntsen and Rubin, 2002; Bluck et al., 2005). The sampling proportions are generation heuristics. The resulting histories provide varied contexts in which each constructed persona can be expressed; their ecological validity remains an empirical question.

---

## 6. Relation to other tasks

```
Task 1: Rubin (2006)           → 5-dimension content structure of an individual memory
Task 2: Young (2003)           → trajectory mechanism of life_threads
Task 3: Singer (1993)          → initial memory-selection strategy
Task 4: McAdams (1995)         → unifying snapshots + threads architecture
Task 5: Erikson + Levinson     → time-window segmentation
This doc: Berntsen + Fredrickson  → memory intensity stratification + noise memories  ← NEW
```

The core question this document answers is **"what should the overall distribution of the memory set look like?"** — not how to write a single memory (Task 1), nor how memories aggregate into a thread (Task 2), but rather, taking 1,000 memories as a whole, what distribution of intensity and topic diversity should they exhibit.

---

## References

1. Conway, M.A. & Pleydell-Pearce, C.W. (2000). The construction of autobiographical memories in the self-memory system. *Psychological Review*, 107(2), 261-288.
2. Pillemer, D.B. (1998). *Momentous Events, Vivid Memories*. Cambridge, MA: Harvard University Press.
3. Waters, T.E. & Fivush, R. (2015). Relations between narrative coherence, identity, and psychological well-being. *Journal of Personality*, 83, 441-451.
4. Bluck, S., Alea, N., Habermas, T., & Rubin, D.C. (2005). A tale of three functions. *Social Cognition*, 23, 91-117.
5. Barrett, L.F. (2004). Feelings or words? *Journal of Personality and Social Psychology*, 87, 266-281.
6. Tugade, M.M., Fredrickson, B.L., & Barrett, L.F. (2004). Psychological resilience and positive emotional granularity. *Journal of Personality*, 72, 1161-1190.
7. Fredrickson, B.L. (2001). The role of positive emotions in positive psychology. *American Psychologist*, 56, 218-226.
8. Berntsen, D. & Rubin, D.C. (2002). Emotionally charged autobiographical memories across the life span. *Psychology and Aging*, 17, 636-652.
9. Singer, J.A. & Salovey, P. (1993). *The Remembered Self: Emotion and Memory in Personality*. New York: Free Press.

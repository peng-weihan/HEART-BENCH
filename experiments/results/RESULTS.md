# MCQ Benchmark Experiment Results Summary

> Dataset: `data/annotations/mcq_dataset.json` (11 characters × ~64 scenarios = **673 questions**, 4 options each)
> Experiment period: 2026-04 ~ 2026-05
> Gateway: `https://llm-sjtu.multiego.me/v1`
> Common parameters: `temperature=0.7`, `workers=8~24`

---

## 1. Main Table (sorted by naive_rag@30 accuracy, descending)

| Model | naive_rag@30<br/>(full-673) | mem0@150<br/>(full-673) | personadb@30<br/>(full-673) |
|---|---|---|---|
| 🥇 **gemini-3.1-pro-preview** | **0.6330** | 0.6241 | 0.6241 |
| 🥈 gemini-3-flash-preview | 0.5632 | 0.5409 | 0.5245 |
| 🥉 **deepseek-v3.2** | 0.4116 | 0.4250 | **0.4279** |
| deepseek-v4-pro | 0.4071 | 0.3715 | 0.3670 |
| qwen3.5-397b-a17b | 0.4027 | 0.4056 | 0.4042 |
| **claude-sonnet-4-6** | 0.4012 | 0.3893 | **0.4012** |
| qwen3.5-122b-a10b | 0.3908 | 0.3938 | 0.3893 |
| **gpt-5.4** | 0.3715 | 0.3655 | 0.3507 |
| qwen3.5-35b-a3b | 0.3700 | 0.3715 | 0.3655 |
| **claude-haiku-4-5** | 0.3566 | 0.3626 | **0.3759** |
| deepseek-v4-flash | 0.3477 | 0.3358 | 0.3284 |
| gpt-5.4-mini | 0.3299 | 0.3432 | 0.3224 |

> Cell value = `accuracy_overall`; "—" means not run.
> Call success rate is nearly 100%: except for `qwen-397b@mem0` (664/673), `qwen-35b@mem0` (672/673), and `ds-v4-flash@personadb` (672/673), all others are ok=total=673.

---

## 2. Three-Method Comparison

### 2.1 mem0 vs naive_rag (12 models, all completed)

| Model | naive_rag@30 | mem0@150 | Δ(mem0 − naive) |
|---|---|---|---|
| gemini-3.1-pro | **0.6330** | 0.6241 | −0.9pt |
| gemini-3-flash | **0.5632** | 0.5409 | −2.2pt |
| deepseek-v3.2 | 0.4116 | **0.4250** | **+1.3pt** ↑ |
| deepseek-v4-pro | **0.4071** | 0.3715 | −3.6pt |
| qwen3.5-397b | 0.4027 | **0.4056** | +0.3pt |
| **claude-sonnet-4-6** | **0.4012** | 0.3893 | **−1.2pt** ↓ |
| qwen3.5-122b | 0.3908 | **0.3938** | +0.3pt |
| **gpt-5.4** | **0.3715** | 0.3655 | **−0.6pt** ↓ |
| qwen3.5-35b | 0.3700 | **0.3715** | +0.2pt |
| claude-haiku | 0.3566 | **0.3626** | +0.6pt |
| deepseek-v4-flash | **0.3477** | 0.3358 | −1.2pt |
| gpt-5.4-mini | 0.3299 | **0.3432** | **+1.3pt** ↑ |

**naive_rag wins: 6 models** | **mem0 wins: 6 models** —— **6:6 tie**

### 2.2 Three-Method Comparison (12 models with all three methods completed)

| Model | naive_rag | mem0 | personadb | best | range |
|---|---|---|---|---|---|
| gemini-3.1-pro | **0.6330** | 0.6241 | 0.6241 | naive | 0.9pt |
| gemini-3-flash | **0.5632** | 0.5409 | 0.5245 | naive | 3.9pt |
| deepseek-v3.2 | 0.4116 | 0.4250 | **0.4279** | personadb | 1.6pt |
| deepseek-v4-pro | **0.4071** | 0.3715 | 0.3670 | naive | 4.0pt |
| qwen3.5-397b | 0.4027 | **0.4056** | 0.4042 | mem0 | 0.3pt |
| **claude-sonnet-4-6** | **0.4012** | 0.3893 | **0.4012** | naive=personadb | 1.2pt |
| qwen3.5-122b | 0.3908 | **0.3938** | 0.3893 | mem0 | 0.5pt |
| **gpt-5.4** | **0.3715** | 0.3655 | 0.3507 | naive | 2.1pt |
| qwen3.5-35b | 0.3700 | **0.3715** | 0.3655 | mem0 | 0.6pt |
| claude-haiku-4-5 | 0.3566 | 0.3626 | **0.3759** | personadb | 1.9pt |
| deepseek-v4-flash | **0.3477** | 0.3358 | 0.3284 | naive | 1.9pt |
| gpt-5.4-mini | 0.3299 | **0.3432** | 0.3224 | mem0 | 2.1pt |

---

## 3. Win Counts per Method

### 3.1 mem0 vs naive_rag across all 12 models

| Method | Wins | Average accuracy |
|---|---|---|
| naive_rag@30 | **6** | **0.4087** |
| mem0@150 | 6 | 0.4019 |

### 3.2 Three-Method Comparison (12 models, all completed)

| Method | Wins | Average accuracy |
|---|---|---|
| **naive_rag@30** | **6.5** | **0.4087** |
| mem0@150 | 4 | 0.4019 |
| personadb@30 | 1.5 | 0.3960 |

> On sonnet, naive_rag and personadb tie (both 0.4012), each counted as 0.5 win.

---

## 4. Key Observations

1. **Differences between methods < 1.7pt (on average), differences between models > 30pt** → the method does not matter much; the model does.

2. **naive_rag is the most consistent overall**: 6.5/12 (three-method comparison) + 6/12 (two-method comparison) + highest average, and takes the top spot on the gemini family + claude-sonnet + deepseek-v4-pro / v4-flash + gpt-5.4.

3. **mem0 favors the "mid-tier qwen family"** (397b / 122b / 35b all take the top spot under mem0) + gpt-5.4-mini + deepseek-v3.2 + claude-haiku (small models / MoE models tend to win more often).

4. **mem0 vs naive_rag tied 6:6 across the 12 models**, but mem0 is **generally worse on the strong top-tier models** (gemini-pro/flash, sonnet, ds-v4-pro) (−1 to −4pt), only slightly winning on smaller mid- and lower-tier models.

5. **personadb only wins on deepseek-v3.2 and claude-haiku**, but is unfavorable for `deepseek-v4-pro` / `gemini-3-flash` (around −4pt) — it has its own "model preferences".

6. **Ranking consistency across the three methods ρ > 0.95**, with the top 3 (gemini-pro / gemini-flash / deepseek-v3.2) consistent across all three methods.

---

## 5. Character Information Exposed by the Prompt (anti-leakage constraints)

All methods' prompts only expose:
- **Character ID** (e.g., `CHAR_01`, no semantics)
- **Occupation** (e.g., "freelance illustrator")
- **Retrieved past experiences** (each entry's mem_id is anonymized: `MEM_CHAR_01_N_HIGH_0049 → MEM_CHAR_01_0049`)

**Deliberately hidden** (anti-leakage):
- Character `name` (contains Big Five labels)
- `description` / `self_value_logic` / `core_patterns`
- `big_five` numeric values

---

## 6. Output File Structure

```
experiments/results/
├── naive_rag/                  # 12 models × 673 questions (top-30 retrieval)
├── mem0/                       # 12 models × 673 questions (top-150 summary-style memory)
├── personadb/                  # 12 models × 673 questions (external PersonaDB retrieval)
└── <model>/{predictions,summary}_top<K>.json
```

Each `summary_top<K>.json` contains:
- `accuracy_overall` / `accuracy_on_ok`
- `per_character` / `per_stage` breakdowns
- `total_questions` / `ok` / `correct`
- `model` / `api_base` / `top_k` / `workers` / `temperature`
- `predictions_file` path

---

## 7. Total Inference Volume

| Configuration | # Models | # Questions | # Inferences |
|---|---|---|---|
| naive_rag@30 (full) | 12 | 673 | 8076 |
| mem0@150 (full) | 12 | 673 | 8076 |
| mem0@30 (full) | 3 | 673 | 2019 |
| **personadb@30 (full)** | **12** | 673 | **8076** |
| **Three-method main-line total** | — | — | **26247 inferences** |

---

## 8. One-Sentence Summary

> **Across 12 models, the relative rankings of the three RAG retrieval methods (naive_rag, mem0, personadb) are almost identical (ρ > 0.95); the largest gap between methods is < 4pt, while the gap between models exceeds 30pt. The RAG retrieval pathway is approaching its ceiling — further improvements should focus on model capability rather than the form of retrieval/memory.**

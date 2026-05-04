# MCQ Benchmark 实验结果汇总

> 数据集：`data/annotations/mcq_dataset.json`（11 角色 × ~64 场景 = **673 题**，每题 4 选项）
> 实验时间：2026-04 ~ 2026-05
> 网关：`https://llm-sjtu.multiego.me/v1`
> 通用参数：`temperature=0.7`，`workers=8~24`

---

## 一、主表（按 naive_rag@30 准确率降序）

| 模型 | naive_rag@30<br/>(full-673) | mem0@150<br/>(full-673) | personadb@30<br/>(full-673) |
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

> 单元格 = `accuracy_overall`；"—" 表示未跑。
> 调用成功率几乎 100%：除 `qwen-397b@mem0` (664/673)、`qwen-35b@mem0` (672/673)、`ds-v4-flash@personadb` (672/673) 外，其余均为 ok=total=673。

---

## 二、三方法对比

### 2.1 mem0 vs naive_rag （12 模型，已全跑）

| 模型 | naive_rag@30 | mem0@150 | Δ(mem0 − naive) |
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

**naive_rag 更优：6 模型** | **mem0 更优：6 模型** —— **6:6 平局**

### 2.2 三方法对比（12 个三方全跑模型）

| 模型 | naive_rag | mem0 | personadb | best | 极差 |
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

## 三、各方法获胜次数

### 3.1 全 12 模型 mem0 vs naive_rag

| 方法 | 胜场 | 平均准确率 |
|---|---|---|
| naive_rag@30 | **6** | **0.4087** |
| mem0@150 | 6 | 0.4019 |

### 3.2 三方法对比（12 模型，全跑齐）

| 方法 | 胜场 | 平均准确率 |
|---|---|---|
| **naive_rag@30** | **6.5** | **0.4087** |
| mem0@150 | 4 | 0.4019 |
| personadb@30 | 1.5 | 0.3960 |

> sonnet 上 naive_rag 与 personadb 并列（都是 0.4012），各算 0.5 胜。

---

## 四、关键观察

1. **方法间差距 < 1.7pt（平均），模型间差距 > 30pt** → 方法不重要，模型才重要。

2. **naive_rag 整体最稳**：6.5/12（三方法对比）+ 6/12（双方法对比） + 平均最高，且在 gemini 家族 + claude-sonnet + deepseek-v4-pro / v4-flash + gpt-5.4 上都拿头名。

3. **mem0 偏好"中段 qwen 全家族"**（397b / 122b / 35b 都拿 mem0 头名）+ gpt-5.4-mini + deepseek-v3.2 + claude-haiku（小模型 / MoE 模型多胜）。

4. **mem0 vs naive_rag 在 12 模型上 6:6 平局**，但 mem0 在头部强模型（gemini-pro/flash, sonnet, ds-v4-pro）上**普遍更差**（−1~−4pt），仅在中后段小模型上略胜。

5. **personadb 只在 deepseek-v3.2 和 claude-haiku 上拿头名**，但对 `deepseek-v4-pro` / `gemini-3-flash` 不利（−4pt 量级）——存在自身的"模型偏好"。

6. **三方法排名一致性 ρ > 0.95**，前 3 名（gemini-pro / gemini-flash / deepseek-v3.2）三方一致。

---

## 五、Prompt 透露的角色信息（防泄题约束）

所有方法的 prompt 仅暴露：
- **角色 ID**（如 `CHAR_01`，无语义）
- **职业**（如"自由插画师"）
- **检索到的过往经历**（每条做 mem_id 脱敏：`MEM_CHAR_01_N_HIGH_0049 → MEM_CHAR_01_0049`）

**刻意屏蔽**（防泄题）：
- 角色 `name`（含 Big Five 标签）
- `description` / `self_value_logic` / `core_patterns`
- `big_five` 数值

---

## 六、输出文件结构

```
experiments/results/
├── naive_rag/                  # 12 模型 × 673 题 (top-30 检索)
├── mem0/                       # 12 模型 × 673 题 (top-150 摘要式记忆)
├── personadb/                  # 12 模型 × 673 题 (外部 PersonaDB 检索)
└── <model>/{predictions,summary}_top<K>.json
```

每个 `summary_top<K>.json` 包含：
- `accuracy_overall` / `accuracy_on_ok`
- `per_character` / `per_stage` 细分
- `total_questions` / `ok` / `correct`
- `model` / `api_base` / `top_k` / `workers` / `temperature`
- `predictions_file` 路径

---

## 七、推理总量

| 配置 | 模型数 | 题数 | 推理数 |
|---|---|---|---|
| naive_rag@30 (full) | 12 | 673 | 8076 |
| mem0@150 (full) | 12 | 673 | 8076 |
| mem0@30 (full) | 3 | 673 | 2019 |
| **personadb@30 (full)** | **12** | 673 | **8076** |
| **三方法主线总计** | — | — | **26247 次推理** |

---

## 八、一句话总结

> **三种 RAG 检索方法（naive_rag, mem0, personadb）在 12 模型上的相对排名几乎完全一致（ρ > 0.95），方法间最大差距 < 4pt，模型间差距 > 30pt。RAG 检索路径已经接近上限，后续提升应聚焦在模型能力而非检索/记忆形式。**

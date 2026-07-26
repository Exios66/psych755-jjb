---
title: "Llama-3.3-70B-Instruct full-cohort baseline (prompt v1)"
subtitle: "vLLM digital-twin results — mode-collapse cautionary case"
---

**Model:** `meta-llama/Llama-3.3-70B-Instruct` · **Prompt generation:** version 1 core ladder  
**Companion:** [Cross-model memo](../memos/vllm_v1_cross_model_comparison.md) · [70B research memo](../memos/vllm_v1_llama33_70b.md) · [Llama-3.1](llm_baseline_llama31_v1.md) · [Llama-3.2-3B](llm_baseline_llama32_instruct_v1.md) · [DeepSeek](llm_baseline_deepseek_r1_distill_v1.md) · [Prompt versions](persona_prompt_versions.md)

> **Read first:** This run is a **mode-collapse** failure, not a scale win. ≈93% of generations are the constant pair `(group=18, interpersonal=12)`. Treat MAE / band tables as diagnostics of that prior, not as evidence that 70B recovers PRCA.

---

**Model tag:** `llama33_70b_instruct`  
**Sample:** N = 241 × 5 tiers = **1205** prompts  
**Parse success:** **1205/1205 (100%)**  
**Unique generated JSON texts:** **8**  
**Result CSV:** `data/vllm/llama-3-70b/llama_70b.csv`

---

## 1. Overall metrics (pooled)

| Metric | Value |
|---|---:|
| MAE group | **6.02** |
| MAE interpersonal | **4.65** |
| Exact acc group | 6.4% |
| Exact acc interpersonal | 12.4% |
| Band acc group | 26.3% |
| Band acc interpersonal | 52.0% |
| Mean signed error group | **+3.51** |
| Mean signed error IP | **−2.10** |

## 2. Mode collapse

| pred_group | n | share |
|---:|---:|---:|
| 18 | 1118 | 92.8% |
| 22 | 51 | 4.2% |
| 16 | 25 | 2.1% |
| other | 11 | 0.9% |

| pred_IP | n | share |
|---:|---:|---:|
| 12 | 1123 | 93.2% |
| 18 | 51 | 4.2% |
| 10 | 27 | 2.2% |
| other | 4 | 0.3% |

## 3. Comparison vs smaller prompt-v1 models

| Metric (all tiers) | Llama-3.3-70B | DeepSeek-R1-Distill-8B | Llama-3.2-3B | Llama-3.1-8B |
|---|---:|---:|---:|---:|
| MAE group | 6.02 | **5.22** | 5.51 | 5.92 |
| MAE interpersonal | **4.65** | 5.73 | 5.35 | 5.82 |
| Band group | 26.3% | 33.4% | **52.7%** | 28.8% |
| Band interpersonal | **52.0%** | 35.4% | 30.0% | 40.2% |

IP MAE / band look competitive only because the constant **low-IP** prior aligns with cohort base rates. Group recovery is the **worst** of the four models.

## 4. RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.05 → 6.07; IP 4.71 → 4.66  
- **RQ2 transit:** group 6.05 → 6.08; IP 4.71 → 4.69 (flat; no 8B-style IP collapse)  
- **RQ3 full:** group MAE **5.83**; IP **4.51** — tiny movement, still collapsed

## 5. Metrics by tier

| tier | n_GT | mae_group | mae_IP | band_group | band_IP | mean_err_group | mean_err_IP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1205 | 6.02 | 4.65 | 26.3% | 52.0% | +3.51 | −2.10 |
| demos | 241 | 6.05 | 4.71 | 26.6% | 51.0% | +3.58 | −2.01 |
| employment | 241 | 6.07 | 4.66 | 25.7% | 51.9% | +3.53 | −2.09 |
| geo | 241 | 6.07 | 4.66 | 26.1% | 51.9% | +3.53 | −2.09 |
| transit | 241 | 6.08 | 4.69 | 26.1% | 51.9% | +3.58 | −2.01 |
| full | 241 | 5.83 | 4.51 | 27.0% | 53.5% | +3.32 | −2.28 |

## 6. Band confusion (pooled)

### Group

| gt_band | high | low | moderate |
|:--------|----:|----:|---------:|
| high | 23 | 0 | 227 |
| low | 24 | 1 | 625 |
| moderate | 11 | 1 | 293 |

### Interpersonal

| gt_band | high | low | moderate |
|:--------|----:|----:|---------:|
| high | 2 | 204 | 14 |
| low | 0 | 614 | 26 |
| moderate | 0 | 334 | 11 |

## 7. Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | Spurious IP; group below chance |
| Person-specific variance | **No** (8 unique outputs) |
| RQ1–RQ3 tier gains | Null under collapse |

---

*Source report: `data/vllm/llama-3-70b/llama_70b.md` · evaluator artifacts in the same directory.*

---
title: "DeepSeek-R1-Distill-Llama-8B full-cohort baseline (prompt v1)"
subtitle: "vLLM digital-twin results on the original 5-tier Terrarium prompts"
---

**Model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` · **Prompt generation:** version 1 core ladder  
**Companion:** [Cross-model memo](../memos/vllm_v1_cross_model_comparison.qmd) · [Llama-3.1](llm_baseline_llama31_v1.md) · [Llama-3.2-3B](llm_baseline_llama32_instruct_v1.md) · [Llama-3.3-70B](llm_baseline_llama33_70b_v1.md) · [Prompt versions](persona_prompt_versions.md)

> R1-distill generations may include `<think>…</think>` reasoning; the project ingest prefers post-`</think>` text and normalizes SentencePiece `Ġ`/`Ċ` before JSON parse.

---

**Model tag:** `deepseek_r1_distill_llama8b`  
**Sample:** N = 241 × 5 tiers = **1205** prompts  
**Parse success:** **1202/1205 (99.8%)**  
**Quantization:** fp8 (Marlin weight-only)  
**Throughput:** 1.34 samples/s · wall ≈ 450.1s  
**Export:** `psych755_vllm_deepseek_r1_distill_llama8b_full_cohort_20260726_0324`

---

## 1. Overall metrics (pooled)

| Metric | Value |
|---|---:|
| MAE group | **5.22** |
| MAE interpersonal | 5.73 |
| Exact acc group | 6.2% |
| Exact acc interpersonal | 5.9% |
| Band acc group | 33.4% |
| Band acc interpersonal | 35.4% |

## 2. Comparison vs Llama-3.1-8B-Instruct

| Metric (all tiers) | DeepSeek-R1-Distill | Llama-3.1-8B |
|---|---:|---:|
| MAE group | **5.22** | 5.92 |
| MAE interpersonal | **5.73** | 5.82 |
| Band group | **33.4%** | 28.8% |
| Band interpersonal | 35.4% | **40.2%** |

## 3. RQ deltas (vs demos)

- **RQ1 employment:** group MAE 5.12 → 5.16; IP 5.93 → 5.66  
- **RQ2 transit:** group MAE 5.12 → 5.15; IP 5.93 → 5.80; IP band 32.4% → 35.0%  
- **RQ3 full:** group MAE 5.40; group band 31.1% → 35.0%; IP MAE 5.93 → **5.42**

## 4. Metrics by tier

| tier | n_GT | mae_group | mae_IP | band_group | band_IP | mean_err_group | mean_err_IP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1202 | 5.22 | 5.73 | 33.4% | 35.4% | −0.87 | +0.65 |
| demos | 241 | 5.12 | 5.93 | 31.1% | 32.4% | −0.55 | +1.25 |
| employment | 241 | 5.16 | 5.66 | 31.5% | 36.1% | −0.43 | +1.55 |
| geo | 240 | 5.30 | 5.86 | 33.3% | 32.1% | −1.26 | +0.99 |
| transit | 240 | 5.15 | 5.80 | 36.3% | 35.0% | −0.71 | −0.03 |
| full | 240 | 5.40 | 5.42 | 35.0% | 41.7% | −1.38 | −0.52 |

## 5. Band confusion (pooled)

### Group

| gt_band | low | moderate | high |
|:--------|----:|---------:|-----:|
| low | 199 | 446 | 3 |
| moderate | 99 | 203 | 2 |
| high | 96 | 154 | 0 |

### Interpersonal

| gt_band | low | moderate | high |
|:--------|----:|---------:|-----:|
| low | 225 | 335 | 78 |
| moderate | 127 | 170 | 47 |
| high | 70 | 119 | 31 |

## 6. Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (99.8%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | Near chance; IP best at `full` |
| RQ1 employment helps | Negligible / slight IP |
| RQ2 transit helps | Mostly flat (no IP collapse) |
| RQ3 more context helps | IP yes at full; group mixed |

---

*Source report: `data/vllm/deepseek-distilled/deepseek_v1.md`.*

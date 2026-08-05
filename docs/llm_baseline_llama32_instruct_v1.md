---
title: "Llama-3.2-3B-Instruct full-cohort baseline (prompt v1)"
subtitle: "vLLM digital-twin results on the original 5-tier Terrarium prompts"
---

**Model:** `meta-llama/Llama-3.2-3B-Instruct` · **Prompt generation:** pre–v3.1 packaging (version 1 core ladder)  
**Companion:** [Cross-model memo](../memos/vllm_v1_cross_model_comparison.qmd) · [Llama-3.1-8B](llm_baseline_llama31_v1.md) · [DeepSeek](llm_baseline_deepseek_r1_distill_v1.md) · [Llama-3.3-70B](llm_baseline_llama33_70b_v1.md) · [Prompt version map](persona_prompt_versions.md)

> Same **prompt v1** ladder and N=241 cohort as the 8B baseline. A `Llama-3.2-3B` (base) run only parsed ~7% of outputs; this page is the **Instruct** counterpart used for evaluation.

---

**Model tag:** `llama32_3b_instruct`  
**Sample:** N = 241 × 5 tiers = **1205** prompts  
**Parse success:** 1205/1205 (100.0%)  
**Throughput:** 20.97 samples/s (~57.5s)  
**Export:** `psych755_vllm_llama32_3b_instruct_full_cohort_20260726_0252`

---

## 1. Research questions

1. **RQ1 — Employment:** Does employment context reduce absolute PRCA error vs demos?
2. **RQ2 — Transit:** Do transportation cues improve CA prediction / change error?
3. **RQ3 — Full context:** Does cumulative context improve recovery / alter stereotyping?

Tracked metrics: MAE, exact score match, band accuracy, band distance, signed mean error.

---

## 2. Executive interpretation

Across 1,205 predictions:

| Metric | Observed | Naive baseline | Verdict |
|---|---:|---:|---|
| Exact group | 9.1% | ~4% | Above chance |
| Exact interpersonal | 7.7% | ~4% | Above chance |
| Band group | 52.7% | ~33% | Above chance |
| Band interpersonal | 30.0% | ~33% | At/below chance |
| MAE group | 5.51 | — | Still large vs classical ML floor (~4.5) |
| MAE interpersonal | 5.35 | — | Still large |

**RQ1:** Employment vs demos: group MAE 5.59→5.59 (Δ 0.00); IP 5.07→5.25 (Δ 0.18).

**RQ2:** Transit vs demos: group MAE 5.59→5.47 (Δ −0.12); IP 5.07→5.65 (Δ 0.58); IP band 29.5%→29.5%.

**RQ3:** Full vs demos: group MAE 5.29 vs 5.59; group band 53.9%→50.6%; IP MAE 5.07→5.53; IP band 29.5%→31.5%.

---

## 3. Comparison vs Llama-3.1-8B-Instruct (same prompts/cohort)

| Metric (all tiers) | 3B-Instruct | 8B-Instruct |
|---|---:|---:|
| MAE group | 5.51 | 5.92 |
| MAE interpersonal | 5.35 | 5.82 |
| Band acc group | 52.7% | 28.8% |
| Band acc interpersonal | 30.0% | 40.2% |

On this v1 ladder, the 3B Instruct model has **lower MAE** and **much higher group band accuracy** than 8B Instruct, while 8B leads on interpersonal band accuracy.

---

## 4. Metrics by tier

| tier | n | mae_group | mae_interpersonal | exact_group | band_group | exact_IP | band_IP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1205 | 5.51 | 5.35 | 9.1% | 52.7% | 7.7% | 30.0% |
| demos | 241 | 5.59 | 5.07 | 8.3% | 53.9% | 7.5% | 29.5% |
| employment | 241 | 5.59 | 5.25 | 8.7% | 53.5% | 7.9% | 29.0% |
| geo | 241 | 5.63 | 5.27 | 8.7% | 52.7% | 7.9% | 30.3% |
| transit | 241 | 5.47 | 5.65 | 8.3% | 52.7% | 7.5% | 29.5% |
| full | 241 | 5.29 | 5.53 | 11.2% | 50.6% | 7.9% | 31.5% |

### Deltas vs demos

| tier | Δ MAE group | Δ MAE IP | Δ band group | Δ band IP |
|---|---:|---:|---:|---:|
| employment | 0.00 | +0.18 | −0.4 pp | −0.4 pp |
| geo | +0.04 | +0.21 | −1.2 pp | +0.8 pp |
| transit | −0.12 | +0.58 | −1.2 pp | 0.0 pp |
| full | −0.29 | +0.46 | −3.3 pp | +2.1 pp |

---

## 5. Band confusion (all tiers)

### Group

| gt_band | low | moderate | high |
|:--------|----:|---------:|-----:|
| low | 620 | 30 | 0 |
| moderate | 290 | 15 | 0 |
| high | 246 | 4 | 0 |

### Interpersonal

| gt_band | low | moderate | high |
|:--------|----:|---------:|-----:|
| low | 54 | 558 | 28 |
| moderate | 38 | 299 | 8 |
| high | 38 | 174 | 8 |

---

## 6. Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | **Yes** (100%) |
| Exact digital-twin recovery | **No** (single-digit exact rates) |
| Coarse band recovery | Partial — group above chance; IP at/below chance |
| RQ1 employment helps | Negligible MAE change |
| RQ2 transit helps | Small group MAE gain; IP MAE worse |
| RQ3 more context helps | Mixed; full lowers group MAE, IP mixed |

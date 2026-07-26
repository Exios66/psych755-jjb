---
title: "Memo: Do v1 vLLM models recover PRCA as digital twins?"
subtitle: "Research memorandum — cross-model comparison on prompt-v1 Terrarium tiers"
author: Jack J. Burleson
date: 2026-07-26
---

**Research question:** Across full-cohort vLLM runs on the **same prompt-v1** five-tier ladder (`demos` → `employment` → `geo` → `transit` → `full`), which open-weight instruct / distill models best recover ground-truth PRCA group and interpersonal scores — and do employment, transit, or cumulative context reduce absolute error (RQ1–RQ3)?

**Formal method pages:** [`docs/llm_baseline_llama31_v1.md`](../docs/llm_baseline_llama31_v1.md) · [`docs/llm_baseline_llama32_instruct_v1.md`](../docs/llm_baseline_llama32_instruct_v1.md) · [`docs/llm_baseline_deepseek_r1_distill_v1.md`](../docs/llm_baseline_deepseek_r1_distill_v1.md) · [`docs/persona_prompt_versions.md`](../docs/persona_prompt_versions.md)

**Sibling memos (per model):** [`vllm_v1_llama31_8b.md`](vllm_v1_llama31_8b.md) · [`vllm_v1_llama32_3b.md`](vllm_v1_llama32_3b.md) · [`vllm_v1_deepseek_r1_distill.md`](vllm_v1_deepseek_r1_distill.md)

---

## Answer, Response, + Summary of Results

Using the Prolific↔Qualtrics matched analytic cohort (**n = 241**; inner join; complete scorable PRCA), we evaluated three GPU vLLM exports on **identical prompt-v1** Terrarium narratives (241 × 5 tiers = **1,205** prompts each):

| Model | Tag / export | Parse | Hardware notes |
|---|---|---:|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `20260726_0221` | 100% | fp8 Marlin WO · A5000 · 12.3 samp/s |
| `meta-llama/Llama-3.2-3B-Instruct` | `20260726_0252` | 100% | ~21.0 samp/s |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `20260726_0324` | **99.8%** (1202/1205) | fp8 Marlin WO · 1.34 samp/s · `</think>` strip |

Metrics follow the project evaluator (`ca_personas.evaluate`): MAE / exact match / band accuracy on group & interpersonal PRCA (6–30; bands low≤13 / moderate 14–19 / high≥20). Classical ML floor for comparison: best transit **group MAE ≈ 4.49** (Ridge; [`docs/ml_baselines.md`](../docs/ml_baselines.md)).

**Short answer:** None of the three models is a high-fidelity digital twin (exact match single-digit). On pooled MAE, **DeepSeek-R1-Distill-Llama-8B** wins group CA (**5.22**), **Llama-3.2-3B-Instruct** wins interpersonal MAE (**5.35**) and group band accuracy (**52.7%**), and **Llama-3.1-8B-Instruct** uniquely recovers interpersonal bands above chance at demos/geo (~52%) but **collapses when transit is added** (IP MAE 4.67 → **8.17**). Employment (RQ1) is negligible for all three; transit/full (RQ2–RQ3) help or hurt depending on model × subscale.

![Cross-model MAE, band accuracy, and tier trajectories](figures/vllm_v1_cross_model_memo.png)

### Head-to-head (all tiers pooled)

| Model | MAE group ↓ | MAE IP ↓ | Exact G | Exact IP | Band G | Band IP |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 5.92 | 5.82 | 6.1% | 9.0% | 28.8% | **40.2%** |
| Llama-3.2-3B-Instruct | 5.51 | **5.35** | **9.1%** | 7.7% | **52.7%** | 30.0% |
| DeepSeek-R1-Distill-Llama-8B | **5.22** | 5.73 | 6.2% | 5.9% | 33.4% | 35.4% |

All remain **above** the tabular Ridge floor (group MAE 4.49). Band accuracy vs ~33% chance: Llama-3.2 dominates **group** bands; Llama-3.1 leads **interpersonal** bands (driven by demos/employment/geo, not transit).

### RQ1–RQ3 pattern by model

| RQ | Llama-3.1-8B | Llama-3.2-3B | DeepSeek-R1-Distill-8B |
|---|---|---|---|
| **RQ1 employment** | Negligible (Δ MAE G −0.02; IP +0.07) | Negligible (G 0.00; IP +0.18) | Negligible / slight IP help (G +0.03; IP −0.27) |
| **RQ2 transit** | Group slight help; **IP disaster** (+3.51 MAE; band 51.9%→15.4%) | Small group help; IP worse (+0.58 MAE) | Flat group; IP nearly flat (−0.13 MAE); IP band +2.6 pp |
| **RQ3 full** | Group band up; IP still worse than demos | Best group MAE (5.29); IP mixed | Best IP MAE in set at full (**5.42**); group MAE not best |

**Interpretation.** Prompt-v1 transit text is a **model-dependent hazard**, not a universal aid. Llama-3.1 over-predicts interpersonal CA once mobility cues appear (signed IP error ≈ +6.5 at transit). Llama-3.2 and DeepSeek are more robust; DeepSeek is the most **tier-stable**, while Llama-3.2 is the best **coarse group-band** classifier among the three.

**Conclusion.** For prompt-v1 digital-twin evaluation on this cohort: prefer **DeepSeek-R1-Distill-Llama-8B** when minimizing group MAE, **Llama-3.2-3B-Instruct** when maximizing group band recovery / IP MAE, and treat **Llama-3.1-8B-Instruct** as the cautionary baseline that motivated prompt packaging v2/v3 (transit-induced interpersonal failure). No model yet closes the gap to classical ML (~4.5 MAE). Re-runs on v2/v3 prompts remain necessary before claiming packaging fixes stereotyping.

*Sources:* `data/vllm/llama3_1.md` · `data/vllm/llama32_instruct.md` · `data/vllm/deepseek-distilled/deepseek_v1.md` · docs LLM baselines · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

Do v2 (signal-first) and v3 (rideshare / public-transit / voice ablations) remove Llama-3.1’s transit IP collapse without harming DeepSeek / 3B gains? Are signed-error / stereotyping slices (Sex, Age, Student, Employment) aligned across models? Would temperature / decoding changes alter band recovery more than architecture?

## What other analyses pair with this memo?

Per-model memos below unpack RQ tables and confusion matrices. Classical ceilings: [`docs/ml_baselines.md`](../docs/ml_baselines.md). Feature attributions (mock LLM era): [`feature_predictive_power_ml_llm.md`](feature_predictive_power_ml_llm.md). Prompt redesign rationale: [`docs/persona_prompt_efficiency.md`](../docs/persona_prompt_efficiency.md).

---
title: "Research memo agenda — v1 vLLM digital-twin baselines"
subtitle: "Cross-model memoranda for prompt-v1 Terrarium runs"
---

**Project:** PSYCH 755 — CA persona / PRCA framework  
**Author:** Jack J. Burleson  
**Date:** 2026-07-26

---

## Purpose

Document the first wave of **live GPU vLLM** evaluations on the matched analytic cohort (**N = 241**) using **prompt version 1** (five cumulative tiers: `demos` → `full`). These memos answer the primary persona RQs (employment / transit / cumulative context) for each model and rank models head-to-head against the classical ML floor (~4.5 group MAE).

## Completed memos (vLLM wave 1)

| Memo | Model | Headline |
|---|---|---|
| [`memos/vllm_v1_cross_model_comparison.md`](../memos/vllm_v1_cross_model_comparison.md) | All three | DeepSeek best group MAE (5.22); 3B best group bands (52.7%); 8B IP collapses at transit |
| [`memos/vllm_v1_llama31_8b.md`](../memos/vllm_v1_llama31_8b.md) | Llama-3.1-8B-Instruct | IP MAE 4.67 → 8.17 at transit; packaging v2/v3 motivation |
| [`memos/vllm_v1_llama32_3b.md`](../memos/vllm_v1_llama32_3b.md) | Llama-3.2-3B-Instruct | Lower MAE than 8B; robust-ish transit; IP bands ~chance |
| [`memos/vllm_v1_deepseek_r1_distill.md`](../memos/vllm_v1_deepseek_r1_distill.md) | DeepSeek-R1-Distill-Llama-8B | Best group MAE; tier-stable; 99.8% parse |

## Method pages (tables)

| Page | Role |
|---|---|
| [`llm_baseline_llama31_v1.md`](llm_baseline_llama31_v1.md) | Full Llama-3.1 tables |
| [`llm_baseline_llama32_instruct_v1.md`](llm_baseline_llama32_instruct_v1.md) | Full Llama-3.2 Instruct tables |
| [`llm_baseline_deepseek_r1_distill_v1.md`](llm_baseline_deepseek_r1_distill_v1.md) | Full DeepSeek tables |
| [`persona_prompt_versions.md`](persona_prompt_versions.md) | v1 → v2 → v3 map |

## Shared design

- **Cohort:** File A/B/C inner join; analytic n = 241  
- **Prompts:** Terrarium second-person; tiers demos / employment / geo / transit / full  
- **Metrics:** MAE, exact match, band accuracy, signed error (`ca_personas.evaluate`)  
- **Figure:** [`memos/figures/vllm_v1_cross_model_memo.png`](../memos/figures/vllm_v1_cross_model_memo.png)

## Intentionally deferred

- v2 / v3 prompt re-runs on the same three models  
- Live TreeSHAP / stereotyping slice memos per model  
- Temperature / constrained-decoding ablations  

## Reproduction

Re-score from packaged exports under `data/vllm/` (gitignored raw generations) or re-run GPU `scripts/run_vllm.sh` against `outputs/vllm_prompts/`. Publish memos with Quarto render + Posit publish skill.

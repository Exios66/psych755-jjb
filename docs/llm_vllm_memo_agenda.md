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
| [`memos/vllm_v1_cross_model_comparison.qmd`](../memos/vllm_v1_cross_model_comparison.qmd) | All four | DeepSeek best group MAE (5.22); 3B best group bands (52.7%); 8B IP collapses at transit; 70B mode-collapse |
| [`memos/vllm_v1_llama31_8b.qmd`](../memos/vllm_v1_llama31_8b.qmd) | Llama-3.1-8B-Instruct | IP MAE 4.67 → 8.17 at transit; packaging v2/v3 motivation |
| [`memos/vllm_v1_llama32_3b.qmd`](../memos/vllm_v1_llama32_3b.qmd) | Llama-3.2-3B-Instruct | Lower MAE than 8B; robust-ish transit; IP bands ~chance |
| [`memos/vllm_v1_deepseek_r1_distill.qmd`](../memos/vllm_v1_deepseek_r1_distill.qmd) | DeepSeek-R1-Distill-Llama-8B | Best group MAE; tier-stable; 99.8% parse |
| [`memos/vllm_v1_llama33_70b.qmd`](../memos/vllm_v1_llama33_70b.qmd) | Llama-3.3-70B-Instruct | Mode collapse (~93% constant 18/12); not a scale win |

## Method pages (tables)

| Page | Role |
|---|---|
| [`llm_baseline_llama31_v1.md`](llm_baseline_llama31_v1.md) | Full Llama-3.1 tables |
| [`llm_baseline_llama32_instruct_v1.md`](llm_baseline_llama32_instruct_v1.md) | Full Llama-3.2 Instruct tables |
| [`llm_baseline_deepseek_r1_distill_v1.md`](llm_baseline_deepseek_r1_distill_v1.md) | Full DeepSeek tables |
| [`persona_prompt_versions.qmd`](persona_prompt_versions.qmd) | v1 → v2 → v3 map |

## Shared design

- **Cohort:** File A/B/C inner join; analytic n = 241  
- **Prompts:** Terrarium second-person; tiers demos / employment / geo / transit / full  
- **Metrics:** MAE, exact match, band accuracy, signed error (`ca_personas.evaluate`)  
- **Figure:** [`memos/figures/vllm_v1_cross_model_memo.png`](../memos/figures/vllm_v1_cross_model_memo.png)

## Intentionally deferred (future work)

- **Canonical `v3_enhanced` refresh** — v2 is GPU-evaluated on Llama-3.1-8B, Llama-3.2-3B-Instruct, and DeepSeek-R1-Distill-8B (v2_enhanced; `exports/v2/`), and v3 greedy 8-tier ablations on Llama-3.1, 3.2-3B-Instruct, and 3.3-70B (`exports/v3/`); the committed v3 packages are greedy-decode (identical to `prior_v3_greedy`), so the `v3_enhanced` decode refresh remains — see [`persona_prompt_versions.qmd`](persona_prompt_versions.qmd).  
- **Completed since the v1 wave:** v2/v3 evaluation memo ([`memos/vllm_v2_v3_evaluation.qmd`](../memos/vllm_v2_v3_evaluation.qmd)) and real live-export stereotyping slices ([`memos/live_llm_stereotyping_slices.qmd`](../memos/live_llm_stereotyping_slices.qmd)).  
- Temperature / constrained-decoding ablations (70B `large_model` preset)  

## Reproduction

Re-score from packaged exports under `data/vllm/` (gitignored raw generations) or re-run GPU `scripts/run_vllm.sh` against `outputs/vllm_prompts/`. Publish memos with Quarto render + Posit publish skill.

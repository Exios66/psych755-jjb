---
title: "Research memo agenda — extended secondary experiments"
subtitle: "Post-hoc memorandum plan aligned to existing transit / CA methods"
---

**Project:** PSYCH 755 — CA persona / PRCA framework  
**Author:** Jack J. Burleson  
**Date:** 2026-07-25

---

## Purpose

After the primary persona-tier LLM pipeline and the first wave of secondary Random Forest / Welch analyses, several **open questions** remained at the ends of the published memos (geo → transit; CA → transit; Q27/Q28; covariate follow-ups). This agenda lists the additional research memoranda designed to answer those questions with the **same methodological family**: matched File A/B/C cohort, `Q26` weekly+ `regular_transit` outcome (unless noted), stratified CV balanced Random Forests, and Welch / Mann–Whitney contrasts where appropriate.

## Already completed (wave 1)

| Memo | Finding (seeded) |
|---|---|
| Transit riders → CA | Regular riders lower CA (group *d* ≈ −0.46) |
| Geo → transit | Lat/long AUC ≈ 0.551 |
| CA → transit | Group+IP CA AUC ≈ 0.590 |
| Car / employment / rideshare | Rideshare ≫ car ≫ CA ≈ geo ≫ employment |
| Q27 / Q28 traditional ML | Q28 AUC ≈ 0.762; Q27 ≈ 0.589 |

## Extended agenda (wave 2 — implemented)

| # | Memo | Open question answered | CLI / module | Primary result (seed 42) |
|---|---|---|---|---|
| 1 | [`memos/demographics_predict_transit.md`](../memos/demographics_predict_transit.md) | Unused Prolific demos; student status named open | `followup-experiments --experiments demographics` | AUC ≈ **0.618** (Age dominates) |
| 2 | [`memos/country_predicts_transit.md`](../memos/country_predicts_transit.md) | Country vs raw coordinates | `--experiments country` | AUC ≈ **0.552** ≈ geo |
| 3 | [`memos/q28_conditioned_on_car.md`](../memos/q28_conditioned_on_car.md) | Does Q28 retain lift after car? | `--experiments nested_q28_car` | Q28 alone 0.665 → Q28+Q21 **0.730** (n=149) |
| 4 | [`memos/ca_mobility_joint_predicts_transit.md`](../memos/ca_mobility_joint_predicts_transit.md) | Psych vs mobility importance (complete-case) | `--experiments ca_q28_car` | Joint AUC **0.736**; CA alone weak on this subset |
| 5 | [`memos/country_car_predicts_transit.md`](../memos/country_car_predicts_transit.md) | Country × car interaction | `--experiments country_car` | Joint AUC **0.699** > either alone |
| 6 | [`memos/q27_intensity_among_riders.md`](../memos/q27_intensity_among_riders.md) | Does Q27 help *within* regular riders? | `--experiments q27_among_regular` | Best AUC ≈ **0.549** — weak |
| 7 | [`memos/common_n_head_to_head.md`](../memos/common_n_head_to_head.md) | AUC ranks confounded by missingness? | `--experiments common_n` | On n=139, **Q28 still first** (0.659) |
| 8 | [`memos/residual_ca_after_rideshare.md`](../memos/residual_ca_after_rideshare.md) | Third-variable / residual CA after Q28 | `--experiments residual_ca_q28` | CA adds only **+0.021** AUC over Q28 |
| 9 | [`memos/mi_head_to_head.md`](../memos/mi_head_to_head.md) | MI restore demos/CA vs attenuate Q28? | `--experiments mi_head_to_head` | **Q28 lead preserved** (0.762); demos/CA not restored |

## Methods shared across wave 2

- **Package:** [`src/ca_personas/followup_experiments.py`](../src/ca_personas/followup_experiments.py)
- **CLI:** `ca-personas followup-experiments --join inner --seed 42`
- **Artifacts:** `outputs/followup_experiments/`
- **Figures:** `memos/figures/*_followup_memo.png` + overview
- **Write-up hub:** [`docs/secondary_rq_followup_experiments.md`](secondary_rq_followup_experiments.md)

## vLLM digital-twin memos (wave 1 — prompt v1)

Live GPU baselines are documented separately in [`docs/llm_vllm_memo_agenda.md`](llm_vllm_memo_agenda.md):

| Memo | Headline |
|---|---|
| [`memos/vllm_v1_cross_model_comparison.md`](../memos/vllm_v1_cross_model_comparison.md) | DeepSeek best group MAE; 3B best group bands; 8B IP collapses at transit; 70B collapsed |
| [`memos/vllm_v1_llama31_8b.md`](../memos/vllm_v1_llama31_8b.md) | Llama-3.1-8B cautionary baseline |
| [`memos/vllm_v1_llama32_3b.md`](../memos/vllm_v1_llama32_3b.md) | Llama-3.2-3B Instruct lower MAE than 8B |
| [`memos/vllm_v1_deepseek_r1_distill.md`](../memos/vllm_v1_deepseek_r1_distill.md) | DeepSeek-R1-Distill tier-stable; best group MAE |
| [`memos/vllm_v1_llama33_70b.md`](../memos/vllm_v1_llama33_70b.md) | Llama-3.3-70B mode-collapse cautionary case |

## Intentionally deferred (future work; need further live LLM or new data)

- **v2 / v3 prompt re-runs** on the same GPU models (design extensions implemented in code; not yet GPU-evaluated — [`persona_prompt_versions.md`](persona_prompt_versions.md))
- Live-LLM TreeSHAP / surrogate SHAP (mock already covered)
- Richer / MNAR-sensitive MI of car items (wave-2 MI head-to-head is implemented; see [`memos/mi_head_to_head.md`](../memos/mi_head_to_head.md))
- City/density GIS layers beyond Qualtrics lat/long
- Ordinal multiclass `Q26` models (threshold sensitivity exists in transit-CA module)

## Reproduction

```bash
ca-personas followup-experiments --join inner --seed 42
pytest tests/test_followup_experiments.py
quarto render
```

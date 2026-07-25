---
title: "Memo: Feature predictive power for CA (SHAP / F1)"
subtitle: "Research memorandum — ML vs LLM attributions across persona tiers"
author: Jack J. Burleson
date: 2026-07-25
---

**Research question:** Which demographic and behavioral features have the greatest predictive power for PRCA group and interpersonal communication-apprehension scores under traditional machine-learning models and LLM persona agents — as measured by SHAP attributions and band-level F1?

---

## Answer, Response, + Summary of Results

Using the Prolific↔Qualtrics matched analytic cohort (File A + File B stacked joined to File C; **252** matched; analytic **n = 241** with complete scorable PRCA items), we evaluated feature predictive power across the project’s cumulative information tiers (`demos` → `employment` → `geo` → `transit`). Traditional ML used cross-validated **Random Forest** and **KNN** regressors targeting ground-truth Group and Interpersonal CA (6–30). LLM persona agents were evaluated on the same tiers and metrics (pipeline validated with the deterministic **mock** provider; swap `--provider ollama|openrouter` for live models). Attribution used **TreeSHAP** on RF models predicting true CA, plus **surrogate SHAP** on an RF fit to LLM numeric outputs. Classification quality used **macro / weighted / per-band F1** after mapping scores to classroom bands (low ≤13, moderate 14–19, high ≥20).

Full notebook: [`notebooks/feature_predictive_power_shap.ipynb`](../notebooks/feature_predictive_power_shap.ipynb) · module `ca_personas.shap_eval` · CLI `ca-personas shap-eval --seed 42`.

**Short answer:** Yes — structured covariates carry measurable predictive power for true CA under Random Forest, with **ride-share frequency (Q28)**, **employment status**, **public-transit frequency (Q26)**, and **survey geolocation** dominating TreeSHAP importance at the transit tier. Band **macro-F1 rises from ~0.31 (demos) to ~0.42 (transit)** for RF as information accumulates. LLM-surrogate SHAP shows the persona agent’s outputs also track transit/geo/demographics (surrogate \(R^2 \approx 0.77\) for Group CA under mock), but **ML recovers true CA with substantially lower MAE** than the mock LLM baseline at every tier.

![Composite: SHAP comparison, F1/MAE by tier, and tier ablation](figures/fig_memo_feature_power_composite.png)

### Traditional ML — predictive power by tier

Mean metrics across Group + Interpersonal targets (`random_state=42`, stratified/K-fold CV):

| Tier | RF MAE ↓ | RF macro-F1 ↑ | RF band acc | KNN MAE | KNN macro-F1 |
|---|---:|---:|---:|---:|---:|
| demos | 5.47 | 0.314 | 0.367 | 5.54 | 0.309 |
| employment | 5.35 | 0.312 | 0.369 | 5.54 | 0.334 |
| geo | 4.75 | 0.386 | 0.452 | 5.37 | 0.345 |
| **transit** | **4.48** | **0.422** | **0.506** | 4.98 | 0.372 |

At the transit tier, Group CA RF band F1 components were low=0.64 · moderate=0.42 · high=0.10 (macro=0.39); Interpersonal CA was stronger on high-band F1 (macro=0.46). Exact integer accuracy stayed low (~0.07), confirming that **band F1**, not exact match, is the appropriate classification summary on a 6–30 scale.

**Tier ablation (Group CA, RF).** Adding geo cut MAE by **−0.55** points vs employment; adding transit cut another **−0.37**, with corresponding macro-F1 gains (+0.05 then +0.03). Employment alone produced only a small MAE improvement over demos.

### SHAP — features driving ML predictions of true CA

TreeSHAP mean |SHAP| (aggregated to raw fields) for RF → true **Group CA** at the transit tier:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---:|
| 1 | Q28 (ride-share days) | 1.93 |
| 2 | Employment status | 0.83 |
| 3 | Q26 (public-transit days) | 0.58 |
| 4 | LocationLatitude | 0.57 |
| 5 | LocationLongitude | 0.54 |
| 6 | Age | 0.53 |

![ML TreeSHAP bar chart for Group CA](figures/fig_shap_bar_ml_group.png)

Transportation items and employment dominate over core demographics (Sex, Student status), supporting the primary research focus: **RQ1–RQ2 covariates are not decorative — they reshape predicted apprehension under classical ML.**

### LLM persona agents — performance and surrogate SHAP

Under the offline **mock** provider (pipeline dry-run), mean LLM MAE at transit was **7.99** with band macro-F1 **0.32** — worse than RF on both score and band metrics. Live models should replace these figures before substantive LLM claims.

Surrogate SHAP (RF predicting LLM Group CA from tabular features; \(R^2 = 0.77\)) ranked **Q28, lat/long, Sex, Age, Q26** highest — overlapping the ML ranking on transit/geo but elevating Sex/Age relative to employment. Side-by-side comparison:

![ML SHAP vs LLM-surrogate SHAP](figures/fig_shap_ml_vs_llm_compare.png)

![Band macro-F1 by tier — RF vs LLM](figures/fig_f1_ml_vs_llm.png)

**Conclusion.** Feature predictive power for communication apprehension in this matched cohort is **real but moderate**: richer tiers improve RF MAE and band F1, and TreeSHAP attributes that lift primarily to **transit use, employment, and geolocation**. LLM persona evaluation should always report **band F1 alongside MAE**, and should use **surrogate SHAP (or tier ablation)** to audit whether the agent tracks the same covariates that predict true CA — or instead over-weights demographic stereotypes. Mock results establish the measurement design; live LLM runs are required for final stereotyping conclusions.

*Sources:* `notebooks/feature_predictive_power_shap.ipynb` · `src/ca_personas/shap_eval.py` · `ca-personas shap-eval` · artifacts `outputs/shap_eval/` · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

Do live Ollama/OpenRouter models show the same SHAP overlap with RF as the mock agent, or do they amplify Sex/Age at the expense of transit? Would interaction SHAP (employment × transit) explain residual stereotyping patterns? How stable are rankings under alternate band cutoffs?

## What other analyses pair with this memo?

Secondary RQs on transit↔CA mean differences and CA/geo → regular-transit Random Forests (`memos/ca_scores_predict_transit.md`, `memos/geo_predicts_transit.md`) ask the reverse predictive direction. The present memo asks which **inputs** best recover CA under ML and LLM personification.

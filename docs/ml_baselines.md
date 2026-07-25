---
title: "Classical ML baselines for CA prediction"
subtitle: "Stage-one Random Forest & KNN on persona information tiers"
---

**Notebook:** [`notebooks/stage_one_ml_baseline.ipynb`](../notebooks/stage_one_ml_baseline.ipynb)  
**Code:** [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)  
**Artifacts:** `outputs/ml_baseline/`  
**Companion comparison:** [`ml_vs_llm.md`](ml_vs_llm.md)

---

## 1. Research role

Before asking whether an LLM stereotypes CA from persona prompts, we need a **tabular ceiling / floor**: how well do standard regressors recover the same PRCA targets from the same cumulative feature sets?

| Tier | Features (available in full File A/B/C) |
|---|---|
| `demos` | Age, Sex, Country of residence, **Student status** (base demographics layer; optional ethnicity/nationality/language when present) |
| `employment` | + Employment status |
| `geo` | + survey lat/long |
| `transit` | + Q26–Q29, Q20, Q21 |

Targets: `gt_group_ca`, `gt_interpersonal_ca` (6–30). Models: Random Forest regressor (200 trees) and distance-weighted KNN (*k* = 3). Evaluation: 5-fold CV MAE, band accuracy, and related metrics (`random_state=42`).

## 2. Full-cohort results (N = 241)

![Group-CA MAE by tier](figures/ml_baseline_mae_group.png)

### Group CA (MAE ↓ better)

| Tier | RF MAE | KNN MAE | RF band acc. |
|---|---:|---:|---:|
| demos | 5.72 | 5.83 | 0.365 |
| employment | 5.59 | 5.80 | 0.365 |
| geo | 5.05 | 5.74 | 0.419 |
| **transit** | **4.68** | 5.21 | **0.481** |

### Interpersonal CA

| Tier | RF MAE | KNN MAE | RF band acc. |
|---|---:|---:|---:|
| demos | 5.22 | 5.27 | 0.369 |
| employment | 5.11 | 5.22 | 0.373 |
| geo | 4.45 | 5.00 | 0.485 |
| **transit** | **4.27** | 4.75 | **0.531** |

## 3. Interpretation

1. **Adding context reduces MAE.** RF group MAE falls from **5.72** (`demos`) to **4.68** (`transit`), a **1.04**-point improvement on a 24-point scale (−18.2%).
2. **Geography and transit matter more than employment alone.** RF group MAE drops 0.13 from `demos`→`employment`, then 0.54 from `employment`→`geo`, then 0.37 from `geo`→`transit`.
3. **RF beats KNN** at every tier on MAE for both subscales in this run (e.g., transit group 4.68 vs 5.21).
4. **Band accuracy** ranges from **0.365** (`demos` RF group) to **0.531** (`transit` RF interpersonal). Tabular models recover some signal but do not pin the low/moderate/high band reliably from demographics + context alone.
5. **R²** remains near zero or negative until the `transit` tier (group R² = 0.06; interpersonal R² = 0.10 at `transit`) — CA is only weakly linear in these covariates.

## 4. Link to the LLM manuscript

- These MAE numbers are the **classical reference** for [`ml_vs_llm.md`](ml_vs_llm.md): an LLM tier “helps” only if it approaches or beats the best ML MAE on the same tier.
- Because even the best RF MAE remains **4.27–4.68** points, persona prompts that land in that range are competitive with the tabular learner — not magical. The mock LLM on Posit remains higher (group MAE ≈7.70 at `demos` / ≈8.40 at `transit` after the student-status base-demos prompt refresh; [`index.qmd`](../index.qmd)).
- Feature-importance diagnostics ([factor_feature_importance.md](factor_feature_importance.md); [memo](../memos/feature_predictive_power_ml_llm.md)) show *which* covariates carry the tabular signal the LLM is being asked to use (Q28 ranks first for predicting CA).

## 5. Reproducibility

```bash
# Requires ../sibling_data/ File A, File B, File C
jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
```

Seeded metrics live in `outputs/ml_baseline/ml_baseline_metrics.csv`.

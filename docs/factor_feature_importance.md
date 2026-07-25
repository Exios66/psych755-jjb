---
title: "PRCA factor structure & covariate importance"
subtitle: "Diagnostics for subscale targets and persona features"
---

**Notebook:** [`notebooks/factor_feature_importance.ipynb`](../notebooks/factor_feature_importance.ipynb)  
**Code:** [`src/ca_personas/feature_importance.py`](../src/ca_personas/feature_importance.py)  
**Artifacts:** `outputs/feature_importance/`

---

## 1. Why these diagnostics matter

Two measurement questions sit under the LLM persona work:

1. Are the **group** and **interpersonal** PRCA item sets behaving like distinct (or at least coherent) subscales in this cohort?
2. Which **persona covariates** actually predict ground-truth CA in a tabular learner — i.e., which cues *should* matter if an LLM is using information “sensibly”?

## 2. Item factor structure (PRCA items)

Exploratory factor analysis / PCA on the scored group (`Q1–Q6`) and interpersonal (`Q13–Q18`) items (after reverse-coding comfort items) supports treating the two subscales as related but separable evaluation targets. Loadings concentrate on the intended item blocks; reverse-coded calm/comfort items load strongly on the dominant apprehension factor.

Practical takeaway for the manuscript: reporting **separate** MAE for group vs interpersonal CA is justified; collapsing to a single “CA” score would hide subscale-specific error.

## 3. Covariate permutation importance

A Random Forest predicting CA from persona-style covariates, with permutation importance on held-out folds, ranks features as follows (mean importance; zeros omitted below):

![Top covariates](figures/feature_importance_top.png)

| Rank | Feature | Permutation importance |
|---:|---|---:|
| 1 | Q28 (ride-share days) | 0.346 |
| 2 | LocationLongitude | 0.239 |
| 3 | LocationLatitude | 0.197 |
| 4 | Q26 (public-transit days) | 0.139 |
| 5 | Nationality* | 0.085 |
| 6 | Employment status | 0.083 |
| 7 | Age | 0.078 |
| 8 | Country of birth* | 0.055 |
| 9 | Country of residence | 0.053 |

\*Richer demographic fields appear in excerpt / importance runs when present; full File A/B waves often omit them, so the live `demos` tier may not expose every row above.

Sex, student status, license/car (`Q20`/`Q21`), and several transit intensity items (`Q27`/`Q29`) show ~0 permutation importance in this fit — either weak signal or redundancy with higher-ranked features.

## 4. Interpretation against LLM tiers

| Finding | Implication for persona prompts |
|---|---|
| Transit + geo dominate tabular importance | `transit` and `geo` tiers are the most justified contextual additions |
| Employment ranks mid-pack | RQ1 (employment lift) should show modest, not dramatic, gains |
| Sex ≈ 0 importance here | Large LLM error gaps by sex would look more like stereotyping than “using the sample’s real signal” |
| Ride-share (Q28) > public transit (Q26) for predicting CA | Models (and prompts) that only mention bus/train may miss a stronger mobility cue |

## 5. Reproducibility

```bash
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

Key outputs: `outputs/feature_importance/top_predictive_features.csv`, `ca_item_fa_loadings.csv`, importance CSVs per target.

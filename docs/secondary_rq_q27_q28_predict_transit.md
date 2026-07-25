---
title: "Do Q27 and Q28 predict regular public transit use?"
subtitle: "Secondary research question — traditional ML write-up for transit intensity & ride-share days"
---

**Project:** PSYCH 755 — CA persona / PRCA framework  
**Analysis code:** [`src/ca_personas/transit_covariate_rf.py`](../src/ca_personas/transit_covariate_rf.py)  
**CLI:** `ca-personas covariate-transit-rf --specs q27_intensity q28_days q27_q28 --join inner --seed 42`  
**Artifacts:** `outputs/transit_covariate_rf/{q27_intensity,q28_days,q27_q28}/`  
**Research memo:** [`memos/q27_q28_predict_transit.md`](../memos/q27_q28_predict_transit.md)  
**Manuscript:** [`index.qmd`](../index.qmd)

---

## 1. Research question

Among matched Prolific↔Qualtrics respondents, do **Q27** (public-transit rides on a typical use day) and **Q28** (ride-share days in the last three months) predict **regular** public-transit use (`Q26` weekly+) in a traditional Random Forest classifier—and which item carries the usable signal?

## 2. Methods

- **Sample:** File A + File B inner-joined to File C; analytic base n = 241 with complete PRCA and usable `Q26`. Complete-case n = 239 (Q27), 241 (Q28), 239 (joint).
- **Outcome:** `regular_transit` = `Q26` ∈ {`4-8 days a month`, `8 or more days a month`}.
- **Models:** One-hot categorical features → balanced `RandomForestClassifier` (500 trees, `min_samples_leaf=3`), stratified 5-fold `cross_val_predict`, seed 42; Gini + permutation importance (`scoring="roc_auc"`).
- **Benchmarks:** chance AUC = 0.500; geo RF = 0.551; CA RF = 0.590; rideshare family Q28+Q29 = 0.745.

## 3. Results

| Model | n | ROC-AUC | AP | Bal. acc. | F1 |
|---|---:|---:|---:|---:|---:|
| Q28 only | 241 | **0.762** | 0.689 | 0.730 | 0.702 |
| Q27 + Q28 | 239 | **0.761** | 0.690 | 0.696 | 0.667 |
| Q28 + Q29 | 233 | 0.745 | 0.655 | 0.731 | 0.709 |
| Q27 only | 239 | **0.589** | 0.513 | 0.623 | 0.467 |
| CA benchmark | 241 | 0.590 | — | — | — |
| Geo benchmark | 241 | 0.551 | — | — | — |
| Chance | — | 0.500 | — | 0.500 | — |

Regular-transit prevalence rises from 15.7% (Q28 Never) to 94.1% (Q28 8+ days/month). Q27 shows a milder intensity gradient concentrated in the 1–2 rides bin. Joint-model permutation importance: Q28 ≫ Q27.

## 4. Interpretation

Q28 is a practically meaningful traditional-ML predictor of weekly+ transit in this cohort. Q27 is weaker (AUC = 0.589) and redundant once Q28 is included. See the research memo for full prose interpretation and caveats.

## 5. Reproducibility

```bash
ca-personas covariate-transit-rf \
  --specs q27_intensity q28_days q27_q28 \
  --join inner --seed 42 \
  --figures-dir memos/figures
```

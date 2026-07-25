---
title: "Do car access, employment, or ride-share predict regular transit?"
subtitle: "Secondary research questions — geo-memo follow-up Random Forests"
---

**Project:** PSYCH 755 — CA persona / PRCA framework  
**Analysis code:** [`src/ca_personas/transit_covariate_rf.py`](../src/ca_personas/transit_covariate_rf.py)  
**Notebooks:**  
[`secondary_rq_car_access_transit_rf.ipynb`](../notebooks/secondary_rq_car_access_transit_rf.ipynb) ·  
[`secondary_rq_employment_transit_rf.ipynb`](../notebooks/secondary_rq_employment_transit_rf.ipynb) ·  
[`secondary_rq_rideshare_transit_rf.ipynb`](../notebooks/secondary_rq_rideshare_transit_rf.ipynb) ·  
[`secondary_rq_transit_covariate_followups.ipynb`](../notebooks/secondary_rq_transit_covariate_followups.ipynb)  
**CLI:** `ca-personas covariate-transit-rf --join inner --seed 42`  
**Artifacts:** `outputs/transit_covariate_rf/`  
**Memos:** [`memos/transit_covariate_followups.md`](../memos/transit_covariate_followups.md) and family memos under `memos/`

---

## 1. Research questions

The geography → transit memo ([`memos/geo_predicts_transit.md`](../memos/geo_predicts_transit.md)) asked which other features might predict regular public-transit use after finding lat/long discrimination of AUC = **0.551**. Communication-apprehension scores were already tested (AUC = **0.590**; [`memos/ca_scores_predict_transit.md`](../memos/ca_scores_predict_transit.md)). This write-up evaluates the remaining named candidates:

1. Driver's license & car access (`Q20`, `Q21`)
2. Employment status
3. Ride-share frequency (`Q28`, `Q29`)
4. Joint mobility bundle (all of the above)

## 2. Methods

### 2.1 Sample & outcome

- **Sources:** Prolific File A + File B stacked, inner-joined to Qualtrics File C  
- **Base analytic cohort:** n = 241 with complete PRCA group/interpersonal items and usable `Q26`  
- **Outcome:** `regular_transit` = `Q26` ∈ {`4-8 days a month`, `8 or more days a month`}  
- **Complete-case frames:** car access n = 149; employment n = 241; ride-share n = 233; bundle n = 143  

### 2.2 Models

For each feature family: `OneHotEncoder` → balanced `RandomForestClassifier` (500 trees, `min_samples_leaf=3`), stratified 5-fold `cross_val_predict` probabilities, Gini + permutation importance (`scoring="roc_auc"`, 30 repeats). Seed = 42.

Benchmarks pasted from companion seeded runs: geo AUC = 0.551; CA AUC = 0.590; chance = 0.500.

## 3. Results

| Spec | n | ROC-AUC | AP | Bal. acc. | F1 |
|---|---:|---:|---:|---:|---:|
| Mobility bundle | 143 | 0.747 | 0.605 | 0.717 | 0.667 |
| Ride-share (Q28/Q29) | 233 | 0.745 | 0.655 | 0.731 | 0.709 |
| Car access (Q20/Q21) | 149 | 0.607 | 0.515 | 0.639 | 0.476 |
| CA benchmark | 241 | 0.590 | — | — | — |
| Geo benchmark | 241 | 0.551 | — | — | — |
| Employment | 241 | 0.528 | 0.424 | 0.560 | 0.586 |
| Chance | — | 0.500 | — | 0.500 | — |

**Ride-share days (`Q28`)** dominate: regular-transit prevalence rises from 15.7% (Never) to 94.1% (8+ days/month). **Car access (`Q21`)** is the main car-family driver (No access = 75.9% regular vs Yes = 30.8%). **Employment** differences are small.

## 4. Interpretation

Mobility self-reports—especially ride-share frequency—predict weekly+ transit far better than survey geolocation or CA scores in this cohort. The joint bundle’s near-parity with ride-share alone suggests limited incremental value from car/employment once `Q28` is included, on the smaller overlapping sample. Associations are descriptive, not causal.

## 5. Limitations

- Unequal complete-case *N* across families; car items are ~38% missing.  
- Same-wave survey items; shared method variance between `Q26` and `Q28` is possible.  
- Coarse employment coding.  
- Qualtrics/Prolific convenience sample; not population transit forecasting.

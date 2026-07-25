---
title: "Memo: Do CA scores predict regular transit?"
subtitle: "Research memorandum — group & interpersonal PRCA"
author: Jack J. Burleson
date: 2026-07-25
---

**Research question:** Do group and interpersonal communication-apprehension (PRCA) scores predict whether an individual takes public transportation regularly?

**Formal write-up:** [`docs/secondary_rq_ca_predicts_transit.md`](../docs/secondary_rq_ca_predicts_transit.md)

---

## Answer, Response, + Summary of Results

Using the Prolific↔Qualtrics matched cohort for this project (File A + File B stacked joined to File C on Prolific ID / `Q0`; **252** matched rows; analytic **n = 241** with complete scorable PRCA items and usable `Q26`), we asked whether ground-truth **group** and **interpersonal** CA scores (range 6–30) predict **regular** public-transit use, defined as `Q26` ∈ {`4-8 days a month`, `8 or more days a month`} (weekly-or-more). A balanced **Random Forest** (stratified 5-fold CV, `random_state=42`) was trained on the two CA subscales, with group-only and interpersonal-only ablations and a chance baseline (ROC-AUC = 0.50).

Figures below match the formal write-up [`docs/secondary_rq_ca_predicts_transit.md`](../docs/secondary_rq_ca_predicts_transit.md) and the seeded CLI artifacts under `outputs/ca_transit_rf/` (`ca-personas ca-transit-rf --join inner --seed 42`).

**Short answer:** Yes — but modestly. Higher CA is associated with *lower* odds of regular transit, and a CA-only Random Forest recovers **above-chance** discrimination (CV ROC-AUC = **0.590**). Group CA is the stronger predictor.

![Mean CA by transit group and ROC curve for the CA Random Forest](figures/ca_predicts_transit_memo.png)

**Descriptive associations (regular vs not-regular).** Regular riders (n = 101) report lower CA than non-regular riders (n = 140):

- **Group CA:** M = 13.04 vs 15.76 (Δ = **−2.72**); point-biserial *r* = −0.223, *p* = 0.0005  
- **Interpersonal CA:** M = 13.31 vs 15.04 (Δ = **−1.73**); point-biserial *r* = −0.147, *p* = 0.023  

**Random Forest (stratified CV).** Predicting regular transit from CA scores:

| Model | ROC-AUC |
|---|---:|
| Group + interpersonal CA | **0.590** |
| Group CA only | 0.555 |
| Interpersonal CA only | 0.506 |
| Chance | 0.500 |

Permutation importance likewise ranks **group CA** above interpersonal CA. Combining both subscales improves AUC by about +0.035 over the better single-feature model.

**Conclusion.** Group and interpersonal CA do carry usable—but weak—predictive signal for regular public-transit use in this matched sample. The relationship is consistent in direction (higher apprehension ↔ less weekly+ transit) and is driven more by **group** than interpersonal CA. CA alone is not a strong classifier of transit habits (many high-CA respondents still ride regularly), so these scores should be treated as a modest behavioral correlate, not a deterministic proxy.

*Sources:* `notebooks/secondary_rq_ca_transit_rf.ipynb` · `src/ca_personas/ca_transit_rf.py` · `ca-personas ca-transit-rf` · write-up `docs/secondary_rq_ca_predicts_transit.md` · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

Is the CA–transit link causal (e.g., anxiety reducing use of shared vehicles), reverse (transit exposure reducing apprehension), or driven by third variables such as urbanicity, car access, employment, or country of residence? Same-wave self-reports cannot separate these accounts.

## What other features may also well-predict regular public transit use?

Survey geolocation (latitude/longitude) was examined in a companion Random Forest (`secondary_rq_geo_transit_rf.ipynb`) and showed similarly modest AUC (≈ 0.55). Promising next features include car access / license (`Q20`/`Q21`), employment status, country or city density proxies, and ride-share frequency (`Q28`/`Q29`), alone and jointly with CA.

---
title: "Memo: Do regular transit riders differ in CA?"
subtitle: "Research memorandum — observational PRCA contrast"
author: Jack J. Burleson
date: 2026-07-25
---

**Research question:** Do individuals who take public transportation regularly have communication-apprehension (CA) scores that differ from non-regular riders in the matched cohort?

**Formal write-up:** [`docs/secondary_rq_transit_ca.md`](../docs/secondary_rq_transit_ca.md)

---

## Answer, Response, + Summary of Results

Using the Prolific↔Qualtrics matched cohort (File A + File B stacked joined to File C; **252** matched rows; analytic **n = 241** with complete PRCA items), we compared ground-truth **group** and **interpersonal** CA (range 6–30) for **regular** public-transit riders vs everyone else. Regular transit = `Q26` ∈ {`4-8 days a month`, `8 or more days a month`} (weekly-or-more). Primary test: Welch *t*; effect sizes Cohen's *d* / Hedges' *g*; bootstrap 95% CIs (`ca-personas transit-ca --join inner --seed 42`).

**Short answer:** Yes. Regular riders report **lower** CA on both subscales. Group CA shows the clearer gap (Δ = **−2.72**, *d* = −0.46, *p* = .0003); interpersonal CA is smaller but still significant (Δ = **−1.73**, *d* = −0.30, *p* = .020).

![Mean CA by Q26 ridership and regular vs not-regular](figures/transit_riders_ca_memo.png)

**Key numbers (regular n = 101 vs not-regular n = 140):**

| Subscale | Regular M | Not-regular M | Δ | Welch *p* | Cohen's *d* |
|---|---:|---:|---:|---:|---:|
| Group CA | 13.04 | 15.76 | **−2.72** | 0.0003 | −0.46 |
| Interpersonal CA | 13.31 | 15.04 | **−1.73** | 0.020 | −0.30 |

Mean CA also falls as `Q26` intensity rises (“Never” group CA M ≈ 17.4; “8+ days/month” M ≈ 12.8).

**Conclusion.** Weekly+ public-transit use is associated with lower PRCA scores in this sample — especially **group** CA. The association is real but modest; it motivates (and is consistent with) the companion CA→transit Random Forest (AUC = 0.590).

*Sources:* `notebooks/secondary_rq_transit_ca.ipynb` · `src/ca_personas/transit_ca.py` · write-up `docs/secondary_rq_transit_ca.md` · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

Is lower CA a cause of transit use, a consequence of routine shared-vehicle exposure, or a correlate of urbanicity / car access / employment? Same-wave data cannot decide.

## What other features may also well-predict regular public transit use?

Geography (lat/long RF AUC = 0.551) and CA scores jointly (AUC = 0.590) each carry modest signal. Car access / license (`Q20`/`Q21`), employment, and ride-share frequency (`Q28`) are natural next covariates.

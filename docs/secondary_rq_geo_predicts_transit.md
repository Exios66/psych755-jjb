---
title: "Does geography predict regular public transit use?"
subtitle: "Secondary research question 4 — Random Forest write-up"
---

**Project:** PSYCH 755 — CA persona / PRCA framework  
**Analysis code:** [`src/ca_personas/geo_transit_rf.py`](../src/ca_personas/geo_transit_rf.py)  
**Notebook:** [`notebooks/secondary_rq_geo_transit_rf.ipynb`](../notebooks/secondary_rq_geo_transit_rf.ipynb)  
**CLI:** `ca-personas geo-transit-rf --join inner`  
**Artifacts:** `outputs/geo_transit_rf/`  
**Companion memo:** [`memos/geo_predicts_transit.md`](../memos/geo_predicts_transit.md)

---

## 1. Research question

Among Prolific↔Qualtrics **matched** respondents with complete PRCA ground truth, do Qualtrics survey **latitude and longitude** predict whether an individual takes **public transportation regularly**?

This complements the observational CA contrast (`secondary_rq_transit_ca`) and the CA→transit RF (`secondary_rq_ca_transit_rf`) by asking how much **place** alone classifies the same binary transit outcome.

## 2. Methods

### 2.1 Sample

- **Sources:** Prolific File A + File B (stacked) joined to Qualtrics File C
- **Analytic N:** 241 with complete PRCA items and non-missing `LocationLatitude` / `LocationLongitude`
- **Class balance:** 101 regular (41.9%) / 140 not regular (58.1%)

### 2.2 Outcome & features

| Piece | Detail |
|---|---|
| Outcome | Regular transit = `Q26` ∈ {`4-8 days a month`, `8 or more days a month`} |
| Features | `LocationLatitude`, `LocationLongitude` (approximate IP/browser geolocation) |
| Model | Balanced `RandomForestClassifier` (500 trees, `min_samples_leaf=3`), features z-scored |
| Validation | Stratified 5-fold CV; out-of-fold probabilities |
| Baselines | Chance (ROC-AUC = 0.50); country-of-residence-only RF |

## 3. Results

![Survey geolocation by transit use](figures/geo_scatter_latlon_by_transit.png)

**Spatial descriptives.** Regular and non-regular riders overlap heavily in lat/lon space:

| Group | n | Mean lat | Mean lon |
|---|---:|---:|---:|
| Regular | 101 | 43.05 | −62.00 |
| Not regular | 140 | 41.10 | −69.07 |

![ROC-AUC vs baselines](figures/geo_rf_auc_vs_baselines.png)

### 3.1 Predictive performance

| Model | ROC-AUC |
|---|---:|
| Latitude + longitude RF | **0.551** |
| Country-of-residence RF | 0.549 |
| Chance / prevalence | 0.500 |

Permutation importance ranks **longitude** above **latitude**. Continuous coordinates lift AUC by only ≈ **+0.002** over a country-only forest — essentially no gain beyond coarse country membership.

## 4. Interpretation

**Geography, as captured by Qualtrics lat/long, is a weak predictor of regular transit in this sample.**

1. Discrimination is **above chance** but stays in the weak band (AUC ≈ 0.55).
2. Nearly all of that signal is redundant with **country of residence**.
3. Compared with the CA→transit RF (AUC ≈ 0.59), place alone is a slightly weaker classifier of weekly+ transit.
4. For the primary persona project, the `geo` tier supplies a coarse place cue that may help CA prediction indirectly (via country / urbanicity correlates) but should not be treated as a precise transit-accessibility measure.

## 5. Limitations

1. Qualtrics coordinates are approximate IP/browser locations, not verified home addresses.
2. Country composition and urbanicity may confound continuous lat/long effects.
3. Observational association ≠ causal effect of place on transit use.
4. External validity limited to this matched cohort.

## 6. Reproducibility

```bash
ca-personas geo-transit-rf --join inner
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_geo_transit_rf.ipynb
```

Key outputs: `outputs/geo_transit_rf/geo_transit_rf_results_card.json`, `geo_transit_rf_metrics.csv`, `fig_*.png`.

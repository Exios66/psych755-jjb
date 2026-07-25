# Reproducible Science

## A Research Memorandum

**Researcher:** Jack J. Burleson  
**Date:** July 25, 2026  

**Research Or Analytical Question:**  
Does survey geographical location (latitude and longitude) predict whether an individual takes public transportation regularly?

---

## Answer, Response, + Summary of Results

Using the Prolific↔Qualtrics matched cohort (File A + File B stacked joined to File C on Prolific ID / `Q0`; **252** matched rows; analytic **n ≈ 241** with complete PRCA items and non-missing Qualtrics `LocationLatitude` / `LocationLongitude`), we asked whether approximate survey **geolocation** predicts **regular** public-transit use. Regular transit is defined as `Q26` ∈ {`4-8 days a month`, `8 or more days a month`} (weekly-or-more). A balanced **Random Forest** (stratified 5-fold CV, `random_state=42`) used latitude and longitude as the sole features, compared against chance (ROC-AUC = 0.50) and a country-of-residence-only Random Forest baseline. Re-run with `ca-personas geo-transit-rf --join inner --seed 42` and cite `outputs/geo_transit_rf/` for exact N.

**Short answer:** Only modestly. Lat/long recover **above-chance** but weak discrimination (CV ROC-AUC ≈ **0.55**), essentially matching a country-only model. Geography alone is not a strong predictor of regular transit in this sample. The companion CA→transit RF yields a similar modest AUC (**0.572**; see `docs/secondary_rq_ca_predicts_transit.md`).

![Survey geolocation by transit use and ROC curve for the lat/lon Random Forest](figures/geo_predicts_transit_memo.png)

**Spatial descriptives.** Regular riders (n = 101) and non-regular riders (n = 140) overlap heavily in lat/lon space; mean coordinates differ only slightly (regular: lat ≈ 43.05, lon ≈ −62.00; not regular: lat ≈ 41.10, lon ≈ −69.07). Visual inspection of the scatter shows no sharp geographic separation between groups.

**Random Forest (stratified CV).** Predicting regular transit from latitude and longitude:

| Model | ROC-AUC |
|---|---:|
| Latitude + longitude RF | **0.551** |
| Country-of-residence RF | 0.549 |
| Chance / prevalence | 0.500 |

Permutation importance ranks **longitude** (mean AUC drop ≈ 0.34) above **latitude** (≈ 0.26). The lift of continuous coordinates over a country-only forest is negligible (Δ AUC ≈ +0.002).

**Conclusion.** Qualtrics survey latitude and longitude show a **small** predictive association with regular public-transit use—statistically above chance, but practically weak and largely redundant with coarse country membership. These coordinates reflect approximate IP/browser geolocation rather than verified home addresses, so they should be interpreted as a rough place cue, not a precise measure of transit accessibility. Thus we conclude that geographical position, as captured here, is a **limited** predictor of weekly+ public transportation use.

*Sources:* `notebooks/secondary_rq_geo_transit_rf.ipynb` · `src/ca_personas/geo_transit_rf.py` · `ca-personas geo-transit-rf` · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

How much of the lat/long signal is simply country or urban/rural composition rather than local transit infrastructure? Would city-level or density features outperform raw coordinates? Same-wave Qualtrics geolocation may also misplace travelers or VPN users.

## What other features may also well-predict regular public transit use?

Communication-apprehension scores (group and interpersonal PRCA) were examined in a companion Random Forest (`notebooks/secondary_rq_ca_transit_rf.ipynb`) and showed slightly stronger discrimination (AUC ≈ 0.59). Other candidates include car access / license (`Q20`/`Q21`), employment status, and ride-share frequency (`Q28`/`Q29`).

# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`cleaning_eda_full_cohort.ipynb`](cleaning_eda_full_cohort.ipynb) | Cleaning + EDA | Load `../sibling_data` File A/B/C, clean to the analytic sample, score PRCA GT, and produce RQ-aligned EDA tables |
| [`secondary_rq_transit_ca.ipynb`](secondary_rq_transit_ca.ipynb) | Secondary RQ 1–3 | Test whether regular public-transit riders differ in CA from the larger matched cohort (Welch *t*, effect sizes, distributions, sensitivity) |
| [`secondary_rq_geo_transit_rf.ipynb`](secondary_rq_geo_transit_rf.ipynb) | Secondary RQ 4 | Random Forest: does lat/long predict regular public-transit use? (CV metrics, baselines, importances, decision surface) |
| [`secondary_rq_ca_transit_rf.ipynb`](secondary_rq_ca_transit_rf.ipynb) | Secondary RQ 5 | Random Forest: do group & interpersonal CA scores predict regular public-transit use? (+ [write-up](../docs/secondary_rq_ca_predicts_transit.md)) |
| [`secondary_rq_car_access_transit_rf.ipynb`](secondary_rq_car_access_transit_rf.ipynb) | Geo follow-up | RF: car license/access (`Q20`/`Q21`) → regular transit · [memo](../memos/car_access_predicts_transit.md) |
| [`secondary_rq_employment_transit_rf.ipynb`](secondary_rq_employment_transit_rf.ipynb) | Geo follow-up | RF: employment status → regular transit · [memo](../memos/employment_predicts_transit.md) |
| [`secondary_rq_rideshare_transit_rf.ipynb`](secondary_rq_rideshare_transit_rf.ipynb) | Geo follow-up | RF: ride-share (`Q28`/`Q29`) → regular transit · [memo](../memos/rideshare_predicts_transit.md) |
| [`secondary_rq_transit_covariate_followups.ipynb`](secondary_rq_transit_covariate_followups.ipynb) | Geo follow-ups | Head-to-head comparison of car / employment / ride-share · [memo](../memos/transit_covariate_followups.md) · [write-up](../docs/secondary_rq_transit_covariate_followups.md) |
| [`secondary_rq_followup_experiments.ipynb`](secondary_rq_followup_experiments.ipynb) | Wave-2 follow-ups | Demographics / country / nested Q28\|car / CA+mobility / common-*N* / residual CA / Q27-among-riders · [agenda](../docs/research_memo_agenda.md) · [write-up](../docs/secondary_rq_followup_experiments.md) |
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |
| [`ml_vs_llm_comparison.ipynb`](ml_vs_llm_comparison.ipynb) | Comparison | Side-by-side ML vs LLM evaluation on shared metrics (MAE, exact/band accuracy, distance-from-correct) |
| [`factor_feature_importance.ipynb`](factor_feature_importance.ipynb) | Diagnostics | Factor analysis of PRCA items + RF/permutation feature importance for persona covariates |
| [`feature_predictive_power_shap.ipynb`](feature_predictive_power_shap.ipynb) | Feature power | Dedicated SHAP + band-F1 suite for ML and LLM persona agents across tiers · [memo](../memos/feature_predictive_power_ml_llm.md) |

Supporting code:

- [`src/ca_personas/load.py`](../src/ca_personas/load.py) / [`clean.py`](../src/ca_personas/clean.py) / [`eda.py`](../src/ca_personas/eda.py)
- [`src/ca_personas/transit_ca.py`](../src/ca_personas/transit_ca.py)
- [`src/ca_personas/geo_transit_rf.py`](../src/ca_personas/geo_transit_rf.py)
- [`src/ca_personas/ca_transit_rf.py`](../src/ca_personas/ca_transit_rf.py)
- [`src/ca_personas/transit_covariate_rf.py`](../src/ca_personas/transit_covariate_rf.py)
- [`src/ca_personas/followup_experiments.py`](../src/ca_personas/followup_experiments.py)
- [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)
- [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)
- [`src/ca_personas/feature_importance.py`](../src/ca_personas/feature_importance.py)
- [`src/ca_personas/shap_eval.py`](../src/ca_personas/shap_eval.py)

```bash
pip install -r requirements.txt
pip install -e .

# Requires full-cohort File A/B/C (../sibling_data, /tmp/sibling_data, or CA_SIBLING_DATA).
# Notebooks assert the full matched cohort and do not run on excerpt fixtures.
jupyter nbconvert --to notebook --execute notebooks/cleaning_eda_full_cohort.ipynb

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_transit_ca.ipynb
# or: ca-personas transit-ca --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_geo_transit_rf.ipynb
# or: ca-personas geo-transit-rf --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_ca_transit_rf.ipynb
# or: ca-personas ca-transit-rf --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_transit_covariate_followups.ipynb
# or: ca-personas covariate-transit-rf --join inner --seed 42

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_followup_experiments.ipynb
# or: ca-personas followup-experiments --join inner --seed 42

jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb

# SHAP + band F1 feature predictive power (ML + LLM)
ca-personas shap-eval --join inner --provider mock --shap-tier transit
jupyter nbconvert --to notebook --execute notebooks/feature_predictive_power_shap.ipynb
```

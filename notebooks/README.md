# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`cleaning_eda_full_cohort.ipynb`](cleaning_eda_full_cohort.ipynb) | Cleaning + EDA | Load `../sibling_data` File A/B/C, clean to the analytic sample, score PRCA GT, and produce RQ-aligned EDA tables |
| [`secondary_rq_transit_ca.ipynb`](secondary_rq_transit_ca.ipynb) | Secondary RQ 1–3 | Test whether regular public-transit riders differ in CA from the larger matched cohort (Welch *t*, effect sizes, distributions, sensitivity) |
| [`secondary_rq_geo_transit_rf.ipynb`](secondary_rq_geo_transit_rf.ipynb) | Secondary RQ 4 | Random Forest: does lat/long predict regular public-transit use? (CV metrics, baselines, importances, decision surface) |
| [`secondary_rq_ca_transit_rf.ipynb`](secondary_rq_ca_transit_rf.ipynb) | Secondary RQ 5 | Random Forest: do group & interpersonal CA scores predict regular public-transit use? (+ [write-up](../docs/secondary_rq_ca_predicts_transit.md)) |
| [`secondary_rq_comprehensive_transit_rf.ipynb`](secondary_rq_comprehensive_transit_rf.ipynb) | Secondary RQ 6 | Feature importance + tuned kitchen-sink RF maximizing ROC-AUC for regular transit (+ [memo](../memos/comprehensive_predictors_transit.md)) |
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |
| [`ml_vs_llm_comparison.ipynb`](ml_vs_llm_comparison.ipynb) | Comparison | Side-by-side ML vs LLM evaluation on shared metrics (MAE, exact/band accuracy, distance-from-correct) |
| [`factor_feature_importance.ipynb`](factor_feature_importance.ipynb) | Diagnostics | Factor analysis of PRCA items + RF/permutation feature importance for persona covariates |

Supporting code:

- [`src/ca_personas/load.py`](../src/ca_personas/load.py) / [`clean.py`](../src/ca_personas/clean.py) / [`eda.py`](../src/ca_personas/eda.py)
- [`src/ca_personas/transit_ca.py`](../src/ca_personas/transit_ca.py)
- [`src/ca_personas/geo_transit_rf.py`](../src/ca_personas/geo_transit_rf.py)
- [`src/ca_personas/ca_transit_rf.py`](../src/ca_personas/ca_transit_rf.py)
- [`src/ca_personas/comprehensive_transit_rf.py`](../src/ca_personas/comprehensive_transit_rf.py)
- [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)
- [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)
- [`src/ca_personas/feature_importance.py`](../src/ca_personas/feature_importance.py)

```bash
pip install -r requirements.txt
pip install -e .

# Requires ../sibling_data/PRCA{ProlificExport_FileA,ProlificExport_FileB,QualtricsExport_FileC}.csv
jupyter nbconvert --to notebook --execute notebooks/cleaning_eda_full_cohort.ipynb

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_transit_ca.ipynb
# or: ca-personas transit-ca --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_geo_transit_rf.ipynb
# or: ca-personas geo-transit-rf --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_ca_transit_rf.ipynb
# or: ca-personas ca-transit-rf --join inner

jupyter nbconvert --to notebook --execute notebooks/secondary_rq_comprehensive_transit_rf.ipynb
# or: ca-personas comprehensive-transit-rf --join inner

jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

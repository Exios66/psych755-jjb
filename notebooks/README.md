# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`cleaning_eda_full_cohort.ipynb`](cleaning_eda_full_cohort.ipynb) | Cleaning + EDA | Load `../sibling_data` File A/B/C, clean to the analytic sample, score PRCA GT, and produce RQ-aligned EDA tables |
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |
| [`ml_vs_llm_comparison.ipynb`](ml_vs_llm_comparison.ipynb) | Comparison | Side-by-side ML vs LLM evaluation on shared metrics (MAE, exact/band accuracy, distance-from-correct) |
| [`factor_feature_importance.ipynb`](factor_feature_importance.ipynb) | Diagnostics | Factor analysis of PRCA items + RF/permutation feature importance for persona covariates |

Supporting code:

- [`src/ca_personas/load.py`](../src/ca_personas/load.py) / [`clean.py`](../src/ca_personas/clean.py) / [`eda.py`](../src/ca_personas/eda.py)
- [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)
- [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)
- [`src/ca_personas/feature_importance.py`](../src/ca_personas/feature_importance.py)

```bash
pip install -r requirements.txt
pip install -e .

# Requires ../sibling_data/PRCA{ProlificExport_FileA,ProlificExport_FileB,QualtricsExport_FileC}.csv
jupyter nbconvert --to notebook --execute notebooks/cleaning_eda_full_cohort.ipynb

jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |
| [`factor_feature_importance.ipynb`](factor_feature_importance.ipynb) | Diagnostics | Factor analysis of PRCA items + RF/permutation feature importance for persona covariates |

Supporting code:

- [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)
- [`src/ca_personas/feature_importance.py`](../src/ca_personas/feature_importance.py)

```bash
pip install -r requirements.txt
pip install -e .
jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

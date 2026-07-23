# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |

Supporting code: [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py).

```bash
pip install -r requirements.txt
pip install -e .
jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
```

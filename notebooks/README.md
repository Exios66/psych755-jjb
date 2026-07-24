# Notebooks

| Notebook | Stage | Purpose |
|---|---|---|
| [`stage_one_ml_baseline.ipynb`](stage_one_ml_baseline.ipynb) | Stage one | Train/evaluate Random Forest + KNN baselines on the same tiered CA prediction task used for LLM personas |
| [`ml_vs_llm_comparison.ipynb`](ml_vs_llm_comparison.ipynb) | Comparison | Side-by-side ML vs LLM evaluation on shared metrics (MAE, exact/band accuracy, distance-from-correct) |

Supporting code:

- [`src/ca_personas/ml_baseline.py`](../src/ca_personas/ml_baseline.py)
- [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)

```bash
pip install -r requirements.txt
pip install -e .
jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
```

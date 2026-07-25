---
title: "ML vs LLM on shared CA metrics"
subtitle: "Head-to-head evaluation across persona information tiers"
---

**Notebook:** [`notebooks/ml_vs_llm_comparison.ipynb`](../notebooks/ml_vs_llm_comparison.ipynb)  
**Code:** [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)  
**CLI:** `ca-personas compare --provider mock|ollama|openrouter`  
**ML reference:** [`ml_baselines.md`](ml_baselines.md)  
**Manuscript demo:** [`index.qmd`](../index.qmd) (mock pipeline on the full matched cohort; Posit uses committed `artifacts/posit_full_cohort/`)

---

## What is compared

Both agent families predict the same targets — PRCA **group** and **interpersonal** scores (6–30) — from the same cumulative information tiers (`demos` → `employment` → `geo` → `transit`).

| Family | How predictions are produced |
|---|---|
| **ML** | Cross-validated suite on tabular tier features (Ridge, Elastic Net, k-NN, RF, HistGradientBoosting, XGBoost, MLP — see [ml_baselines.md](ml_baselines.md)) |
| **LLM** | Tiered persona prompts → JSON scores/bands via Ollama or OpenRouter (`mock` for offline CI / Connect) |

Shared metrics (via `evaluate_predictions` / `summarize_errors`):

- MAE (group / interpersonal)
- Exact-score accuracy
- Band accuracy (low / moderate / high)
- Normalized score distance and band distance

## How to read the comparison

For each tier:

1. Identify the **best ML MAE** across the full suite (often Ridge / Elastic Net — not always RF or MLP; see [ml_baselines.md](ml_baselines.md)).
2. Compute `delta_vs_best_ml` for each LLM agent.
3. **Negative** Δ MAE means the LLM beat the strongest tabular baseline on that tier; **positive** means it did worse.

Band accuracy is the coarser stereotyping-relevant lens: does the model place someone in the right low/moderate/high bin even when the exact integer is off?

## Classical reference (full cohort, N = 241)

Suite **best** MAE by tier (seed = 42; full tables in [`ml_baselines.md`](ml_baselines.md)):

| Tier | Best group MAE (model) | Best interpersonal MAE (model) | RF group (ref) |
|---|---:|---:|---:|
| demos | **4.90** (MLP) | **4.52** (MLP) | 5.72 |
| employment | **4.65** (Elastic Net) | **4.39** (Ridge) | 5.59 |
| geo | **4.68** (Ridge) | **4.35** (Ridge) | 5.05 |
| transit | **4.49** (Ridge) | **4.25** (Elastic Net) | 4.68 |

Live LLM numbers depend on provider/model and are **not** baked into the Posit Connect render (the manuscript uses `provider="mock"` so Cloud builds stay keyless). To publish a specific model’s head-to-head table:

```bash
CA_LLM_PROVIDER=openrouter ca-personas compare --join inner \
  --tiers demos employment geo transit
# or
CA_LLM_PROVIDER=ollama jupyter nbconvert --to notebook --execute \
  notebooks/ml_vs_llm_comparison.ipynb
```

Artifacts land in `outputs/ml_vs_llm/` (comparison CSV, deltas, optional figures).

## Interpretation checklist for results

When a live comparison is in hand, ask:

1. **Does any LLM tier beat RF MAE?** If not, persona prompting is not recovering CA better than a shallow tabular learner.
2. **Does error shrink with richer tiers?** Parallel improvement with ML would suggest both families use employment/geo/transit signal; flat LLM curves with falling ML MAE would suggest the LLM ignores those cues.
3. **Does residual error track demographics?** That is the stereotyping RQ in the manuscript — compare absolute error by sex / **student status** (base `demos` layer) / employment / country. Use `summarize_errors_by_group(evaluation, "Student status")` or the committed `artifacts/posit_full_cohort/error_by_student_status.csv` table.
4. **Mock ≠ science.** Treat `provider=mock` outputs as deterministic pipeline diagnostics, not production-LLM claims.

## Reproducibility

```bash
pip install -e .
CA_LLM_PROVIDER=mock ca-personas compare --join inner
# Full-cohort live run (requires sibling_data + API keys / Ollama):
CA_LLM_PROVIDER=openrouter ca-personas compare --join inner
```

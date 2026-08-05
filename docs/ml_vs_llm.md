---
title: "ML vs LLM on shared CA metrics"
subtitle: "Head-to-head evaluation across persona information tiers"
---

**Notebook:** [`notebooks/ml_vs_llm_comparison.ipynb`](../notebooks/ml_vs_llm_comparison.ipynb)  
**Code:** [`src/ca_personas/compare_agents.py`](../src/ca_personas/compare_agents.py)  
**CLI:** `ca-personas compare --provider mock|ollama|openrouter`  
**Real-export harness:** [`scripts/regenerate_ml_vs_llm_shap.py`](../scripts/regenerate_ml_vs_llm_shap.py) (F1 / SHAP / band discrimination from committed `exports/`)  
**ML reference:** [`ml_baselines.md`](ml_baselines.md)  
**Live LLM baselines:** [`persona_prompt_versions.md`](persona_prompt_versions.md), [`llm_v2_v3_enhanced_variants.md`](llm_v2_v3_enhanced_variants.md)  
**Band / SHAP memo:** [`memos/feature_predictive_power_ml_llm.qmd`](../memos/feature_predictive_power_ml_llm.qmd)  
**Manuscript:** [`index.qmd`](../index.qmd)

---

## What is compared

Both agent families predict the same targets — PRCA **group** and **interpersonal** scores (6–30) — from the same cumulative information tiers (`demos` → `employment` → `geo` → `transit`).

| Family | How predictions are produced |
|---|---|
| **ML** | Cross-validated suite on tabular tier features (Ridge, Elastic Net, k-NN, RF, HistGradientBoosting, XGBoost, MLP — see [ml_baselines.md](ml_baselines.md)) |
| **LLM** | Tiered persona prompts → JSON scores via **vLLM full-cohort exports** (`exports/v1/`, `exports/v2/`, `exports/v3/`); the `mock` provider remains only for credential-free CI / Connect renders |

Shared metrics (via `evaluate_predictions` / `summarize_errors`):

- MAE (group / interpersonal)
- Exact-score accuracy
- Band accuracy (low / moderate / high) and per-band F1
- Mean band distance and normalized score / band distance
- **Low-vs-high ROC-AUC and ordinal (Hand–Till) AUC** for band discrimination (see [memo](../memos/feature_predictive_power_ml_llm.qmd))

## How to read the comparison

For each tier:

1. Identify the **best ML MAE** across the full suite (often Ridge / Elastic Net — not always RF or MLP; see [ml_baselines.md](ml_baselines.md)).
2. Compare each live LLM agent to that floor.
3. **Negative** Δ MAE means the LLM beat the strongest tabular baseline on that tier; **positive** means it did worse.

Band metrics are the coarser stereotyping-relevant lens: does the model place someone in the right low/moderate/high bin even when the exact integer is off? Report **band F1, band distance, and low-vs-high AUC alongside MAE** — the memo shows the LLM is ~1.1× worse on MAE but at chance on low-vs-high discrimination.

## Classical reference (full cohort, N = 241)

Suite **best** MAE by tier (seed = 42; full tables in [`ml_baselines.md`](ml_baselines.md)):

| Tier | Best group MAE (model) | Best interpersonal MAE (model) | RF group (ref) |
|---|---|---:|---:|---:|
| demos | **4.90** (MLP) | **4.52** (MLP) | 5.72 |
| employment | **4.65** (Elastic Net) | **4.39** (Ridge) | 5.59 |
| geo | **4.68** (Ridge) | **4.35** (Ridge) | 5.05 |
| transit | **4.49** (Ridge) | **4.25** (Elastic Net) | 4.68 |

## Live LLM baselines (full-cohort vLLM exports)

### Pooled across the 5-tier ladder (demos → full)

| Model (version) | MAE group ↓ | MAE IP ↓ | Band G | Band IP | Source |
|---|---|---:|---:|---:|---:|---|
| DeepSeek-R1-Distill-Llama-8B (v2) | **5.02** | **5.26** | 35.7% | 34.2% | `exports/v2/deepseek*` |
| DeepSeek-R1-Distill-Llama-8B (v1) | 5.22 | 5.73 | 33.4% | 35.4% | `exports/v1/deepseek*` |
| Llama-3.2-3B-Instruct (v1) | 5.51 | 5.35 | **52.7%** | 30.0% | `exports/v1/llama32*` |
| Llama-3.2-3B-Instruct (v2) | 5.73 | 6.07 | 32.2% | 42.3% | `exports/v2/llama32*` |
| Llama-3.1-8B-Instruct (v1) | 5.92 | 5.82 | 28.8% | 40.2% | `exports/v1/llama31*` |
| Llama-3.1-8B-Instruct (v2) | 5.99 | 7.63 | 29.9% | 25.7% | `exports/v2/llama31*` |
| Llama-3.3-70B-Instruct (v1) | 6.02 | 4.65† | 26.3% | 52.0%† | `exports/v1/llama33*` |

† Constant-prior mode collapse `(18, 12)` — not person-tracking. See [llm_v2_v3_enhanced_variants.md](llm_v2_v3_enhanced_variants.md) for v3 (8-tier greedy) ablations.

**Every live model remains above the classical suite floor** (best transit group MAE 4.49 Ridge; best interpersonal 4.25 Elastic Net). The best live result, DeepSeek v2 (group 5.02 / IP 5.26), is ~1.1× the Ridge floor on group and ~1.2× on interpersonal.

### Per-tier head-to-head (best live model vs ML best)

| Tier | ML best group | DeepSeek v2 group | ML best IP | DeepSeek v2 IP |
|---|---|---:|---:|---:|---:|
| demos | 4.90 | 5.05 | 4.52 | 5.12 |
| employment | 4.65 | 5.15 | 4.39 | 5.78 |
| geo | 4.68 | 4.89 | 4.35 | 5.20 |
| transit | 4.49 | 4.97 | 4.25 | 5.09 |

Δ MAE vs the suite best is positive on every tier and target; the LLM does **not** improve with richer tiers the way RF does (employment and transit slightly *hurt* DeepSeek's error).

### Band discrimination (the metric MAE hides)

At `transit`, RF separates low from high CA (AUC 0.72 group / 0.74 interpersonal; band accuracy 0.48 / 0.53; mean band distance 0.57 / 0.52), while DeepSeek v2 is at chance (low-vs-high AUC 0.53 / 0.49; band accuracy 0.36 / 0.36; band distance 0.70 / 0.68) and never emits the high band. Full tables and figures: [`memos/feature_predictive_power_ml_llm.qmd`](../memos/feature_predictive_power_ml_llm.qmd).

## Reproduce

```bash
# Real vLLM-backed F1 / SHAP / discrimination tables + figures (needs sibling data):
python scripts/regenerate_ml_vs_llm_shap.py
# Deterministic mock pipeline check (CI / Connect, no keys):
pip install -e .
CA_LLM_PROVIDER=mock ca-personas compare --join inner
# Live comparison requires the GPU export packages (exports/v1, exports/v2):
```

Artifacts land in `outputs/shap_eval/` (metrics, ablation, SHAP, band profile, discrimination CSVs) and are copied to `memos/figures/`.

## Interpretation checklist for results

1. **Does any LLM tier beat the ML floor?** No — every live model is above the Ridge/Elastic Net floor and the RF reference on all tiers.
2. **Does error shrink with richer tiers?** Parallel improvement with ML would suggest both families use employment/geo/transit signal; flat LLM curves with falling ML MAE suggest the LLM ignores those cues. DeepSeek v2 is flat-to-worse with tiers; RF improves monotonically.
3. **Does residual error track demographics?** That is the stereotyping RQ in the manuscript — compare absolute error by sex / student status / employment / mobility exposure on the live exports. Use `summarize_errors_by_group(evaluation, "Student status")` on the export row-level files.
4. **Discrimination vs calibration.** Low MAE is not low bias: report band F1, band distance, and low-vs-high AUC. The memo shows the LLM's continuous scores cannot rank apprehension even when its MAE is within ~1.1× of ML.

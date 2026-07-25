# psych755-jjb - A Summer Semester Final Project

- **Group:** Jack J. Burleson 
- **Course:** PSYCH 755, Summer Semester 2026 - UW-Madison | Madison, WI
- **Professor:** Dr. Adam Ross Nelson

<a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://wisc.edu/"><img src="https://img.shields.io/badge/UW--Madison-Research-C5050C?style=for-the-badge&logo=google-scholar&logoColor=white" alt="UW-Madison"></a>

This is a final semester project for the course PSYCH 755 with Dr. Adam Ross Nelson; It is a showcase of the effective usage of mainstream data science software & utilities.

## Members

| Name | GitHub Username |
|---|---|
| Jack J. Burleson | @Exios66 |
| Jack J. Burleson | @jjburleson |

## Research Question (primary)

When an LLM is given a dynamically-constructed, first-person “embodiment” prompt built from an individual respondent’s demographic and behavioral attributes, and instructed to predict that person’s own communication-apprehension (CA) score, how accurately does the model recover the respondent’s true PRCA subscale scores — and where accuracy fails, does the error pattern correlate systematically with demographic group membership (i.e., stereotyping) rather than random noise?

## Research Focus (primary — persona tiers)

Cumulative information tiers map onto the research focus:

| Tier | Adds | Research focus |
|---|---|---|
| `demos` | **Base demographics layer:** Age, Sex, Country, Student status (+ optional ethnicity/language when present) | Baseline / demos stereotyping |
| `employment` | Employment status | RQ1 — does employment improve accuracy / change bias? |
| `geo` | Survey lat/long context | Intermediate place cue |
| `transit` | Public transit / ride-share / license / car access | RQ2 — does transportation-use help, and is it used sensibly? |
| `full` | Open-text attitudes (Q18.1 / Q19) | RQ3 — richest personification / combined signal vs redundancy |

1. Does employment status improve prediction accuracy over demographics alone, and does it change the bias pattern (e.g., does the model now stereotype “unemployed” respondents as higher-CA, correctly or not)?
2. Does transportation-use data improve prediction accuracy, and does the model use it sensibly (e.g., inferring low transit use → higher avoidance → higher CA) or does it ignore it/misuse it?
3. Does combining employment, geography, transit, and open-text attitudes help beyond earlier tiers, or do the cues carry redundant signal?

## Secondary research questions (observational — matched cohort)

These questions use the **full Prolific↔Qualtrics matched analytic sample** (File A + File B stacked joined to File C on `Q0` / `Participant id`, complete PRCA ground truth). They do **not** require an LLM; they characterize real self-reported CA and transit behavior in the cohort.

| # | Question | Notebook | CLI / write-up |
|---|---|---|---|
| 1 | Do regular public-transit riders differ in CA from the larger matched population? By how much? | [`secondary_rq_transit_ca.ipynb`](notebooks/secondary_rq_transit_ca.ipynb) | `ca-personas transit-ca` |
| 2 | How are group / interpersonal CA distributed among regular vs non-regular vs overall? | same notebook | `outputs/transit_ca/` |
| 3 | Does the answer change under alternate “regular” cutoffs on `Q26`? | same notebook | sensitivity tables in `outputs/transit_ca/` |
| 4 | Do Qualtrics **latitude & longitude** predict regular transit use? (RF + CV) | [`secondary_rq_geo_transit_rf.ipynb`](notebooks/secondary_rq_geo_transit_rf.ipynb) | `ca-personas geo-transit-rf` |
| 5 | Do **group & interpersonal CA** scores predict regular transit use? (RF + CV) | [`secondary_rq_ca_transit_rf.ipynb`](notebooks/secondary_rq_ca_transit_rf.ipynb) | `ca-personas ca-transit-rf` · [write-up](docs/secondary_rq_ca_predicts_transit.md) |
| 6 | Do **car access / employment / ride-share** predict regular transit? (geo-memo follow-ups) | [`secondary_rq_transit_covariate_followups.ipynb`](notebooks/secondary_rq_transit_covariate_followups.ipynb) | `ca-personas covariate-transit-rf` · [write-up](docs/secondary_rq_transit_covariate_followups.md) |
| 7 | Do **Q27** (transit intensity) & **Q28** (ride-share days) predict regular transit? (traditional ML) | same follow-up CLI (`--specs q27_intensity q28_days q27_q28`) | [write-up](docs/secondary_rq_q27_q28_predict_transit.md) · [memo](memos/q27_q28_predict_transit.md) · manuscript `index.qmd` |
| 8–15 | **Wave-2 follow-ups** answering open memo questions (demographics, country, Q28\|car, CA+mobility, country×car, Q27-among-riders, common-*N*, residual CA) | [`secondary_rq_followup_experiments.ipynb`](notebooks/secondary_rq_followup_experiments.ipynb) | `ca-personas followup-experiments` · [agenda](docs/research_memo_agenda.md) · [write-up](docs/secondary_rq_followup_experiments.md) |

**Primary operationalization of “regular transit” (RQs 1–6):** `Q26` ∈ {`4-8 days a month`, `8 or more days a month`} (weekly-or-more public transit).

```bash
# RQ1–3: regular transit vs cohort CA
ca-personas transit-ca --join inner
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_transit_ca.ipynb

# RQ4: lat/long Random Forest → regular transit
ca-personas geo-transit-rf --join inner
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_geo_transit_rf.ipynb

# RQ5: group + interpersonal CA Random Forest → regular transit
ca-personas ca-transit-rf --join inner
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_ca_transit_rf.ipynb

# RQ6: car access / employment / ride-share follow-ups
ca-personas covariate-transit-rf --join inner --seed 42
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_transit_covariate_followups.ipynb

# RQ8–15: wave-2 extended follow-ups (demographics, nesting, residual CA, …)
ca-personas followup-experiments --join inner --seed 42
jupyter nbconvert --to notebook --execute notebooks/secondary_rq_followup_experiments.ipynb
```

Supporting code:

- [`src/ca_personas/transit_ca.py`](src/ca_personas/transit_ca.py) · [`notebooks/secondary_rq_transit_ca.ipynb`](notebooks/secondary_rq_transit_ca.ipynb)
- [`src/ca_personas/geo_transit_rf.py`](src/ca_personas/geo_transit_rf.py) · [`notebooks/secondary_rq_geo_transit_rf.ipynb`](notebooks/secondary_rq_geo_transit_rf.ipynb)
- [`src/ca_personas/ca_transit_rf.py`](src/ca_personas/ca_transit_rf.py) · [`notebooks/secondary_rq_ca_transit_rf.ipynb`](notebooks/secondary_rq_ca_transit_rf.ipynb) · [`docs/secondary_rq_ca_predicts_transit.md`](docs/secondary_rq_ca_predicts_transit.md)
- [`src/ca_personas/transit_covariate_rf.py`](src/ca_personas/transit_covariate_rf.py) · [`notebooks/secondary_rq_transit_covariate_followups.ipynb`](notebooks/secondary_rq_transit_covariate_followups.ipynb) · [`docs/secondary_rq_transit_covariate_followups.md`](docs/secondary_rq_transit_covariate_followups.md)
- [`src/ca_personas/followup_experiments.py`](src/ca_personas/followup_experiments.py) · [`notebooks/secondary_rq_followup_experiments.ipynb`](notebooks/secondary_rq_followup_experiments.ipynb) · [`docs/secondary_rq_followup_experiments.md`](docs/secondary_rq_followup_experiments.md) · [`docs/research_memo_agenda.md`](docs/research_memo_agenda.md)

## Suggested Project Structure + Contents

| Path | What it is |
|---|---|
| `index.qmd` | The primary manuscript. Start here. |
| `Contributions.md` | Who owned what. |
| `Getstarted.md` | Local setup (venv, sibling data, CLI, Quarto). |
| `memos/` | Individual research memos. |
| `references.bib` | Shared BibTeX file for the manuscript and memos. |
| `src/ca_personas/` | Analysis package + `ca-personas` CLI. |
| `src/inference/` | vLLM digital-twin export / batch / ingest. |
| `notebooks/` | Executable analyses (ML, FA, secondary RQs). |
| `docs/` | Framework, merge audit, secondary RQ write-ups, bugfix audit. |
| `config/default.yaml` | Default paths, tiers, LLM + cleaning settings. |
| `data/excerpts/` | Public fixtures for **unit tests only** (never displayed on Posit). |
| `artifacts/posit_full_cohort/` | Committed full-cohort mock tables for Connect Cloud renders. |
| `prompts/system_prompt.md` | Documented system prompt (synced with code). |
| `prompts/examples/<tier>/` | Two sample persona prompts per cumulative tier. |

## Data layout (private full cohort)

‼️ **Do not commit Prolific/Qualtrics exports to GitHub.** Place them in a sibling folder:

```text
../sibling_data/PRCAProlificExport_FileA.csv
../sibling_data/PRCAProlificExport_FileB.csv
../sibling_data/PRCAQualtricsExport_FileC.csv
```

```python
pd.read_csv("../sibling_data/PRCAProlificExport_FileA.csv")
```

- **File A + File B** — two Prolific recruitment waves (same columns; stacked; **262** unique IDs).
- **File C** — Qualtrics responses; merge key is typed Prolific ID in `Q0` (**273** rows).
- **Merge coverage:** **252** matched · **21** Qualtrics-only (disregard) · **10** Prolific-only (disregard).
- Public **excerpt fixtures** in `data/excerpts/` remain for unit tests only; Posit Connect displays full-cohort results via staged File A/B/C or `artifacts/posit_full_cohort/`.
- Column labels: [`docs/qualtrics_data_dictionary.csv`](docs/qualtrics_data_dictionary.csv).

See [`data/README.md`](data/README.md).

## Persona / LLM prediction framework

Python package under [`src/ca_personas/`](src/ca_personas/) extracts Prolific + Qualtrics fields, **cleans** to an analytic sample, runs **RQ-aligned EDA**, scores ground-truth PRCA subscales (6–30), builds **tiered / full** persona prompts, calls **Ollama** or **OpenRouter**, and evaluates agents on:

1. **Exact score precision** — MAE + exact integer match on the 6–30 scale  
2. **Band accuracy** — whether predicted low / moderate / high matches the participant  
3. **Distance from correct** — normalized score distance (`|pred−gt| / 24`) and ordinal band distance (0–2 steps; also normalized to 0–1)  

Information tiers map to the research focus: `demos` → `employment` (RQ1) → `geo` → `transit` (RQ2) → `full` (RQ3 / richest personification).

See [`docs/framework.qmd`](docs/framework.qmd) for architecture details.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env   # set Ollama, OpenRouter, or CA_LLM_PROVIDER=mock

# Clean File A/B/C + EDA only (no LLM); requires staged full cohort for research runs
ca-personas prepare --join inner

# Score + aggregate participant ground truth (shared ML/LLM evaluation targets)
ca-personas score-gt --join inner

# Build digital-twin persona prompts (natural-language narrative; research tiers + full)
ca-personas build-personas --tiers demos employment geo transit full

# Offline dry run on the full cohort (deterministic mock model)
ca-personas run --provider mock --join inner

# Local Ollama / OpenRouter
ca-personas run --provider ollama --model llama3.2
ca-personas run --provider openrouter --model meta-llama/llama-3.2-3b-instruct:free

pytest
```

Artifacts land in `data/processed/`, `outputs/eda/`, `outputs/ground_truth/`, `outputs/personas/`, `outputs/predictions/`, and `outputs/evaluation/` (includes `band_acc_*` + `exact_acc_*` in `summary_by_tier.csv`).

## Stage one: ML baselines (full suite)

Before comparing LLMs, establish tabular baselines on the **same** tiered prediction task (predict group / interpersonal CA from demographics → employment → geo → transit). The suite includes **Ridge, Elastic Net, k-NN, Random Forest, HistGradientBoosting, XGBoost, and MLP**:

```bash
pip install -r requirements.txt
pip install -e .
ca-personas ml-baseline --join inner --seed 42
# or
jupyter nbconvert --to notebook --execute notebooks/stage_one_ml_baseline.ipynb --output stage_one_ml_baseline.executed.ipynb
```

Or from Python:

```python
from ca_personas.ml_baseline import run_stage_one_baselines, save_baseline_artifacts
participants, preds, metrics = run_stage_one_baselines(
    "data/excerpts/prolific_excerpt.csv",
    "data/excerpts/qualtrics_excerpt.csv",
)
save_baseline_artifacts(preds, metrics, "outputs/ml_baseline")
```

Metrics land in `outputs/ml_baseline/` (MAE pivots, leaderboard, exact-score / band accuracy) for later comparison to LLM summaries by tier. Write-up: [`docs/ml_baselines.md`](docs/ml_baselines.md).

## ML vs LLM comparison

Evaluate the ML suite against LLM persona agents on the **same** tiers and shared metrics:

```bash
ca-personas compare --provider mock --join inner
# or
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
```

Artifacts write to `outputs/ml_vs_llm/` (`ml_vs_llm_comparison.csv`, deltas, shared evaluation tables).

## Factor analysis & feature importance

Rank the strongest predictive covariates in the sample and inspect PRCA item factor structure:

```bash
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

Artifacts (loadings, permutation/impurity importances, `top_predictive_features.csv`) write to `outputs/feature_importance/`.

### SHAP values, band F1, and ML vs LLM feature power

Dedicated evaluation of which tier features drive CA predictions for Random Forest / KNN **and** LLM persona agents (TreeSHAP + LLM-surrogate SHAP, macro/per-band F1, tier ablation):

```bash
pip install -e ".[dev]"   # includes shap>=0.44
ca-personas shap-eval --join inner --provider mock --shap-tier transit
jupyter nbconvert --to notebook --execute notebooks/feature_predictive_power_shap.ipynb
```

Artifacts write to `outputs/shap_eval/` (metrics, SHAP tables, figures). Research memo: [`memos/feature_predictive_power_ml_llm.md`](memos/feature_predictive_power_ml_llm.md).

## vLLM digital-twin batch inference

For local GPU batch runs, export persona prompts into the `caseid` / `prompt` schema and generate with vLLM (launcher adapted from [`ai_terrarium_v2`](https://github.com/Exios66/ai_terrarium_v2)):

```bash
pip install -e ".[vllm]"
# Defaults prefer ../sibling_data File A/B/C (else excerpts). Pass paths to override.
python -m inference.export_prompts \
    --tiers demos employment geo transit full \
    --output-dir outputs/vllm_prompts
./scripts/run_vllm.sh
python -m inference.ingest_results \
    --result_csv outputs/vllm_results/results.csv \
    --predictions_csv outputs/predictions/vllm_predictions.csv
```

See [`src/inference/README.md`](src/inference/README.md) for checkpoint-resume, GPU flags, and HF token setup. Bug-fix notes from the 2026-07-25 audit live in [`docs/bugfix_audit.md`](docs/bugfix_audit.md).

## Regenerating memo figures

Publication-styled figures (Liberation Sans, shared palette, annotated AUC/prevalence panels) are produced by:

```bash
ca-personas followup-experiments --join inner --seed 42
python scripts/regenerate_memo_figures.py
```

Shared styling lives in [`src/ca_personas/viz_style.py`](src/ca_personas/viz_style.py).

## Quarto manuscript website

The project is a Quarto **website** configured by [`_quarto.yml`](_quarto.yml). The primary manuscript is [`index.qmd`](index.qmd); it runs the offline mock pipeline on the **full matched cohort** at render time (or loads committed `artifacts/posit_full_cohort/` on Connect Cloud) so the site never displays excerpt-fixture statistics.

```bash
# from the root of the repo
pip install -r requirements.txt
quarto check
quarto render                 # builds _site/
quarto preview                # local preview
```

### Publish to Posit Connect Cloud (JackJBurleson)

Canonical deployment: account **`jackjburleson`**, content  
`019f9a10-ebb9-d1d5-839f-97e794bfd0ca`  
(share: https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/).

Publishing **requires full-cohort File A/B/C** (not excerpts). Use the helper (device-code OAuth or `POSIT_CONNECT_CLOUD_*` env vars):

```bash
python scripts/publish_posit_jackjburleson.py
```

Agent runbook: [`.cursor/skills/posit-connect-publish/SKILL.md`](.cursor/skills/posit-connect-publish/SKILL.md).

## Notes

Excerpt fixtures live in `data/excerpts/` (tests only). Full-cohort Posit mock tables live in `artifacts/posit_full_cohort/`. Generated `data/processed/`, `outputs/`, `_site/`, and `_freeze/` are gitignored. Never commit API keys; use `.env` locally.

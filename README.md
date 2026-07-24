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

## Research Question

When an LLM is given a dynamically-constructed, first-person “embodiment” prompt built from an individual respondent’s demographic and behavioral attributes, and instructed to predict that person’s own communication-apprehension (CA) score, how accurately does the model recover the respondent’s true PRCA subscale scores — and where accuracy fails, does the error pattern correlate systematically with demographic group membership (i.e., stereotyping) rather than random noise?

## Research Focus

1. Does employment status improve prediction accuracy over demographics alone, and does it change the bias pattern (e.g., does the model now stereotype “unemployed” respondents as higher-CA, correctly or not)?
2. Does transportation-use data improve prediction accuracy, and does the model use it sensibly (e.g., inferring low transit use → higher avoidance → higher CA) or does it ignore it/misuse it?
3. Does combining both help beyond either alone, or do they carry redundant signal (e.g., employment and transit use may correlate with each other in your sample, so Tier 3 may not beat Tier 1 or 2 individually)?

## Suggested Project Structure + Contents

| Path | What it is |
|---|---|
| `index.qmd` | The primary manuscript. Start here. |
| `contributions.md` | Who owned what. |
| `memos/` | Individual research memos, one per member. |
| `references.bib` | Shared BibTeX file for the manuscript and memos. |
| `data/` | Data used in this project. |

## Persona / LLM prediction framework

Python package under [`src/ca_personas/`](src/ca_personas/) extracts Prolific + Qualtrics fields, scores ground-truth PRCA subscales (6–30), builds **tiered / full** persona prompts, calls **Ollama** or **OpenRouter**, and evaluates agents on:

1. **Exact score precision** — MAE + exact integer match on the 6–30 scale  
2. **Band accuracy** — whether predicted low / moderate / high matches the participant  
3. **Distance from correct** — normalized score distance (`|pred−gt| / 24`) and ordinal band distance (0–2 steps; also normalized to 0–1)  

See [`docs/framework.qmd`](docs/framework.qmd) for architecture details.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set Ollama or OpenRouter credentials

# Score + aggregate participant ground truth (shared ML/LLM evaluation targets)
ca-personas score-gt --join inner

# Build foolproof persona prompts (research tiers + full Qualtrics voice)
ca-personas build-personas --tiers demos employment geo transit full

# Offline dry run (deterministic mock model)
ca-personas run --provider mock --join inner

# Local Ollama / OpenRouter
ca-personas run --provider ollama --model llama3.2
ca-personas run --provider openrouter --model meta-llama/llama-3.2-3b-instruct:free

pytest
```

Artifacts land in `outputs/ground_truth/`, `outputs/personas/`, `outputs/predictions/`, and `outputs/evaluation/` (includes `band_acc_*` + `exact_acc_*` in `summary_by_tier.csv`).

## Stage one: ML baselines (RF + KNN)

Before comparing LLMs, establish tabular baselines on the **same** tiered prediction task (predict group / interpersonal CA from demographics → employment → geo → transit):

```bash
pip install -r requirements.txt
pip install -e .
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

Metrics land in `outputs/ml_baseline/` (MAE, exact-score accuracy, band accuracy, and distance-from-correct) for later comparison to LLM summaries by tier.

<<<<<<< HEAD
## ML vs LLM comparison

Evaluate Random Forest / KNN against LLM persona agents on the **same** tiers and shared metrics:

```bash
ca-personas compare --provider mock --join inner
# or
CA_LLM_PROVIDER=mock jupyter nbconvert --to notebook --execute notebooks/ml_vs_llm_comparison.ipynb
```

Artifacts write to `outputs/ml_vs_llm/` (`ml_vs_llm_comparison.csv`, deltas, shared evaluation tables).
=======
## Factor analysis & feature importance

Rank the strongest predictive covariates in the sample and inspect PRCA item factor structure:

```bash
jupyter nbconvert --to notebook --execute notebooks/factor_feature_importance.ipynb
```

Artifacts (loadings, permutation/impurity importances, `top_predictive_features.csv`) write to `outputs/feature_importance/`.
>>>>>>> origin/main

## Quarto manuscript website

The project is a Quarto **website** configured by [`_quarto.yml`](_quarto.yml). The primary manuscript is [`index.qmd`](index.qmd); it re-runs the excerpt analysis at render time so the site builds on Posit Connect Cloud without live LLM credentials.

```bash
# from the root of the repo
pip install -r requirements.txt
quarto check
quarto render                 # builds _site/
quarto preview                # local preview
```

### Publish to Posit Connect Cloud

1. Push this repository to GitHub (public).
2. In [Posit Connect Cloud](https://connect.posit.cloud), choose **Publish → Quarto**.
3. Select the repository and branch.
4. Set the primary file to **`_quarto.yml`** (recommended website publish) or **`index.qmd`**.
5. Confirm `requirements.txt` is present (Connect installs Jupyter + analysis deps from it).

## Notes

Excerpt fixtures live in `data/excerpts/`. Generated `data/processed/`, `outputs/`, `_site/`, and `_freeze/` are gitignored. Never commit API keys; use `.env` locally.

# Bug-fix audit (2026-07-25)

This note lists defects found in a full-repo audit and the fixes applied on
branch `cursor/fix-bugs-silent-failures-ecf3`. Severity reflects impact on
reproducibility and silent-failure risk.

## Critical / high (fixed)

| # | Issue | Location | Fix |
|---|---|---|---|
| 1 | Config hardcodes `../sibling_data/` paths; `run` / `prepare` never fell back to excerpt fixtures when sibling files were absent | `config/default.yaml`, `pipeline.py` | Resolve config paths only when files exist; otherwise use `default_*_paths()` |
| 2 | Cleaning applied inconsistently (`score-gt`, ML, compare, vLLM export skipped analytic filters) | `load.py`, `ground_truth.py`, `ml_baseline.py`, `feature_importance.py`, `export_prompts.py` | Default `load_and_prepare(..., clean=True)`; all research entry points clean |
| 3 | Batch LLM / vLLM parse failures never failed the job | `predict.py`, `ingest_results.py` | Raise when every row errors (or error rate exceeds threshold) |
| 4 | `--ground_truth_csv` ignored when `answer` already present in prompts | `predict_vllm.py` | Coalesce / fill missing answers; always validate completeness |
| 5 | `export_prompts` defaulted to excerpts and skipped cleaning | `export_prompts.py` | Sibling-data defaults + `clean=True` / `load_full_cohort` |
| 6 | Config `cleaning.*` and unused output-dir keys were dead | `config/default.yaml`, `pipeline.py` | Wire `cleaning` into prepare/run; drop unused dir keys |
| 7 | “Complete CA” counted non-empty strings; unscorable Likert dropped later without audit | `clean.py` | Track `n_dropped_unscorable_ca` after scoring |
| 8 | Model-reported bands preferred even when inconsistent with scores | `evaluate.py`, `llm/base.py` | Validate band labels; resolve band metrics from predicted scores |
| 9 | `int()` truncated float JSON scores silently | `llm/base.py` | Require integral numeric values |
| 10 | Missing config path returned `{}` silently | `pipeline.py` | Raise `FileNotFoundError` |

## Medium (fixed)

| # | Issue | Fix |
|---|---|---|
| 11 | `build_persona_prompts` skipped rows with no log | Emit `warnings.warn` with skip counts |
| 12 | `ml_long_to_eval_format` NaN-truthiness (`or`) and silent incomplete pairs | Explicit null checks; warn on dropped groups |
| 13 | `has_core_demos` was ANY-of instead of ALL-of | Require all present demo fields |
| 14 | Factor analysis used all Qualtrics rows, not analytic sample | Restrict FA matrix to analytic participant IDs |
| 15 | Pipeline loader branch was tautological | Simplify loader selection |
| 16 | Secondary RQ wrapper scripts ignored CLI args | Delegate to `ca_personas.cli.main` |
| 17 | `run_pipeline.py` lacked `sys.path` bootstrap | Match other scripts / CLI delegation |
| 18 | Evaluation left-join had no unmatched warning | Warn when predictions lack ground truth |
| 19 | CA→transit null AUC hardcoded vs geo module | Read prevalence baseline AUC from null table |
| 20 | System prompt markdown drifted from live `SYSTEM_PROMPT` | Resync `prompts/system_prompt.md` |

## Documentation / placeholders (fixed)

| # | Issue | Fix |
|---|---|---|
| 21 | `Contributions.md` still course template (`ORG/REPO`) | Filled for solo author + real repo links |
| 22 | `Getstarted.md` multi-student hour-zero template | Rewritten as project setup for this repo |
| 23 | Excerpts README claimed config consumed excerpt filenames | Document path-resolution + fallback |
| 24 | README tier naming / structure incomplete; `contributions.md` casing | Align tiers, layout, and file casing |
| 25 | Secondary RQ memo vs formal write-up N/AUC drift | Re-ran File A/B/C (`seed=42`); both now report N=241, CA AUC=0.590, geo AUC=0.551 |
| 26 | `index.qmd` missing `pip install -e .`; four-vs-five tier drift | Install step + consistent tier language |
| 27 | `.env.example` omitted `mock` provider | Document dry-run provider |
| 28 | `requirements.txt` / `pyproject.toml` drift | Align packages and comments |

## Intentionally retained patterns

- Per-row `except Exception` in prediction loops still captures individual failures for batch CSV audit columns; the **job** now fails when the batch is unusable.
- HF cache probe in `predict_vllm.py` still catches broad errors while checking local cache (usually offline); it only logs, then continues to Hub load.

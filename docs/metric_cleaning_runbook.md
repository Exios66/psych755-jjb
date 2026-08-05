---
title: "LLM metric-cleaning runbook"
subtitle: "Raw model text → continuous and categorical CA metrics (reproducible pipeline)"
---

**Shared metrics:** [`docs/ml_vs_llm.md`](ml_vs_llm.md)  
**Parsing / ingestion:** `src/inference/ca_prompts.py` + `src/ca_personas/llm/base.py`  
**Scoring key:** `src/ca_personas/scoring.py`  
**Evaluation:** `src/ca_personas/evaluate.py`  
**Braintrust logging:** `src/inference/braintrust_tracing.py`

This runbook documents the exact cleaning and scoring steps that turn raw LLM
text outputs into the continuous and categorical metrics reported on the site.
Every step below maps to real code (no hand-waving): running the listed
functions end-to-end reproduces the N = 1,205 v1 tables (241 participants × 5
tiers) and the band / distance / exact-match metrics.

---

## Phase 1 — Data ingestion & JSON parsing

**Source table.** A vLLM run produces a result CSV with two required columns,
`caseid` (e.g. `P1023__demos`) and `generated_text`.

**CLI entry point:**

```bash
python -m inference.ingest_results \
  --result_csv outputs/vllm_results/results.csv \
  --predictions_csv outputs/predictions/vllm_predictions.csv
```

**What it does** (`src/inference/ca_prompts.py::results_to_predictions`):

1. Reads the result CSV and normalizes `caseid`; `parse_caseid` splits it back
   into `participant_id` + `tier`. Rows with malformed ids are logged as
   `error=ValueError: …` rows and excluded from metrics.
2. For each row, passes `generated_text` to
   `extract_json_object` (`src/ca_personas/llm/base.py:47`):
   - Strips surrounding conversational text with `_normalize_model_text`.
   - If a fenced block exists, keeps the content of the first
     `` ```json … ``` `` fence.
   - Otherwise extracts the text between the **first `{` and the last `}`**.
   - `json.loads` the remaining string.
3. Per-row failures are captured — they never crash the batch:
   `error = "<ExceptionType>: <message>"`. The CLI aborts (non-zero exit) only
   if **zero** rows parsed, so a silent all-fail run cannot masquerade as
   success.

**Parse success is a reported quality gate.** The v1 Llama-3.2-3B-Instruct run
reached 100 % parse; the Llama-3.2-3B **base** run produced malformed JSON and
is excluded from all analyses (see
[`docs/persona_prompt_versions.md`](persona_prompt_versions.md)).

## Phase 2 — Merge with ground truth & validation

**Validation at parse time** (`src/ca_personas/llm/base.py::validate_prediction`):

- Required keys are `self_reported_group_ca` and
  `self_reported_interpersonal_ca`; missing keys raise `ValueError`.
- Scores are coerced with `_as_int_score`: bools and non-integral floats are
  rejected; the value must parse to an integer.
- **Bounds check:** both scores must satisfy `6 ≤ score ≤ 30` (the PRCA
  subscale range). Out-of-range predictions are rejected, not clipped.

**Merge** (`src/ca_personas/evaluate.py::evaluate_predictions`):

- Ground truth is joined to predictions **on `participant_id`** (left join).
- Rows without a matching `gt_group_ca` / `gt_interpersonal_ca` emit a warning
  and are excluded from metric means; all metrics use the matched subset only.
- Ground-truth subscales come from `add_ground_truth_scores`
  (`src/ca_personas/scoring.py:130`): six Likert items per subscale, with
  reverse coding applied — group comfort items `Q2 / Q4 / Q6`, interpersonal
  comfort items `Q14 / Q16 / Q17` (`reverse_score = 6 - item`). Any missing
  item yields `None` for that subscale.

## Phase 3 — Scoring & banding

**Continuous subscales** (`scoring.py::subscale_score`): each side sums six
items on the 6–30 scale.

**Bands** (`scoring.py::ca_band`, `low_max=13`, `high_min=20`):

| Condition | Band |
|---|---|
| `score ≤ 13` | low |
| `14–19` | moderate |
| `score ≥ 20` | high |

The same function derives **both** ground-truth bands and (when a model reports
its own band) resolved prediction bands. Reported bands are normalized to
`low / moderate / high`; whenever a numeric score is present, the band is
**re-derived from the score** so a model's self-reported band can never
silently disagree with its own score (`pred_*_band_resolved`).

**Ordinal encoding** for distance math (`scoring.py::BAND_ORDINAL`):
`low = 0`, `moderate = 1`, `high = 2`.

## Phase 4 — Continuous metrics (row level)

For each side `{group, interpersonal}` (`evaluate.py::evaluate_predictions`):

| Column | Definition |
|---|---|
| `error_<side>` | `pred − gt` signed error (under-prediction is negative) |
| `abs_error_<side>` | `|pred − gt|` |
| `score_distance_<side>` | `abs_error` on the 6–30 scale |
| `norm_score_distance_<side>` | `abs_error / 24`, clipped to `[0, 1]` |
| `exact_match_<side>` | rounded `pred == gt` (integer equality) |

`PRCA_SCORE_RANGE = 30 − 6 = 24` is the normalizer; clipping keeps
out-of-range predictions from producing distances > 1.

## Phase 5 — Categorical / band metrics (row level)

| Column | Definition |
|---|---|
| `band_match_<side>` | resolved pred band == gt band (missing if either is missing) |
| `band_distance_<side>` | ordinal steps `|ord(pred) − ord(gt)| ∈ {0, 1, 2}` (low↔high = 2) |
| `norm_band_distance_<side>` | `band_distance / 2 ∈ [0, 1]` |

Adjacent misses (moderate↔high) are therefore counted as closer than opposite
misses (low↔high).

## Phase 6 — Aggregation across tiers and groups

**By tier / overall** (`evaluate.py::summarize_errors`): MAE, RMSE-style
spread, exact-match accuracy, band accuracy, and mean score/band distances for
the whole set and each `tier`.

**By demographic group** (`evaluate.py::summarize_errors_by_group`): MAE and
band accuracy sliced on Sex / Age / Student / Employment / transit / Q28, the
inputs to the stereotyping Δ-MAE audits
([`docs/stereotyping_evaluation.md`](stereotyping_evaluation.md)).

**Inverse-MAE (higher is better, for prompt ranking)** — computed in
`src/inference/braintrust_tracing.py:266`:

```text
inverse_mae_mean = 1 − min(1, mae_mean / 24)
```

where `mae_mean` averages `abs_error_group` and `abs_error_interpersonal`.
A perfect predictor scores 1.0; chance-level absolute error (≥ 24) scores 0.

## Reproducing the N = 1,205 v1 tables

```bash
python -m inference.ingest_results \
  --result_csv outputs/vllm_results/results.csv \
  --predictions_csv outputs/predictions/vllm_predictions.csv
```

then, inside Python:

```python
from ca_personas.evaluate import evaluate_predictions, summarize_errors
from ca_personas.paths import full_cohort_paths
from ca_personas.load import load_participants

participants, _ = load_participants(*full_cohort_paths(), join_how="inner")
predictions = pd.read_csv("outputs/predictions/vllm_predictions.csv")
evaluation = evaluate_predictions(participants, predictions)
summary = summarize_errors(evaluation)   # MAE / exact / band / distance by tier
```

Expect 1,205 prediction rows (241 × 5 tiers) for the v1 prompt generation, and
a reported parse count matching the run's `n_parsed` in the ingest payload.

## Failure modes & how the pipeline handles them

| Symptom | Handling |
|---|---|
| Conversational text around JSON | `extract_json_object` strips fences / keeps first `{`→last `}` |
| Malformed `caseid` | Logged as `error` row, excluded from metrics |
| Missing / non-integer / out-of-range score | `validate_prediction` raises → logged per row |
| Model-reported band vs score disagreement | Band re-derived from score (`pred_*_band_resolved`) |
| Zero rows parsed | `ingest_results` exits non-zero — no silent empty tables |
| No ground-truth match | Warning + matched-subset-only metrics |

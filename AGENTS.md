# AGENTS.md

Instructions for coding agents working in **psych755-jjb**
([Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)).

Human setup and research context live in `README.md` and `Getstarted.md`.
This file is the agent-oriented runbook: layout, safe defaults, gotchas, and
quality gates.

---

## Project at a glance

| Piece | Role |
|---|---|
| `src/ca_personas/` | Analysis package + `ca-personas` CLI (clean, score, personas, ML, secondary RQs, SHAP) |
| `src/inference/` | vLLM digital-twin export / batch generation / ingest (+ optional Braintrust) |
| `index.qmd` + `_quarto.yml` | Primary Quarto **website** manuscript (the main “app”) |
| `docs/`, `memos/` | Methods write-ups and research memos (also rendered into the site) |
| `notebooks/` | Executable analyses mirroring CLI workflows |
| `config/default.yaml` | Default paths, tiers, LLM + cleaning settings |
| `data/excerpts/` | Public fixtures for **unit tests only** |
| `artifacts/posit_full_cohort/` | Committed full-cohort mock tables for Connect Cloud renders |
| `prompts/` | System prompt + per-tier persona examples (keep in sync with code) |
| `tests/` | `pytest` quality gate (~2 min, no network) |
| `scripts/` | Publish helpers, figure regen, Posit artifact sync |
| `.cursor/skills/posit-connect-publish/` | Agent skill for full-cohort → render → Posit publish |

**Primary research:** tiered LLM persona prompts predicting PRCA communication-apprehension scores (exact / band / distance metrics), with stereotyping analysis across demographic groups.

**Secondary research:** observational RQs on the matched Prolific↔Qualtrics cohort (transit↔CA, geo/CA/mobility → transit RFs) — no LLM required.

---

## Hard rules

1. **Never commit private Prolific/Qualtrics exports** (File A/B/C), API keys, or `.env`.
2. Prefer **`CA_LLM_PROVIDER=mock`** for local/cloud dry runs; do not require live Ollama/OpenRouter unless the user asks.
3. Do **not** publish excerpt-only results to Posit Connect unless the user explicitly overrides.
4. Keep manuscript numbers tied to seeded full-cohort runs (`--seed 42`, `--join inner`) or committed `artifacts/posit_full_cohort/` tables.
5. Focused diffs only — no drive-by refactors or unsolicited markdown.

---

## Environment

### Python

```bash
source .venv/bin/activate          # create if missing: python3.11 -m venv .venv
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"            # registers `ca-personas`; includes pytest
# optional GPU batch:
# pip install -e ".[vllm]"
```

- Requires **Python ≥ 3.11**.
- Package config: `pyproject.toml` (`ca-personas` console script → `ca_personas.cli:main`).
- `pyrightconfig.json` exists; pyright is **not** a declared dependency. No standalone formatter/linter is configured.

### Quarto

Quarto CLI is a **system** install (not pip). Needed for `quarto render` / `preview`.
`R` is **not** required — the site uses the Python/Jupyter engine.

If `quarto` is missing on a Cursor Cloud VM, install the `.deb` from
[quarto-cli releases](https://github.com/quarto-dev/quarto-cli/releases).
It is intentionally **not** in the update script (persisted via snapshot).

### Cursor Cloud apt clock skew

`apt` may fail with “Release file … is not valid yet”. Work around with:

```bash
sudo apt-get -o Acquire::Check-Valid-Until=false update
```

### `.env` (required for Quarto)

`quarto render` / `preview` use dotenv-safe: they need a `.env` file **and**
those variables in the process environment. An **empty** `OPENROUTER_API_KEY`
is rejected.

```bash
cp .env.example .env
sed -i 's/^CA_LLM_PROVIDER=ollama/CA_LLM_PROVIDER=mock/' .env
sed -i 's/^OPENROUTER_API_KEY=$/OPENROUTER_API_KEY=unused-mock/' .env
set -a; . ./.env; set +a
```

Key vars (see `.env.example`): `CA_LLM_PROVIDER` (`ollama` \| `openrouter` \| `mock`),
`OLLAMA_*`, `OPENROUTER_*`.

---

## Data

| Location | Purpose |
|---|---|
| `../sibling_data/PRCAProlificExport_FileA.csv` (+ FileB, FileC) | Private full cohort (preferred) |
| `/tmp/sibling_data/` (same filenames) | Cloud-agent staging; or set `CA_SIBLING_DATA` |
| `data/excerpts/*.csv` | Public unit-test fixtures only |
| `artifacts/posit_full_cohort/` | Committed mock tables when private data is absent |

**Merge coverage targets:** ~252 matched · 21 Qualtrics-only · 10 Prolific-only.
Analytic PRCA sample for secondary RQs is typically **n ≈ 241** (seed=42, inner join).

CLI commands auto-prefer sibling data when present; otherwise fall back to excerpts
(or Posit artifacts for site renders). Detect via `ca_personas.paths.sibling_data_available()`.

---

## Common commands

Activate the venv first (`source .venv/bin/activate`).

### Quality gate

```bash
pytest                          # primary gate; offline, ~2 min
```

### Offline pipeline smoke

```bash
ca-personas run --provider mock --join inner
# writes data/processed/ + outputs/{eda,ground_truth,personas,predictions,evaluation}/
```

### Prepare / score / personas (no live LLM)

```bash
ca-personas prepare --join inner
ca-personas score-gt --join inner
ca-personas build-personas --tiers demos employment geo transit full
```

### ML + comparison

```bash
ca-personas ml-baseline --join inner --seed 42
ca-personas compare --provider mock --join inner
ca-personas shap-eval --join inner --provider mock --shap-tier transit
```

### Secondary observational RQs

```bash
ca-personas transit-ca --join inner --seed 42
ca-personas geo-transit-rf --join inner --seed 42
ca-personas ca-transit-rf --join inner --seed 42
ca-personas covariate-transit-rf --join inner --seed 42
ca-personas comprehensive-transit-rf --join inner --seed 42
ca-personas followup-experiments --join inner --seed 42
```

Full-cohort sanity AUCs (seed=42, inner join) used in publish verification:

- Geo → transit RF ≈ **0.551**
- CA → transit RF ≈ **0.590**
- Q28 → transit RF ≈ **0.762**
- Q27 → transit RF ≈ **0.589**

### Quarto site

```bash
set -a; . ./.env; set +a
quarto check
quarto render                   # writes _site/
# preview: quarto preview
# static view: python3 -m http.server -d _site <port>
```

Site cells use the **mock** LLM on the full matched cohort (or load
`artifacts/posit_full_cohort/` when File A/B/C are absent).

### Posit Connect Cloud (JackJBurleson)

Canonical content: `019f9a10-ebb9-d1d5-839f-97e794bfd0ca`  
Share: https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/

**Requires private File A/B/C + credentials.** Follow
[`.cursor/skills/posit-connect-publish/SKILL.md`](.cursor/skills/posit-connect-publish/SKILL.md).

```bash
python scripts/publish_posit_jackjburleson.py
# python scripts/publish_posit_jackjburleson.py --skip-analysis
```

Do **not** publish to older `jjb-morningstar` content unless explicitly requested.

### Posit mock artifact refresh (after persona/prompt changes)

```bash
python scripts/sync_posit_full_cohort_artifacts.py
python scripts/rerun_posit_mock_from_participants.py
```

### vLLM digital-twin (GPU; optional)

```bash
pip install -e ".[vllm]"          # includes braintrust optional dep
python -m inference.export_prompts --tiers demos employment geo transit full \
  --output-dir outputs/vllm_prompts
./scripts/run_vllm.sh
python -m inference.ingest_results \
  --result_csv outputs/vllm_results/results.csv \
  --predictions_csv outputs/predictions/vllm_predictions.csv
```

**Braintrust (opt-in):** set `BRAINTRUST_API_KEY` so each vLLM run logs parse /
exact / band / inverse-MAE scores into project `psych755-ca-personas`. Push /
iterate the system prompt via `bt functions push prompts/braintrust_ca_system.py`
(slug `ca-digital-twin-system`). Post-hoc: `python -m inference.braintrust_log_results`.
See `src/inference/README.md`. Never commit the API key.

### Memo / APA figures

```bash
ca-personas followup-experiments --join inner --seed 42
python scripts/regenerate_memo_figures.py
# python scripts/regenerate_apa_site_figures.py
```

Shared styling: `src/ca_personas/viz_style.py`, `src/ca_personas/apa_plotting.py`.

---

## CLI surface (`ca-personas`)

| Command | Purpose |
|---|---|
| `run` | Full pipeline: clean → EDA → GT → personas → LLM → evaluate |
| `prepare` | Load/clean/score GT + EDA (no LLM) |
| `score-gt` | Ground-truth PRCA subscale artifacts |
| `build-personas` | Tiered persona prompt bundles |
| `ml-baseline` | Tabular ML suite on tiered CA prediction |
| `compare` | ML vs LLM shared metrics |
| `shap-eval` | SHAP / band F1 / tier ablation |
| `stereotype-eval` | Stereotyping MAE gaps + association tests (Sex/Age/Student/Employment/transit/Q28) |
| `transit-ca` | Secondary: regular transit vs CA |
| `geo-transit-rf` | Secondary: lat/long → transit RF |
| `ca-transit-rf` | Secondary: group/interpersonal CA → transit RF |
| `covariate-transit-rf` | Mobility / Q27 / Q28 covariate RFs |
| `comprehensive-transit-rf` | Broader transit predictor RF suite |
| `followup-experiments` | Wave-2 follow-up experiment battery |

Shared flags typically include `--join {inner,outer,left}`, optional
`--prolific` / `--qualtrics` paths, and `--seed` on RF commands.
Persona tiers: `demos` → `employment` → `geo` → `transit` → `full`.

---

## Where to edit what

| Change | Touch |
|---|---|
| Cleaning / merge / scoring | `src/ca_personas/load.py`, `clean.py`, `scoring.py`, `ground_truth.py`, `paths.py` |
| Persona prompts | `src/ca_personas/personas.py`, `prompts/system_prompt.md`, `prompts/examples/` |
| LLM clients | `src/ca_personas/llm/` (`mock`, `ollama`, `openrouter`) |
| Evaluation metrics | `src/ca_personas/evaluate.py` |
| Secondary RQs | matching `*_rf.py` / `transit_ca.py` / `followup_experiments.py` + `docs/` + `memos/` + notebooks |
| Site nav / pages | `_quarto.yml` (`project.render`, navbar/sidebar) |
| Manuscript body | `index.qmd` |
| Authorship record | `Contributions.md` (permalinks pinned with `y` on GitHub) |

After adding a new Quarto page, register it in `_quarto.yml` `project.render`
and navbar/sidebar when user-facing.

---

## Git / PR conventions

- Branch from `main`; cloud agents use `cursor/<descriptive-name>-<suffix>`.
- Run `pytest` before opening/updating a PR.
- Use the PR template under `.github/PULL_REQUEST_TEMPLATE.md`.
- Issue forms live under `.github/ISSUE_TEMPLATE/` (bug, feature, docs).
- Keep private data and secrets out of commits (see `.gitignore`).

---

## Cursor Cloud notes

- Startup update script refreshes `.venv`; Quarto is snapshot-persisted, not
  reinstalled by the update script.
- Posit publishing and private-data full-cohort re-runs usually **cannot** complete
  in the default sandbox (no sibling exports / Connect credentials). Offline mock
  + `pytest` + `quarto render` (with mock `.env`) are the reliable local gates.
- Prefer the posit-connect-publish skill when the user asks to deploy/refresh the
  live JackJBurleson site and credentials/data are available.

---

## Quick checklist (typical agent task)

1. `source .venv/bin/activate` (install `-e ".[dev]"` if needed).
2. Ensure `.env` with `CA_LLM_PROVIDER=mock` and non-empty stub `OPENROUTER_API_KEY` for Quarto.
3. Make focused code/docs changes.
4. `pytest` (+ relevant `ca-personas … --provider mock` smoke if CLI/pipeline touched).
5. `quarto render` if `index.qmd`, `docs/`, `memos/`, or `_quarto.yml` changed.
6. Commit with a descriptive message; do not commit `outputs/`, `_site/`, or private CSVs.

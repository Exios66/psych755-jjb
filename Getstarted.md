# Get started — psych755-jjb

Project setup for the PSYCH 755 CA persona / PRCA framework
([github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)).

## 1. Clone and create a virtualenv

```bash
git clone https://github.com/Exios66/psych755-jjb.git
cd psych755-jjb
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

`requirements.txt` covers analysis + Quarto/Posit Connect. Editable install
registers the `ca-personas` console script. For local GPU batch inference:

```bash
pip install -e ".[vllm]"
```

## 2. Private full-cohort data (optional)

Do **not** commit Prolific/Qualtrics exports. Place them beside the repo:

```text
../sibling_data/PRCAProlificExport_FileA.csv
../sibling_data/PRCAProlificExport_FileB.csv
../sibling_data/PRCAQualtricsExport_FileC.csv
```

When those files are present, CLI commands prefer them. Cloud sandboxes may
stage the same filenames under `/tmp/sibling_data` or `CA_SIBLING_DATA`.
Excerpt fixtures in `data/excerpts/` are for unit tests only; the Posit site
loads committed full-cohort mock artifacts when private exports are absent.

## 3. Environment variables

```bash
cp .env.example .env
```

Set `CA_LLM_PROVIDER` to `ollama`, `openrouter`, or `mock` (offline dry runs).
Fill Ollama / OpenRouter credentials only when calling a live model.

## 4. Smoke-test the pipeline

```bash
# Offline mock run on excerpts (or sibling data if present)
ca-personas run --provider mock --join inner

# Clean + EDA only
ca-personas prepare --join inner

# Ground-truth scoring + persona export
# (demos tier base layer = Age, Sex, Country of residence, Student status)
ca-personas score-gt --join inner
ca-personas build-personas --tiers demos employment geo transit full

# Refresh committed Posit mock tables after persona/prompt changes
# (uses sibling File A/B/C when present; otherwise regenerate from
# artifacts/posit_full_cohort/participants.csv):
#   python scripts/sync_posit_full_cohort_artifacts.py
#   python scripts/rerun_posit_mock_from_participants.py

pytest
```

Secondary observational RQs (require sibling data for the full cohort):

```bash
ca-personas transit-ca --join inner
ca-personas geo-transit-rf --join inner
ca-personas ca-transit-rf --join inner
```

## 5. Quarto manuscript

```bash
quarto check
quarto render                 # builds _site/
quarto preview
```

Primary manuscript: [`index.qmd`](index.qmd). Site config: [`_quarto.yml`](_quarto.yml).

### Publish to JackJBurleson Posit Connect Cloud

Full-cohort File A/B/C must be present (`../sibling_data/` or `/tmp/sibling_data/`). Agent workflow is documented in [`.cursor/skills/posit-connect-publish/SKILL.md`](.cursor/skills/posit-connect-publish/SKILL.md).

```bash
# Re-run full-cohort analyses, render, device-auth (or env tokens), publish, verify
python scripts/publish_posit_jackjburleson.py

# Or if analyses/render already done:
python scripts/publish_posit_jackjburleson.py --skip-analysis
```

Public share URL: https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/

## 6. Project layout

| Path | Role |
|---|---|
| `src/ca_personas/` | Cleaning, scoring, personas, ML baselines, secondary RQs, CLI |
| `src/inference/` | vLLM digital-twin export / batch generation / ingest |
| `notebooks/` | Executable analyses (ML, FA, secondary RQs, ML vs LLM) |
| `docs/` | Framework notes, merge audit, secondary RQ write-up, bugfix audit |
| `config/default.yaml` | Default paths, tiers, LLM + cleaning settings |
| `data/excerpts/` | Public fixtures for unit tests only (never shown on Connect) |
| `memos/` | Research memos |
| `Contributions.md` | Authorship / ownership record |
| `prompts/system_prompt.md` | Documented system prompt (kept in sync with code) |

## 7. Contribution workflow

1. Create a feature branch from `main`.
2. Make focused commits; keep private data out of git.
3. Run `pytest` before opening a PR.
4. Update `Contributions.md` when you own a new component.
5. Prefer permalinks (press `y` on GitHub) when citing line ranges.

See [`README.md`](README.md) for the research questions and full command reference.

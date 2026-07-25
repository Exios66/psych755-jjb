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

When those files are present, CLI commands prefer them. When absent,
`ca_personas.paths` (and config path resolution) fall back to public fixtures in
`data/excerpts/` so tests and Posit Connect Cloud renders still work.

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
ca-personas score-gt --join inner
ca-personas build-personas --tiers demos employment geo transit full

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

## 6. Project layout

| Path | Role |
|---|---|
| `src/ca_personas/` | Cleaning, scoring, personas, ML baselines, secondary RQs, CLI |
| `src/inference/` | vLLM digital-twin export / batch generation / ingest |
| `notebooks/` | Executable analyses (ML, FA, secondary RQs, ML vs LLM) |
| `docs/` | Framework notes, merge audit, secondary RQ write-up, bugfix audit |
| `config/default.yaml` | Default paths, tiers, LLM + cleaning settings |
| `data/excerpts/` | Public fixtures for tests / Connect Cloud |
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

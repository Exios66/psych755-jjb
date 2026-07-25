# AGENTS.md

## Cursor Cloud specific instructions

This is a Python data-science project (`ca-personas` package + CLI) plus a Quarto
manuscript **website** — that website is the primary "application". Standard
commands live in `README.md` and `Getstarted.md`; only the non-obvious caveats
are captured here.

### Environment

- Python deps live in a virtualenv at `.venv`. Activate it before any command:
  `source .venv/bin/activate`. The startup update script recreates/refreshes it.
- **Quarto CLI** (a system-level install, not a pip package) is required to build
  the site. It is installed into the VM (persisted via the snapshot), so it is
  intentionally NOT in the update script. If `quarto` is ever missing, reinstall
  the `.deb` from the quarto-cli GitHub releases. `R` is not installed and not
  needed — the site uses the Python/Jupyter engine.
- `apt` on this VM can fail with "Release file ... is not valid yet" due to clock
  skew. Work around it with `-o Acquire::Check-Valid-Until=false` on
  `apt-get update`/`install`.

### Running / rendering (key gotcha)

- `quarto render` / `quarto preview` use dotenv-safe: they require a `.env` file
  (gitignored) AND the variables to be present in the process environment, and
  they reject an **empty** `OPENROUTER_API_KEY`. Set up once and export before
  rendering:
  ```bash
  cp .env.example .env
  sed -i 's/^CA_LLM_PROVIDER=ollama/CA_LLM_PROVIDER=mock/' .env
  sed -i 's/^OPENROUTER_API_KEY=$/OPENROUTER_API_KEY=unused-mock/' .env  # mock; not a real key
  set -a; . ./.env; set +a
  quarto render         # writes _site/
  ```
  The site runs the offline **mock** LLM pipeline at render time, so no live LLM
  is needed. To view the built site: `python3 -m http.server -d _site <port>`.

### Data

- Full-cohort Prolific/Qualtrics exports are **private** and not in the repo
  (expected under `../sibling_data/` or `/tmp/sibling_data`). When absent, the
  CLI and site fall back to the public fixtures in `data/excerpts/` and the
  committed mock tables in `artifacts/posit_full_cohort/`. Posit Connect
  publishing (`scripts/publish_posit_jackjburleson.py`) requires the private
  data and external credentials, so it cannot run here.

### Test / CLI

- Tests: `pytest` (the project's quality gate; ~2 min, no network). No standalone
  linter/formatter is configured; `pyrightconfig.json` exists but pyright is not
  a declared dependency.
- Offline smoke test of the whole pipeline: `ca-personas run --provider mock --join inner`
  (writes to `data/processed/` and `outputs/`).

# Contributions

**Group:** Jack J. Burleson (solo)  
**Group Members:** Jack J. Burleson (`@Exios66` / `@jjburleson`)  
**Project:** Tiered Persona Prompting for Communication Apprehension: A Digital-Twin Evaluation Against Classical Machine-Learning Baselines (PSYCH 755)  
**Repository:** [https://github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## Student 1: Jack J. Burleson (`Exios66`)

- **The component I "owned" and that I summarize here is best described as** the
  end-to-end CA persona research framework: Prolific↔Qualtrics loading and
  cleaning, PRCA ground-truth scoring, tiered / full persona prompts, Ollama /
  OpenRouter LLM clients, evaluation (exact score, band, distance), ML
  baselines, ML-vs-LLM comparison, vLLM digital-twin inference bridge, and the
  secondary observational RQs (transit↔CA, geo→transit RF, CA→transit RF), plus
  the Quarto manuscript site.
- **You can find this contribution in** `src/ca_personas/` (package + CLI),
  `src/inference/` (vLLM export/ingest), `notebooks/`, `docs/`, `index.qmd`,
  `config/default.yaml`, and `scripts/`. Example entry points:
  - [Permalink to `cli.py`](https://github.com/Exios66/psych755-jjb/blob/db25cff56d6c2fdc931b81d0c21ef5aa83285334/src/ca_personas/cli.py)
  - [Permalink to `load.py`](https://github.com/Exios66/psych755-jjb/blob/db25cff56d6c2fdc931b81d0c21ef5aa83285334/src/ca_personas/load.py)
  - [Permalink to `index.qmd`](https://github.com/Exios66/psych755-jjb/blob/db25cff56d6c2fdc931b81d0c21ef5aa83285334/index.qmd)
- **Owning this component means** designing the cumulative persona tiers mapped
  to the research questions; implementing File A/B/C merge coverage (252 / 21 /
  10); writing the cleaning + PRCA scoring path; building secondary RQ analyses
  and memos; wiring Quarto / Posit Connect Cloud reproduction on full-cohort artifacts;
  and maintaining tests under `tests/`.
- **The commits or PRs that are most relevant are**
  [#4 — tiered CA persona framework](https://github.com/Exios66/psych755-jjb/pull/4),
  [#15 — File A/B/C sibling data + cleaning](https://github.com/Exios66/psych755-jjb/pull/15),
  [#18 — CA→transit Random Forest](https://github.com/Exios66/psych755-jjb/pull/18),
  [#19 — File C flat-header + merge audit](https://github.com/Exios66/psych755-jjb/pull/19).
- **The portion of the data science process that this effort contributes to is**
  stages spanning **data acquisition and ingestion**, **cleaning and
  validation**, **exploratory analysis**, **modeling**, **evaluation**, and
  **visualization and communication** — with the primary emphasis on measurement
  design (persona tiers vs fixed PRCA ground truth) and reproducible evaluation.
- **Caveat — figure software.** All manuscript, memo, and site figures are
  rendered with **seaborn**, which sits on **matplotlib** [@hunter2007;
  @waskom2021], through the shared APA-styling layer in
  `src/ca_personas/apa_plotting.py` and `src/ca_personas/viz_style.py`. This
  choice was made to align closely with APA 7 figure conventions (sans-serif
  8–14 pt labels, no chartjunk, print-safe grayscale/hatching, captions outside
  the image, 300 dpi exports). matplotlib is a general-purpose plotting backend
  rather than a native APA typesetting tool, so the figures approximate — not
  certify — full APA 7 compliance; any residual deviations are the author's
  responsibility.
- **Recent work (2026-08-05 → 2026-08-06).** Since the last Contributions
  update, I completed 18 commits focused on: **(a)** the seaborn/matplotlib
  APA-aligned visualization layer and regenerated vLLM-cohort, memo, and site
  figures (`7cf0a4d`, `c5ea431`, `a91221e`, `b6ba560`, `37ce617`); **(b)** site
  presentation — multi-theme `styles.scss`, pinned page chrome + active TOC
  styling, the interactive PR progress graph with issues and auto-sync
  pre-render hook, a new `publish.sh`, and removal of the GitHub Pages workflow
  (`f9af8eb`, `7009dda`, `ffa760f`, `31c115d`, `db25cff`); and **(c)** manuscript
  and docs alignment — primary RQs matched to the synthesized tier mapping,
  standardized memo formatting, BibTeX updates, and broken `.md` → `.qmd` link
  fixes (`e7963d2`, `3c6b426`, `e53ed1e`, `0e19856`).

### Owned components (summary)

| Component | Where it lives | What "owned" meant | Key commits/PRs | Data Science Process + Notes |
|---|---|---|---|---|
| Prolific↔Qualtrics loading + merge coverage (252 / 21 / 10) | `src/ca_personas/load.py`, `clean.py`, `paths.py` | Designed File A/B/C stacking/join on `Q0`/Prolific ID; analytic sample `n = 241` | [#15](https://github.com/Exios66/psych755-jjb/pull/15), [#19](https://github.com/Exios66/psych755-jjb/pull/19) | Data acquisition, cleaning & validation |
| PRCA ground-truth scoring (group / interpersonal, 6–30) | `src/ca_personas/ground_truth.py`, `scoring.py` | Wrote subscale scoring + item-completeness rules | [#15](https://github.com/Exios66/psych755-jjb/pull/15) | Measurement design / EDA |
| Tiered persona prompt bundles (demos → full) | `src/ca_personas/personas.py`, `prompts/` | Designed cumulative info-tier prompts mapped to RQ1–RQ3 | [#4](https://github.com/Exios66/psych755-jjb/pull/4) | Measurement design / modeling input |
| LLM clients + evaluation (exact / band / distance) | `src/ca_personas/llm/`, `evaluate.py`, `predict.py` | Ollama / OpenRouter / mock providers; shared metrics | [#4](https://github.com/Exios66/psych755-jjb/pull/4) | Modeling & evaluation |
| ML baseline suite (Ridge, EN, k-NN, RF, HGB, XGBoost, **MLP**) + comparison + SHAP | `ml_baseline.py`, `compare_agents.py`, `shap_eval.py` | Cross-validated tabular suite; ML-vs-LLM; band F1 / tier ablation | [#4](https://github.com/Exios66/psych755-jjb/pull/4) | Modeling / evaluation (trained neural net = optional tool #2) |
| Secondary observational RQs (transit↔CA, geo→transit, CA→transit, covariate suites, TF1/TF2) | `transit_ca.py`, `geo_transit_rf.py`, `ca_transit_rf.py`, `transit_covariate_rf.py`, `followup_experiments.py`, `transit_focus.py` + `memos/` | Built RF analyses, follow-up batteries, and research memos | [#18](https://github.com/Exios66/psych755-jjb/pull/18), [#57](https://github.com/Exios66/psych755-jjb/pull/57) | Analysis + research communication |
| vLLM digital-twin export / ingest (+ Braintrust) | `src/inference/` | Batch persona export, GPU generation, result ingestion, tracing | [#55](https://github.com/Exios66/psych755-jjb/pull/55), [#56](https://github.com/Exios66/psych755-jjb/pull/56) | Modeling (off-the-shelf LLMs = optional tool #1) |
| APA-aligned figure layer (seaborn / matplotlib) + site | `src/ca_personas/apa_plotting.py`, `viz_style.py`, `index.qmd`, `_quarto.yml`, `scripts/` | Shared APA-7 styling; Quarto site + Posit Connect Cloud reproduction | `a91221e`, `b6ba560`, `31c115d`, `db25cff` | Visualization & communication |

---

## Group sign-off

By adding your name below, each member affirms that the account of their own
contribution is accurate.

- [x] Jack J. Burleson (`Exios66`) — 2026-08-05
- [x] Jack J. Burleson (`jjburleson`) — 2026-08-05

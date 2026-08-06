# Contributions

**Group:** Jack J. Burleson (solo)  
**Group Members:** Jack J. Burleson (`@Exios66` / `@jjburleson`)  
**Project:** Tiered Persona Prompting for Communication Apprehension: A Digital-Twin Evaluation Against Classical Machine-Learning Baselines (PSYCH 755)  
**Repository:** [https://github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## How to use this file

Each member completes one section below. Claims must be checkable via file paths
and permanent GitHub links (permalinks pinned to a commit SHA).

> **Markdown links.** Write `[link text](https://example.com)` with no space
> between the brackets and parentheses.

### Permalinks

Open a file on GitHub → select lines → press **`y`** to pin the URL to a commit.

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
  - [Permalink to `cli.py`](https://github.com/Exios66/psych755-jjb/blob/35c323f8076ba10abab931f9ec313a7fc0665f9b/src/ca_personas/cli.py)
  - [Permalink to `load.py`](https://github.com/Exios66/psych755-jjb/blob/35c323f8076ba10abab931f9ec313a7fc0665f9b/src/ca_personas/load.py)
  - [Permalink to `index.qmd`](https://github.com/Exios66/psych755-jjb/blob/35c323f8076ba10abab931f9ec313a7fc0665f9b/index.qmd)
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

---

## Group sign-off

By adding your name below, each member affirms that the account of their own
contribution is accurate.

- [x] Jack J. Burleson (`Exios66`) — 2026-08-05
- [x] Jack J. Burleson (`jjburleson`) — 2026-08-05

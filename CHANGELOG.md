# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Covers development on `main` since `0.3.0` (2026-07-27).

### Added

- Real full-cohort vLLM v2/v3 exports committed under `exports/v2/` and `exports/v3/` (signal-first v2 packaging + 8-tier greedy v3 ablations), with pooled evaluation memo `memos/vllm_v2_v3_evaluation.qmd`
- Live-LLM stereotyping slices memo (`memos/live_llm_stereotyping_slices.qmd`) from the committed DeepSeek v2 export
- Live-export TreeSHAP / surrogate-SHAP feature-power memo (`memos/feature_predictive_power_ml_llm.qmd`)
- Ground-truth group-CA descriptive table by demographic slice in the manuscript stereotyping section
- Cross-memo scaffolding links: README index now covers all 27 memos, orphan memos link siblings, wave-1 parents link forward to wave-2 follow-ups, `.md`/`.qmd` link mismatches fixed
- LLM metric-cleaning runbook (`docs/metric_cleaning_runbook.md`) documenting the exact parse → validate → score → band → metric → aggregate pipeline (#63, #64)

### Changed

- Retitle manuscript to "Tiered Persona Prompting for Communication Apprehension: A Digital-Twin Evaluation Against Classical Machine-Learning Baselines"; align site title (`_quarto.yml`), Contributions record, and publish-script expected-title check
- Replace mock-LLM diagnostics in the manuscript with real ML-vs-vLLM results and committed vLLM export tables
- Discussion section: link every RQ/S/TF answer to its memo scaffold and add a Conclusion
- Bibliography: add verified references (Daly 1978; Binz & Schulz 2024; Grossmann et al. 2023; Horton et al. 2023; Salewski et al. 2023; Sclar et al. 2024), fix park2024 title/authors, polish DOI/URL fields
- Sync `pyproject.toml` package version to the changelog's current release (`0.3.0`) so the two stop disagreeing (#65)

### Fixed

- Broken `memos/README.md` links (`.md` → `.qmd`) and README.md memo-link targets
- Relative `exports/…` links in `docs/persona_prompt_versions.md` (missing `../`)
- Stale "mock at render" and "GPU deferred" claims across README, AGENTS.md, docs, and memos
- Remove committed Quarto-render library dir `memos/live_llm_stereotyping_slices_files/` from git and ignore `memos/*_files/`
- Rerun the seeded full-cohort ML baseline so `docs/figures/ml_baseline_*` show the full seven-model suite instead of the stale two-model artifacts (#66)
- Render the mermaid diagram in `docs/persona_prompt_versions.md` (plain fences emit no mermaid JS in `.md` pages; the page now loads the mermaid library and wraps the diagram in a `.mermaid` div) (#60)
- Regenerate follow-up experiment figures from fresh `outputs/followup_experiments/` so the memo charts are no longer stale (#67)
- Regenerate the Q27/Q28 prevalence chart with the corrected level ordering so bars display low→high (#68)

### Removed

- Redundant CLI-wrapper scripts (`scripts/run_pipeline.py`, `prepare_full_cohort.py`, `score_ground_truth.py`, `build_personas.py`, `run_ca_transit_rf.py`, `run_geo_transit_rf.py`, `run_transit_ca.py`, `posit_publish.py`); `_publish.yml` now points to `publish_posit_jackjburleson.py`
- Dead `cleaning.join_how` key from `config/default.yaml`

## [0.3.0] - 2026-07-27

Covers development on `main` from 2026-07-26 through 2026-07-27 (since `ac56b60` / #44).

### Added

- Persona prompt efficiency guide and v3 public-transit tier examples with updated sample prompts (`docs/persona_prompt_efficiency.md`, `526f9db`, `bd4f123`)
- Per-model vLLM baseline docs and cross-model comparison memos (DeepSeek R1, Llama 3.1/3.2/3.3) with site navigation (`310cab7`, `af270df`)
- Immersive PR Progress Graph page on the Quarto site with offline bake via `scripts/sync_pr_graph_data.py` (#47)
- APA citation style file (`apa.csl`) for manuscript references (`c76d8ad`)

### Changed

- Promote prompt-v1 multi-model vLLM digital-twin findings into core manuscript Abstract, Methods, Results, and Discussion (#46)
- Refine persona transit/geo sentence generation and align tier examples with approximate coordinates (`bd4f123`)
- Expand Posit publish configuration and script for LLM baseline artifact paths (`1956be7`)

### Fixed

- Remove duplicate `[tool.pyright]` block in `pyproject.toml` that blocked `pytest` (#47)

## [0.2.0] - 2026-07-26

First changelog entry covering development on `main` from 2026-07-19 through 2026-07-26.

### Added

- Tiered CA persona extraction and LLM prediction framework with the `ca-personas` CLI (#4)
- Quarto manuscript website for Posit Connect Cloud deployment (#6)
- Stage-one Random Forest and k-NN CA prediction baselines (#7)
- Band accuracy metrics, ground-truth scoring script, and full persona prompt bundles (#10)
- Distance-from-correct evaluation metrics for CA subscale prediction (#11)
- Factor analysis and feature-importance notebook (#12)
- ML vs LLM shared-metric comparison notebook (#13)
- vLLM digital-twin inference launcher for batch persona generation (#14)
- Secondary RQ: regular public transit vs cohort CA scores (`ca-personas transit-ca`) (#16)
- Secondary RQ: Random Forest of lat/long predicting regular transit (`ca-personas geo-transit-rf`) (#17)
- Secondary RQ5: Random Forest of CA scores predicting regular transit (`ca-personas ca-transit-rf`) (#18)
- Research memo and CLI for CA scores predicting regular transit use (#20)
- Comprehensive feature-importance RF suite and research memo for regular transit (#23)
- SHAP/F1 feature predictive-power suite for ML and LLM agents (`ca-personas shap-eval`) (#26)
- Geo-memo follow-up RFs: car access, employment, and ride-share predicting regular transit (#27)
- Manuscript and memo for Q27/Q28 traditional ML predictors of regular transit (#28)
- Full-cohort Posit stats rerun and APA-style site figures (#29)
- Wave-2 follow-up experiments, research memos, and Quarto site pages (`ca-personas followup-experiments`) (#30)
- Student status incorporated into the base demos persona tier end-to-end (#31)
- JackJBurleson Posit Connect publish skill (full-data gated) (#32)
- Two sample persona prompts per cumulative context tier (#33)
- Expanded landing abstract, secondary RQ presentation, and GitHub access guidance (#34)
- AI Terrarium natural-language digital-twin persona prompt rework (#35)
- Seven-model CA ML baseline suite: Ridge, Elastic Net, k-NN, RF, HistGradientBoosting, XGBoost, and MLP (#36)
- AI Terrarium persona examples surfaced on the landing page (#38)
- `AGENTS.md` agent runbook with Cursor Cloud dev environment setup notes (#42)
- GitHub PR and issue templates; expanded `AGENTS.md` contributor guidance (#43)

### Changed

- Align data pipeline to File A/B/C sibling exports with RQ-focused EDA (#15)
- Refresh ML suite baseline figures after full-cohort republish (#37)
- Clean Quarto site navigation: separate Memos from Secondary RQs (#39)
- Harden cohort loading and sync secondary Posit full-cohort artifacts (#44)
- Audit Posit site copy: update stale ML claims, tighten prose, and fix path references (#41)

### Fixed

- File C flat-header detection; enforce 252/21/10 merge coverage audit (#19)
- Silent failures, path fallbacks, placeholders, and doc/requirements drift (#24)
- `comprehensive-transit-rf` CLI regression and full-cohort republish (#40)

### Notes

- Posit Connect publishing requires private File A/B/C exports and credentials; see `.cursor/skills/posit-connect-publish/`
- Manuscript and artifact numbers are tied to `--seed 42` and `--join inner` on the full matched cohort

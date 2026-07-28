# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

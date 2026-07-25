#!/usr/bin/env python3
"""Rerun every Posit-facing evaluation on the full File A/B/C cohort.

Writes gitignored ``outputs/`` artifacts, regenerates memo figures, syncs
committed Posit mock tables, and prints a JSON card of headline numbers for
docs verification.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.paths import (  # noqa: E402
    cohort_source_label,
    full_cohort_paths,
    sibling_data_available,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    if not sibling_data_available():
        raise SystemExit("Full cohort not staged; aborting.")
    source = cohort_source_label()
    if "excerpt" in source.lower():
        raise SystemExit(f"Refusing excerpt source: {source}")
    prolific, qualtrics = full_cohort_paths()
    print("DATA_SOURCE:", source)
    print("PROLIFIC:", prolific)
    print("QUALTRICS:", qualtrics)

    ca = str(ROOT / ".venv" / "bin" / "ca-personas")
    if not Path(ca).is_file():
        ca = "ca-personas"

    # Core cleaning / GT / mock pipeline
    _run([ca, "prepare", "--join", "inner"])
    _run([ca, "score-gt", "--join", "inner"])
    _run([ca, "run", "--provider", "mock", "--join", "inner", "--output-dir", "outputs"])

    # Secondary RQs
    _run([ca, "transit-ca", "--join", "inner", "--seed", "42"])
    _run([ca, "geo-transit-rf", "--join", "inner", "--seed", "42"])
    _run([ca, "ca-transit-rf", "--join", "inner", "--seed", "42"])
    _run(
        [
            ca,
            "covariate-transit-rf",
            "--join",
            "inner",
            "--seed",
            "42",
            "--figures-dir",
            "memos/figures",
        ]
    )
    _run(
        [
            ca,
            "compare",
            "--provider",
            "mock",
            "--join",
            "inner",
            "--output-dir",
            "outputs/ml_vs_llm",
        ]
    )
    _run(
        [
            ca,
            "shap-eval",
            "--provider",
            "mock",
            "--join",
            "inner",
            "--seed",
            "42",
            "--figures-dir",
            "memos/figures",
        ]
    )

    # Notebooks that feed Posit docs (full cohort only)
    nbconvert = [
        str(ROOT / ".venv" / "bin" / "jupyter"),
        "nbconvert",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
        "--ExecutePreprocessor.timeout=1800",
    ]
    notebooks = [
        "notebooks/cleaning_eda_full_cohort.ipynb",
        "notebooks/stage_one_ml_baseline.ipynb",
        "notebooks/factor_feature_importance.ipynb",
        "notebooks/ml_vs_llm_comparison.ipynb",
        "notebooks/secondary_rq_transit_ca.ipynb",
        "notebooks/secondary_rq_geo_transit_rf.ipynb",
        "notebooks/secondary_rq_ca_transit_rf.ipynb",
        "notebooks/secondary_rq_transit_covariate_followups.ipynb",
        "notebooks/secondary_rq_car_access_transit_rf.ipynb",
        "notebooks/secondary_rq_employment_transit_rf.ipynb",
        "notebooks/secondary_rq_rideshare_transit_rf.ipynb",
        "notebooks/feature_predictive_power_shap.ipynb",
    ]
    for nb in notebooks:
        _run(nbconvert + [nb])

    # Posit committed mock artifacts (from sibling File A/B/C).
    # If only committed participants exist, prefer:
    #   python scripts/rerun_posit_mock_from_participants.py
    _run([sys.executable, "scripts/sync_posit_full_cohort_artifacts.py"])

    # Refresh docs/figures that notebooks write under outputs/ then copy if present
    fig_candidates = {
        "outputs/ml_baseline/ml_baseline_mae_group.png": "docs/figures/ml_baseline_mae_group.png",
        "outputs/feature_importance/feature_importance_top.png": "docs/figures/feature_importance_top.png",
        "outputs/geo_transit_rf/geo_rf_auc_vs_baselines.png": "docs/figures/geo_rf_auc_vs_baselines.png",
        "outputs/ca_transit_rf/ca_rf_auc_comparison.png": "docs/figures/ca_rf_auc_comparison.png",
        "outputs/transit_ca/transit_ca_distributions.png": "docs/figures/transit_ca_distributions.png",
        "outputs/transit_ca/transit_riders_ca_means.png": "docs/figures/transit_riders_ca_means.png",
    }
    for src, dest in fig_candidates.items():
        sp, dp = ROOT / src, ROOT / dest
        if sp.is_file():
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)
            print(f"copied {src} -> {dest}")

    headline = {"data_source": source, "n_checks": {}}
    # Pull key results cards
    cards = {
        "transit_ca": ROOT / "outputs/transit_ca/transit_ca_results_card.json",
        "geo": ROOT / "outputs/geo_transit_rf/geo_transit_rf_results_card.json",
        "ca_rf": ROOT / "outputs/ca_transit_rf/ca_transit_rf_results_card.json",
        "covariate": ROOT / "outputs/transit_covariate_rf/comparison_summary.json",
        "posit": ROOT / "artifacts/posit_full_cohort/source.json",
    }
    for key, path in cards.items():
        if path.is_file():
            headline[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            # some pipelines use alternate names
            alts = list(path.parent.glob("*results_card*.json")) + list(
                path.parent.glob("*.json")
            )
            headline[key] = {"missing": str(path), "alts": [str(a) for a in alts[:10]]}

    out_card = ROOT / "outputs" / "full_cohort_rerun_headline.json"
    out_card.parent.mkdir(parents=True, exist_ok=True)
    out_card.write_text(json.dumps(headline, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_card), "data_source": source}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

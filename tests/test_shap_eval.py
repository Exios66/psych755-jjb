"""Tests for SHAP / F1 feature predictive-power evaluation."""

from pathlib import Path

import numpy as np
import pytest

from ca_personas.load import load_and_prepare
from ca_personas.shap_eval import (
    band_f1_metrics,
    ml_metrics_with_f1,
    require_shap,
    run_ml_shap_bundle,
    run_shap_feature_eval,
    scores_to_bands,
)

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_scores_to_bands_and_f1():
    assert scores_to_bands([10, 16, 24]) == ["low", "moderate", "high"]
    metrics = band_f1_metrics([10, 16, 24, 12], [11, 18, 22, 20])
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert "f1_low" in metrics


def test_ml_metrics_include_f1():
    participants = load_and_prepare(PROLIFIC, QUALTRICS, how="inner", clean=True)
    preds, metrics = ml_metrics_with_f1(participants, tiers=["demos", "employment"])
    assert not preds.empty
    assert "f1_macro" in metrics.columns
    assert set(metrics["tier"]) <= {"demos", "employment"}


def test_shap_bundle_on_excerpts():
    require_shap()
    participants = load_and_prepare(PROLIFIC, QUALTRICS, how="inner", clean=True)
    bundle = run_ml_shap_bundle(participants, tier="demos", max_samples=50)
    assert "gt_group_ca" in bundle
    raw = bundle["gt_group_ca"]["raw_importance"]
    assert "mean_abs_shap" in raw.columns
    assert raw["mean_abs_shap"].ge(0).all()
    assert np.isfinite(bundle["gt_group_ca"]["expected_value"])


def test_full_shap_eval_pipeline(tmp_path: Path):
    require_shap()
    result = run_shap_feature_eval(
        prolific_paths=[PROLIFIC],
        qualtrics_path=QUALTRICS,
        join_how="inner",
        llm_provider="mock",
        shap_tier="employment",
        output_dir=tmp_path / "shap_eval",
        max_shap_samples=40,
    )
    assert result["paths"]["results_card"].exists()
    assert result["paths"]["metrics_ml_llm"].exists()
    assert len(result["figure_paths"]) >= 6
    card = result["results_card"]
    assert card["sample"]["n_analytic"] >= 1
    assert card["ml_transit"]["top_shap_group"]

from pathlib import Path

from ca_personas.feature_importance import (
    likert_item_matrix,
    predictor_feature_importance,
    run_factor_and_importance_bundle,
    run_item_factor_analysis,
)
from ca_personas.load import load_and_prepare, load_qualtrics

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_item_factor_analysis_on_qualtrics_excerpt():
    qualtrics = load_qualtrics(QUALTRICS)
    items = likert_item_matrix(qualtrics, scored_direction=True)
    assert len(items) >= 3
    result = run_item_factor_analysis(items, n_factors=2)
    assert result["n_factors"] == 2
    assert list(result["pca_variance"]["component"]) == ["PC1", "PC2"]
    assert result["fa_loadings"].shape[1] == 2


def test_predictor_importance_and_bundle(tmp_path: Path):
    participants = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    importances = predictor_feature_importance(participants, tier="employment", n_repeats=5)
    assert "gt_group_ca__raw_permutation" in importances
    perm = importances["gt_group_ca__raw_permutation"]
    assert "Age" in set(perm["feature"])
    assert perm["permutation_importance_mean"].notna().all()

    paths = run_factor_and_importance_bundle(
        PROLIFIC,
        QUALTRICS,
        tmp_path,
        join_how="inner",
        tier="employment",
    )
    assert paths["top_predictive_features"].exists()
    assert paths["item_fa_loadings"].exists()

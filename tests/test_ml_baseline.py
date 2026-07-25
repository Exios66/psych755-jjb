from pathlib import Path

from ca_personas.ml_baseline import (
    CLASSIC_MODEL_SUITE,
    DEFAULT_MODEL_SUITE,
    baseline_models,
    leaderboard,
    mae_pivot,
    metrics_wide,
    prepare_modeling_frame,
    run_stage_one_baselines,
    save_baseline_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_baseline_models_default_suite_includes_modern_learners():
    specs = baseline_models(random_state=0)
    names = [s.name for s in specs]
    assert names == list(DEFAULT_MODEL_SUITE)
    assert {"ridge", "xgboost", "mlp", "hist_gradient_boosting"}.issubset(names)


def test_baseline_models_include_filter():
    specs = baseline_models(include=CLASSIC_MODEL_SUITE, random_state=0)
    assert [s.name for s in specs] == list(CLASSIC_MODEL_SUITE)


def test_stage_one_baselines_full_suite_metrics(tmp_path: Path):
    participants, predictions, metrics = run_stage_one_baselines(
        PROLIFIC,
        QUALTRICS,
        tiers=["demos", "employment"],
        join_how="inner",
        n_neighbors=2,
        random_state=0,
    )

    assert len(participants) >= 2
    assert set(metrics["model"]) == set(DEFAULT_MODEL_SUITE)
    assert set(metrics["tier"]) == {"demos", "employment"}
    assert metrics["mae"].between(0, 24).all()  # max distance on 6–30 scale
    assert predictions["y_pred"].between(6, 30).all()
    assert {"model_label", "model_family"}.issubset(metrics.columns)

    modelable = prepare_modeling_frame(participants, tier="demos")
    assert {"Age", "Sex", "gt_group_ca", "gt_interpersonal_ca"}.issubset(modelable.columns)

    paths = save_baseline_artifacts(predictions, metrics, tmp_path)
    assert paths["metrics"].exists()
    assert paths["leaderboard"].exists()
    assert paths["mae_pivot_group"].exists()
    wide = metrics_wide(metrics)
    assert {"mae_group", "mae_interpersonal"}.issubset(wide.columns)
    board = leaderboard(metrics)
    assert {"best_model", "best_mae"}.issubset(board.columns)
    pivot = mae_pivot(metrics, target="gt_group_ca")
    assert "demos" in pivot.columns


def test_stage_one_baselines_classic_subset():
    _, _, metrics = run_stage_one_baselines(
        PROLIFIC,
        QUALTRICS,
        tiers=["demos"],
        join_how="inner",
        n_neighbors=2,
        random_state=0,
        models=CLASSIC_MODEL_SUITE,
    )
    assert set(metrics["model"]) == set(CLASSIC_MODEL_SUITE)

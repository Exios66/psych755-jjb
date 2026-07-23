from pathlib import Path

from ca_personas.ml_baseline import (
    metrics_wide,
    prepare_modeling_frame,
    run_stage_one_baselines,
    save_baseline_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_stage_one_baselines_produce_rf_and_knn_metrics(tmp_path: Path):
    participants, predictions, metrics = run_stage_one_baselines(
        PROLIFIC,
        QUALTRICS,
        tiers=["demos", "employment"],
        join_how="inner",
        n_neighbors=2,
        random_state=0,
    )

    assert len(participants) >= 2
    assert set(metrics["model"]) == {"random_forest", "knn"}
    assert set(metrics["tier"]) == {"demos", "employment"}
    assert metrics["mae"].between(0, 24).all()  # max distance on 6–30 scale
    assert predictions["y_pred"].between(6, 30).all()

    modelable = prepare_modeling_frame(participants, tier="demos")
    assert {"Age", "Sex", "gt_group_ca", "gt_interpersonal_ca"}.issubset(modelable.columns)

    paths = save_baseline_artifacts(predictions, metrics, tmp_path)
    assert paths["metrics"].exists()
    wide = metrics_wide(metrics)
    assert {"mae_group", "mae_interpersonal"}.issubset(wide.columns)

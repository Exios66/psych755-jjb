from pathlib import Path

from ca_personas.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_mock_pipeline_writes_evaluation(tmp_path: Path):
    artifacts = run_pipeline(
        prolific_path=ROOT / "data/excerpts/prolific_excerpt.csv",
        qualtrics_path=ROOT / "data/excerpts/qualtrics_excerpt.csv",
        tiers=["demos", "employment"],
        provider="mock",
        output_dir=tmp_path / "outputs",
        join_how="inner",
    )
    assert artifacts["predictions"].exists()
    assert artifacts["evaluation"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["ground_truth"].exists()
    summary_text = artifacts["summary"].read_text()
    assert "demos" in summary_text
    assert "mae_group" in summary_text
    assert "band_acc_group" in summary_text
    assert "exact_acc_group" in summary_text
    assert "mean_band_distance_group" in summary_text
    assert "mean_norm_score_distance_group" in summary_text

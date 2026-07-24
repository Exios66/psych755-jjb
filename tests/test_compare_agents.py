from pathlib import Path

from ca_personas.compare_agents import build_comparison_table, run_ml_vs_llm_comparison

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_ml_vs_llm_comparison_shared_metrics(tmp_path: Path):
    result = run_ml_vs_llm_comparison(
        PROLIFIC,
        QUALTRICS,
        tiers=["demos", "employment"],
        llm_provider="mock",
        join_how="inner",
        output_dir=tmp_path / "cmp",
    )
    comparison = result["comparison"]
    assert {"ml", "llm"}.issubset(set(comparison["agent_family"]))
    assert {"mae_group", "band_acc_group", "mean_norm_score_distance_group"}.issubset(
        comparison.columns
    )
    tiers = set(comparison.loc[comparison["tier"] != "all", "tier"])
    assert tiers == {"demos", "employment"}

    assert result["artifacts"]["comparison"].exists()
    assert result["artifacts"]["deltas"].exists()

    ml_preds = result["predictions"]
    ml_preds = ml_preds[ml_preds["agent_family"] == "ml"]
    assert ml_preds["pred_group_ca"].between(6, 30).all()
    assert ml_preds["pred_interpersonal_ca"].between(6, 30).all()

    table = build_comparison_table(result["summaries"])
    assert not table.empty

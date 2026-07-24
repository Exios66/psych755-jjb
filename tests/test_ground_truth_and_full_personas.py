from pathlib import Path

from ca_personas.ground_truth import export_ground_truth_bundle, ground_truth_table
from ca_personas.load import load_and_prepare
from ca_personas.personas import build_persona_prompt, build_persona_prompts, write_persona_bundle

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_ground_truth_export_bundle(tmp_path: Path):
    paths = export_ground_truth_bundle(PROLIFIC, QUALTRICS, tmp_path, join_how="inner")
    assert paths["ground_truth"].exists()
    assert paths["aggregates"].exists()
    gt = ground_truth_table(
        __import__("pandas").read_csv(paths["participants_scored"])
    )
    assert gt["gt_group_ca"].between(6, 30).all()
    assert set(gt["gt_group_band"]).issubset({"low", "moderate", "high"})


def test_full_persona_includes_qualtrics_voice_when_present():
    df = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    row = df.dropna(subset=["Age", "participant_id"]).iloc[0]

    prompt = build_persona_prompt(row, "full")
    assert "Demographics:" in prompt.user_prompt
    assert "Fully personify" in prompt.user_prompt
    assert "band" in prompt.user_prompt.lower()
    # Free-response section appears when Qualtrics voice fields exist on the row.
    if str(row.get("Q18_advice", "")).strip() or str(row.get("Q19", "")).strip():
        assert "Self-described attitudes" in prompt.user_prompt


def test_write_persona_bundle(tmp_path: Path):
    df = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    prompts = build_persona_prompts(df, tiers=["demos", "full"])
    bundle = write_persona_bundle(prompts, tmp_path)
    assert bundle["n_prompts"] == len(df) * 2
    assert Path(bundle["csv"]).exists()
    assert any(p.name.endswith("__full.md") for p in bundle["markdown_files"])

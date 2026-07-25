from pathlib import Path

from ca_personas.ground_truth import export_ground_truth_bundle, ground_truth_table
from ca_personas.load import load_and_prepare
from ca_personas.personas import build_persona_prompt, build_persona_prompts, write_persona_bundle

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_ground_truth_export_bundle(tmp_path: Path):
    import pandas as pd

    from ca_personas.ground_truth import aggregate_ground_truth

    paths = export_ground_truth_bundle(PROLIFIC, QUALTRICS, tmp_path, join_how="inner")
    assert paths["ground_truth"].exists()
    assert paths["aggregates"].exists()
    scored = pd.read_csv(paths["participants_scored"])
    gt = ground_truth_table(scored)
    assert gt["gt_group_ca"].between(6, 30).all()
    assert set(gt["gt_group_band"]).issubset({"low", "moderate", "high"})
    assert "Student status" in gt.columns

    aggregates = pd.read_csv(paths["aggregates"])
    assert "student_status" in set(aggregates["scope"])
    # Direct call mirrors the export path and keeps student in base demos reporting.
    direct = aggregate_ground_truth(scored)
    assert "student_status" in set(direct["scope"])


def test_full_persona_includes_qualtrics_voice_when_present():
    import pandas as pd

    from ca_personas.personas import _present

    df = load_and_prepare(PROLIFIC, QUALTRICS, how="inner", clean=True)
    row = df.dropna(subset=["Age", "participant_id"]).iloc[0]

    prompt = build_persona_prompt(row, "full")
    assert "Demographics:" in prompt.user_prompt
    assert "Fully personify" in prompt.user_prompt
    assert "band" in prompt.user_prompt.lower()
    # Free-response section appears only when voice fields are truly present
    # (NaN stringifies to "nan"; do not treat that as content).
    has_voice = _present(row.get("Q18_advice")) or _present(row.get("Q19"))
    if has_voice:
        assert "Self-described attitudes" in prompt.user_prompt
    else:
        # Prefer a row that actually has open text when the fixture provides one.
        voice_rows = df[
            df.apply(
                lambda r: _present(r.get("Q18_advice")) or _present(r.get("Q19")),
                axis=1,
            )
        ]
        if len(voice_rows):
            voiced = build_persona_prompt(voice_rows.iloc[0], "full")
            assert "Self-described attitudes" in voiced.user_prompt
        else:
            assert isinstance(row.get("participant_id"), (str, int)) or pd.notna(
                row.get("participant_id")
            )


def test_write_persona_bundle(tmp_path: Path):
    df = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    prompts = build_persona_prompts(df, tiers=["demos", "full"])
    bundle = write_persona_bundle(prompts, tmp_path)
    assert bundle["n_prompts"] == len(df) * 2
    assert Path(bundle["csv"]).exists()
    assert any(p.name.endswith("__full.md") for p in bundle["markdown_files"])

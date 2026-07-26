from pathlib import Path

import pandas as pd

from ca_personas.load import load_and_prepare, load_prolific, load_qualtrics
from ca_personas.personas import (
    BASE_DEMO_FIELDS,
    build_persona_prompt,
    build_persona_prompts,
    demos_block,
    geo_sentences,
    transit_sentences,
)

ROOT = Path(__file__).resolve().parents[1]
PROLIFIC = ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
QUALTRICS = ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def test_load_excerpts_and_score():
    prolific = load_prolific(PROLIFIC)
    qualtrics = load_qualtrics(QUALTRICS)
    assert len(prolific) == 9
    assert "participant_id" in qualtrics.columns
    assert "LocationLatitude" in qualtrics.columns

    joined = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    assert len(joined) >= 1
    assert joined["gt_group_ca"].notna().any()
    assert joined["gt_interpersonal_ca"].notna().any()
    # Scores must land in PRCA subscale range when present.
    valid = joined.dropna(subset=["gt_group_ca"])
    assert valid["gt_group_ca"].between(6, 30).all()


def test_base_demo_fields_include_student_status():
    assert "Student status" in BASE_DEMO_FIELDS
    assert BASE_DEMO_FIELDS == (
        "Age",
        "Sex",
        "Country of residence",
        "Student status",
    )


def test_demos_block_includes_student_status_when_present():
    row = pd.Series(
        {
            "Age": 22,
            "Sex": "Female",
            "Country of residence": "United States",
            "Student status": "Yes",
        }
    )
    lines = demos_block(row)
    joined = " ".join(lines)
    # Natural-language digital-twin framing (AI Terrarium style).
    assert "22-year-old" in joined
    assert "woman" in joined
    assert "United States" in joined
    assert "You are a student." in joined


def test_tiered_prompts_include_expected_sections():
    df = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    row = df.iloc[0]
    demos = build_persona_prompt(row, "demos").user_prompt
    employment = build_persona_prompt(row, "employment").user_prompt
    geo = build_persona_prompt(row, "geo").user_prompt
    transit = build_persona_prompt(row, "transit").user_prompt

    # Persona is conveyed as second-person prose, not a labeled checklist.
    assert demos.startswith("You are")
    assert "Demographics:" not in demos
    assert "Adopt the following identity" not in demos
    assert "Fully personify" not in demos

    # Employment cue appears only from the employment tier onward.
    emp_sentence_markers = (
        "You work",
        "You are unemployed",
        "not in paid work",
        "employment status",
    )
    assert not any(marker in demos for marker in emp_sentence_markers)
    has_emp = pd.notna(row.get("Employment status")) and str(row.get("Employment status")).strip()
    if has_emp:
        assert any(marker in employment for marker in emp_sentence_markers)
    assert "approximate survey location" in geo.lower()
    assert "latitude" in geo.lower()
    # Geo uses 1-decimal coords (no false-precision 4-decimal dump).
    assert "public transportation" in transit.lower()
    assert "independently" in demos.lower()
    assert "mid-scale scores are common" in demos.lower()

    # Student status is part of the base demos layer for every tier when present.
    if pd.notna(row.get("Student status")) and str(row.get("Student status")).strip():
        assert "student" in demos.lower()
        assert "student" in employment.lower()
        assert "student" in geo.lower()
        assert "student" in transit.lower()

    system = build_persona_prompt(row, "demos").system_prompt
    assert "student status" in system.lower()
    assert "inhabit" in system.lower()
    assert "independently" in system.lower()

    prompts = build_persona_prompts(df, tiers=["demos", "employment"])
    assert len(prompts) == len(df) * 2


def test_geo_sentences_round_coords_and_skip_country_repeat():
    row = pd.Series(
        {
            "Country of residence": "United States",
            "LocationLatitude": 43.12345,
            "LocationLongitude": -89.45678,
        }
    )
    joined = " ".join(geo_sentences(row))
    assert "approximate survey location" in joined
    assert "43.1" in joined
    assert "-89.5" in joined
    assert "43.12345" not in joined
    # Country belongs in demos; geo must not restate it.
    assert "United States" not in joined


def test_transit_skips_intensity_when_frequency_is_never():
    never_row = pd.Series(
        {
            "Q26": "Never",
            "Q27": "1-2 rides in a typical day",
            "Q28": "Never",
            "Q29": "3-4 rides in a typical day",
            "Q20": "Yes",
            "Q21": "Yes",
        }
    )
    never_text = " ".join(transit_sentences(never_row))
    assert "never used public transportation" in never_text
    assert "never used ride share" in never_text
    assert "typical day of public transportation" not in never_text
    assert "typical day of ride share" not in never_text
    assert "license to drive" in never_text

    used_row = pd.Series(
        {
            "Q26": "4-8 days a month",
            "Q27": "1-2 rides in a typical day",
            "Q28": "0-1 days a month",
            "Q29": "1-2 rides in a typical day",
        }
    )
    used_text = " ".join(transit_sentences(used_row))
    assert "4-8 days a month" in used_text
    assert "typical day of public transportation" in used_text
    assert "typical day of ride share" in used_text

"""Tests for File A/B/C schemas without committing private cohort data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ca_personas.clean import clean_joined_participants, complete_ca_mask
from ca_personas.eda import research_alignment_summary, run_eda
from ca_personas.load import join_participant_data, load_prolific, load_qualtrics
from ca_personas.paths import sibling_data_available
from ca_personas.personas import build_persona_prompts


def _write_flat_qualtrics(path: Path, rows: list[dict]) -> None:
    """Write a File-C-style single-header Qualtrics CSV."""
    cols = [
        "StartDate",
        "EndDate",
        "Duration (in seconds)",
        "ResponseId",
        "LocationLatitude",
        "LocationLongitude",
        "Q0",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q13",
        "Q14",
        "Q15",
        "Q16",
        "Q17",
        "Q18",
        "Q26",
        "Q27",
        "Q28",
        "Q29",
        "Q20",
        "Q21",
        "Q18.1",
        "Q19",
    ]
    pd.DataFrame(rows)[cols].to_csv(path, index=False)


def _likert_row(pid: str, response_id: str) -> dict:
    return {
        "StartDate": "2025",
        "EndDate": "2025",
        "Duration (in seconds)": 120,
        "ResponseId": response_id,
        "LocationLatitude": 43.0,
        "LocationLongitude": -89.0,
        "Q0": pid,
        "Q1": "Somewhat agree",
        "Q2": "Somewhat disagree",
        "Q3": "Somewhat agree",
        "Q4": "Somewhat disagree",
        "Q5": "Somewhat agree",
        "Q6": "Somewhat disagree",
        "Q13": "Somewhat agree",
        "Q14": "Somewhat disagree",
        "Q15": "Somewhat agree",
        "Q16": "Somewhat disagree",
        "Q17": "Somewhat disagree",
        "Q18": "Somewhat agree",
        "Q26": "Never",
        "Q27": "1-2 rides in a typical day",
        "Q28": "Never",
        "Q29": "1-2 rides in a typical day",
        "Q20": "Yes",
        "Q21": "Yes",
        "Q18.1": "Take a breath and say hello.",
        "Q19": "I prefer walking.",
    }


def _write_minimal_prolific(path: Path, pid: str, *, employment: str = "Full-Time") -> None:
    """Write a File-A/B-style Prolific CSV (no ethnicity/nationality/language)."""
    pd.DataFrame(
        [
            {
                "Submission id": "sub1",
                "Participant id": pid,
                "Started at": 2025,
                "Completed at": 2025,
                "Time taken": 100.0,
                "Age": 30.0,
                "Total approvals": "100-199",
                "Sex": "Female",
                "Country of residence": "United States",
                "Student status": "No",
                "Employment status": employment,
            }
        ]
    ).to_csv(path, index=False)


def test_flat_qualtrics_and_sparse_prolific(tmp_path: Path):
    pid = "abc123"
    prolific_a = tmp_path / "PRCAProlificExport_FileA.csv"
    prolific_b = tmp_path / "PRCAProlificExport_FileB.csv"
    qualtrics = tmp_path / "PRCAQualtricsExport_FileC.csv"

    _write_minimal_prolific(prolific_a, pid, employment="Part-Time")
    _write_minimal_prolific(prolific_b, "other999", employment="Full-Time")
    _write_flat_qualtrics(qualtrics, [_likert_row(pid, "r1")])

    prolific = load_prolific([prolific_a, prolific_b], wave_labels=("A", "B"))
    assert len(prolific) == 2
    assert set(prolific["prolific_wave"]) == {"A", "B"}
    assert "Ethnicity simplified" not in prolific.columns

    qdf = load_qualtrics(qualtrics)
    assert len(qdf) == 1
    assert qdf.loc[0, "participant_id"] == pid
    assert "Q18_ca" in qdf.columns
    assert "Q18_advice" in qdf.columns
    assert qdf.loc[0, "Q18_advice"] == "Take a breath and say hello."

    joined = join_participant_data(prolific, qdf, how="inner")
    assert len(joined) == 1
    analytic, report = clean_joined_participants(joined)
    assert report.n_analytic == 1
    assert analytic["gt_group_ca"].between(6, 30).all()
    assert bool(analytic.loc[0, "has_employment_info"])
    assert bool(analytic.loc[0, "has_transit_info"])

    prompts = build_persona_prompts(analytic, tiers=["demos", "employment", "transit", "full"])
    assert len(prompts) == 4
    transit_prompt = next(p for p in prompts if p.tier == "transit")
    assert "Employment status" in transit_prompt.user_prompt
    assert "Public transportation days" in transit_prompt.user_prompt
    full_prompt = next(p for p in prompts if p.tier == "full")
    assert "Take a breath" in full_prompt.user_prompt

    eda_paths = run_eda(analytic, tmp_path / "eda", cleaning_report=report.to_dict())
    assert eda_paths["research_alignment"].exists()
    summary = research_alignment_summary(analytic, report.to_dict())
    assert summary["n_analytic"] == 1
    assert "rq1_employment" in summary["research_questions"]


def test_complete_ca_mask_accepts_q18_ca():
    row = _likert_row("p", "r")
    df = pd.DataFrame([row]).rename(columns={"Q18": "Q18_ca"})
    assert bool(complete_ca_mask(df).iloc[0])


def test_sibling_data_optional_integration():
    """If private sibling exports are present, exercise the full-cohort loader."""
    if not sibling_data_available():
        return
    from ca_personas.load import load_full_cohort

    analytic, report = load_full_cohort(join_how="inner")
    assert report["n_analytic"] >= 1
    assert analytic["gt_group_ca"].notna().all()
    assert analytic["has_employment_info"].any()
    assert analytic["has_transit_info"].any()

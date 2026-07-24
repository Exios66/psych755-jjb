"""Sanity checks for Prolific↔Qualtrics merge coverage and File C loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ca_personas.load import (
    _looks_like_qualtrics_three_row_header,
    load_prolific,
    load_qualtrics,
    merge_coverage_audit,
)
from ca_personas.paths import (
    DEFAULT_PROLIFIC_A,
    DEFAULT_PROLIFIC_B,
    DEFAULT_QUALTRICS_C,
    EXCERPT_QUALTRICS,
    sibling_data_available,
)


def test_excerpt_still_detected_as_three_row_header():
    raw = pd.read_csv(EXCERPT_QUALTRICS, header=None, dtype=str, keep_default_na=False)
    assert _looks_like_qualtrics_three_row_header(raw) is True


def test_flat_file_c_not_misread_as_three_row_when_present():
    if not DEFAULT_QUALTRICS_C.is_file():
        return
    raw = pd.read_csv(DEFAULT_QUALTRICS_C, header=None, dtype=str, keep_default_na=False)
    assert _looks_like_qualtrics_three_row_header(raw) is False
    qual = load_qualtrics(DEFAULT_QUALTRICS_C)
    # Flat File C has 273 response rows (single header).
    assert len(qual) == 273


def test_full_cohort_merge_coverage_252_21_10():
    """
    Expected accounting from the project data owners:

    - 252 matched Prolific ∩ Qualtrics
    - 21 Qualtrics-only (test / unmatched; disregard)
    - 10 Prolific-only (disregard)
    """
    if not sibling_data_available():
        return
    prolific = load_prolific(
        [DEFAULT_PROLIFIC_A, DEFAULT_PROLIFIC_B],
        wave_labels=("A", "B"),
    )
    qualtrics = load_qualtrics(DEFAULT_QUALTRICS_C)
    audit = merge_coverage_audit(prolific, qualtrics)
    assert audit["n_prolific"] == 262
    assert audit["n_qualtrics"] == 273
    assert audit["n_matched_both"] == 252
    assert audit["n_prolific_only"] == 10
    assert audit["n_qualtrics_only"] == 21
    assert audit["n_qualtrics_missing_pid"] == 18
    assert audit["n_qualtrics_only_with_pid"] == 3
    # 252 + 10 = 262 Prolific; 252 + 21 = 273 Qualtrics
    assert audit["n_matched_both"] + audit["n_prolific_only"] == audit["n_prolific"]
    assert audit["n_matched_both"] + audit["n_qualtrics_only"] == audit["n_qualtrics"]


def test_data_dictionary_covers_file_c_instrument():
    root = Path(__file__).resolve().parents[1]
    dd_path = root / "docs" / "qualtrics_data_dictionary.csv"
    assert dd_path.is_file()
    dd = pd.read_csv(dd_path)
    fields = set(dd["field"].astype(str))
    required = {
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
        "LocationLatitude",
        "LocationLongitude",
        "ResponseId",
    }
    assert required <= fields

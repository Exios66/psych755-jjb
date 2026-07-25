"""Full-cohort path resolution prefers staged File A/B/C over excerpts."""

from __future__ import annotations

from pathlib import Path

from ca_personas.paths import (
    EXCERPT_PROLIFIC,
    EXCERPT_QUALTRICS,
    cohort_source_label,
    default_prolific_paths,
    default_qualtrics_path,
    full_cohort_paths,
    sibling_data_available,
    using_excerpt_fallback,
)


def test_full_cohort_preferred_when_staged():
    if not sibling_data_available():
        return
    prolific, qualtrics = full_cohort_paths()
    assert all(p.is_file() for p in prolific)
    assert qualtrics.is_file()
    assert EXCERPT_PROLIFIC not in prolific
    assert qualtrics != EXCERPT_QUALTRICS
    assert using_excerpt_fallback() is False
    assert "excerpt" not in cohort_source_label().lower()
    assert default_prolific_paths() == prolific
    assert default_qualtrics_path() == qualtrics


def test_full_cohort_paths_raises_without_staging(monkeypatch, tmp_path: Path):
    import ca_personas.paths as paths_mod

    monkeypatch.delenv("CA_SIBLING_DATA", raising=False)
    monkeypatch.setattr(
        paths_mod,
        "_staging_dirs",
        lambda: [tmp_path / "empty"],
    )
    try:
        full_cohort_paths()
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    assert using_excerpt_fallback() is True
    assert default_prolific_paths(allow_excerpt_fallback=True) == [EXCERPT_PROLIFIC]
    assert default_qualtrics_path(allow_excerpt_fallback=True) == EXCERPT_QUALTRICS

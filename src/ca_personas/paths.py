"""Resolve project and sibling-data paths without committing private exports."""

from __future__ import annotations

from pathlib import Path

# Repository root: .../psych755-jjb (or /workspace in the cloud agent).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Private exports live next to the repo, never inside it:
#   ../sibling_data/PRCAProlificExport_FileA.csv
#   ../sibling_data/PRCAProlificExport_FileB.csv
#   ../sibling_data/PRCAQualtricsExport_FileC.csv
SIBLING_DATA_DIR = (REPO_ROOT / ".." / "sibling_data").resolve()

DEFAULT_PROLIFIC_A = SIBLING_DATA_DIR / "PRCAProlificExport_FileA.csv"
DEFAULT_PROLIFIC_B = SIBLING_DATA_DIR / "PRCAProlificExport_FileB.csv"
DEFAULT_QUALTRICS_C = SIBLING_DATA_DIR / "PRCAQualtricsExport_FileC.csv"

# Public fixtures used by tests / Posit Connect Cloud renders.
EXCERPT_PROLIFIC = REPO_ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
EXCERPT_QUALTRICS = REPO_ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def sibling_data_available() -> bool:
    """True when all three full-cohort exports are present beside the repo."""
    return all(
        p.is_file()
        for p in (DEFAULT_PROLIFIC_A, DEFAULT_PROLIFIC_B, DEFAULT_QUALTRICS_C)
    )


def default_prolific_paths() -> list[Path]:
    """Prefer stacked full-cohort Prolific waves; fall back to excerpt fixture."""
    if DEFAULT_PROLIFIC_A.is_file() and DEFAULT_PROLIFIC_B.is_file():
        return [DEFAULT_PROLIFIC_A, DEFAULT_PROLIFIC_B]
    if DEFAULT_PROLIFIC_A.is_file():
        return [DEFAULT_PROLIFIC_A]
    return [EXCERPT_PROLIFIC]


def default_qualtrics_path() -> Path:
    """Prefer full Qualtrics export; fall back to excerpt fixture."""
    if DEFAULT_QUALTRICS_C.is_file():
        return DEFAULT_QUALTRICS_C
    return EXCERPT_QUALTRICS

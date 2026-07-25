"""Resolve project and sibling-data paths without committing private exports."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: .../psych755-jjb (or /workspace in the cloud agent).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Private exports live next to the repo, never inside it:
#   ../sibling_data/PRCAProlificExport_FileA.csv
#   ../sibling_data/PRCAProlificExport_FileB.csv
#   ../sibling_data/PRCAQualtricsExport_FileC.csv
#
# Cloud agents / sandboxes may stage the same filenames under /tmp/sibling_data
# or $CA_SIBLING_DATA. Resolution prefers full-cohort locations before excerpts.
SIBLING_DATA_DIR = (REPO_ROOT / ".." / "sibling_data").resolve()

DEFAULT_PROLIFIC_A = SIBLING_DATA_DIR / "PRCAProlificExport_FileA.csv"
DEFAULT_PROLIFIC_B = SIBLING_DATA_DIR / "PRCAProlificExport_FileB.csv"
DEFAULT_QUALTRICS_C = SIBLING_DATA_DIR / "PRCAQualtricsExport_FileC.csv"

FILE_A_NAME = "PRCAProlificExport_FileA.csv"
FILE_B_NAME = "PRCAProlificExport_FileB.csv"
FILE_C_NAME = "PRCAQualtricsExport_FileC.csv"

# Public fixtures used by tests / offline CI when full cohort is unavailable.
EXCERPT_PROLIFIC = REPO_ROOT / "data" / "excerpts" / "prolific_excerpt.csv"
EXCERPT_QUALTRICS = REPO_ROOT / "data" / "excerpts" / "qualtrics_excerpt.csv"


def _staging_dirs() -> list[Path]:
    """Ordered candidate directories that may hold File A/B/C."""
    dirs: list[Path] = []
    env = os.environ.get("CA_SIBLING_DATA", "").strip()
    if env:
        dirs.append(Path(env).expanduser().resolve())
    dirs.append(SIBLING_DATA_DIR)
    dirs.append(Path("/tmp/sibling_data"))
    dirs.append((Path.home() / "sibling_data").resolve())
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _dir_has_full_cohort(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in (FILE_A_NAME, FILE_B_NAME, FILE_C_NAME)
    )


def resolve_sibling_data_dir() -> Path | None:
    """Return the first directory that contains File A/B/C, else None."""
    for directory in _staging_dirs():
        if _dir_has_full_cohort(directory):
            return directory
    return None


def sibling_data_available() -> bool:
    """True when full-cohort File A/B/C exports are resolvable."""
    return resolve_sibling_data_dir() is not None


def full_cohort_paths() -> tuple[list[Path], Path]:
    """Return (prolific_paths, qualtrics_path) for the full cohort.

    Raises ``FileNotFoundError`` when File A/B/C are not staged anywhere.
    """
    directory = resolve_sibling_data_dir()
    if directory is None:
        searched = ", ".join(str(d) for d in _staging_dirs())
        raise FileNotFoundError(
            "Full-cohort File A/B/C not found. Place exports under "
            "../sibling_data/ (or set CA_SIBLING_DATA / stage /tmp/sibling_data). "
            f"Searched: {searched}"
        )
    return (
        [directory / FILE_A_NAME, directory / FILE_B_NAME],
        directory / FILE_C_NAME,
    )


def cohort_source_label() -> str:
    """Human-readable label for the currently resolved data source."""
    directory = resolve_sibling_data_dir()
    if directory is not None:
        if directory == SIBLING_DATA_DIR:
            return "../sibling_data File A/B/C"
        return f"{directory} File A/B/C"
    return "data/excerpts (File A/B/C not found; small-N demo)"


def _partial_sibling_files() -> list[Path]:
    """Return any staged A/B/C files that exist without a complete trio."""
    present: list[Path] = []
    for directory in _staging_dirs():
        for name in (FILE_A_NAME, FILE_B_NAME, FILE_C_NAME):
            path = directory / name
            if path.is_file():
                present.append(path)
        if present and not _dir_has_full_cohort(directory):
            return present
        present = []
    return []


def default_prolific_paths(*, allow_excerpt_fallback: bool = True) -> list[Path]:
    """Prefer stacked full-cohort Prolific waves; optionally fall back to excerpt."""
    directory = resolve_sibling_data_dir()
    if directory is not None:
        return [directory / FILE_A_NAME, directory / FILE_B_NAME]
    partial = _partial_sibling_files()
    if partial:
        raise FileNotFoundError(
            "Partial File A/B/C staging detected; refusing to mix with excerpt "
            f"fixtures. Present: {partial}. Stage the complete trio or remove "
            "the partial exports."
        )
    if not allow_excerpt_fallback:
        full_cohort_paths()  # raises with a clear message
    return [EXCERPT_PROLIFIC]


def default_qualtrics_path(*, allow_excerpt_fallback: bool = True) -> Path:
    """Prefer full Qualtrics export; optionally fall back to excerpt fixture."""
    directory = resolve_sibling_data_dir()
    if directory is not None:
        return directory / FILE_C_NAME
    partial = _partial_sibling_files()
    if partial:
        raise FileNotFoundError(
            "Partial File A/B/C staging detected; refusing to mix with excerpt "
            f"fixtures. Present: {partial}. Stage the complete trio or remove "
            "the partial exports."
        )
    if not allow_excerpt_fallback:
        full_cohort_paths()  # raises with a clear message
    return EXCERPT_QUALTRICS


def using_excerpt_fallback() -> bool:
    """True when path defaults would resolve to public excerpt fixtures."""
    return resolve_sibling_data_dir() is None

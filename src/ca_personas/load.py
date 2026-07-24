"""Load and join Prolific demographics with Qualtrics responses.

Supports:

- Excerpt fixtures (full Prolific column set + 3-row Qualtrics header)
- Full cohort exports in ``../sibling_data/``:
  - File A + File B Prolific waves (stacked)
  - File C flat Qualtrics export (single header; ``Q0`` merge key; ``Q18.1`` open text)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ca_personas.clean import clean_joined_participants
from ca_personas.scoring import add_ground_truth_scores

# Required for every Prolific export we accept.
PROLIFIC_REQUIRED = [
    "Participant id",
    "Age",
    "Sex",
    "Country of residence",
    "Student status",
    "Employment status",
]

# Present in excerpt fixtures / richer Prolific exports; optional for File A/B.
PROLIFIC_OPTIONAL = [
    "Submission id",
    "Status",
    "Started at",
    "Completed at",
    "Time taken",
    "Total approvals",
    "Ethnicity simplified",
    "Country of birth",
    "Nationality",
    "Language",
]

PROLIFIC_KEEP = PROLIFIC_REQUIRED + PROLIFIC_OPTIONAL

QUALTRICS_META = [
    "ResponseId",
    "LocationLatitude",
    "LocationLongitude",
    "UserLanguage",
    "Finished",
    "Progress",
    "Duration (in seconds)",
    "StartDate",
    "EndDate",
]

# CA items + transit / open-text context used by higher tiers.
QUALTRICS_ITEMS = [
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
    "Q18_ca",
    "Q26",
    "Q27",
    "Q28",
    "Q29",
    "Q20",
    "Q21",
    "Q18_advice",
    "Q19",
    "PROLIFIC_PID",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_prolific(
    path: str | Path | Sequence[str | Path],
    *,
    wave_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load one or more Prolific export CSVs and keep demographic fields.

    Multiple paths (File A + File B) are stacked. Optional richer columns are
    retained when present; File A/B may omit ethnicity / nationality / language.
    """
    paths = _as_paths(path)
    frames: list[pd.DataFrame] = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df = _normalize_columns(df)
        missing = [c for c in PROLIFIC_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Prolific file {p} missing columns: {missing}")
        keep = [c for c in PROLIFIC_KEEP if c in df.columns]
        out = df[keep].copy()
        label = None
        if wave_labels is not None and i < len(wave_labels):
            label = wave_labels[i]
        elif len(paths) > 1:
            # Default wave labels for the two recruitment files.
            stem = p.stem.lower()
            if "filea" in stem or stem.endswith("_a") or "file_a" in stem:
                label = "A"
            elif "fileb" in stem or stem.endswith("_b") or "file_b" in stem:
                label = "B"
            else:
                label = f"wave{i + 1}"
        if label is not None:
            out["prolific_wave"] = label
        frames.append(out)

    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked.rename(columns={"Participant id": "participant_id"})
    stacked["participant_id"] = stacked["participant_id"].astype(str).str.strip()
    # Prefer the earliest wave row when the same ID appears twice (should be rare).
    stacked = stacked.drop_duplicates("participant_id", keep="first")
    return stacked


def _looks_like_qualtrics_three_row_header(raw: pd.DataFrame) -> bool:
    """
    Detect the standard Qualtrics export block:

    Row 0: field names, Row 1: question labels, Row 2: ImportIds.
    """
    if raw.shape[0] < 4:
        return False
    row0 = [str(c).strip() for c in raw.iloc[0].tolist()]
    row1 = [str(c).strip() for c in raw.iloc[1].tolist()]
    row2 = [str(c).strip().lower() for c in raw.iloc[2].tolist()]
    has_field_names = "ResponseId" in row0 or "StartDate" in row0
    # Question-label row is long prose, not another copy of field names.
    labelish = any(len(x) > 40 for x in row1)
    importish = any(x.startswith("{") or "importid" in x for x in row2)
    return bool(has_field_names and (labelish or importish))


def _disambiguate_q18_columns(header: list[str]) -> list[str]:
    """
    Map duplicate / alternate Q18 columns to scoring vs open-text names.

    - First ``Q18`` → ``Q18_ca`` (Likert interpersonal item)
    - ``Q18.1`` or second ``Q18`` → ``Q18_advice`` (open prompt)
    """
    renamed: list[str] = []
    q18_seen = 0
    for name in header:
        if name == "Q18":
            q18_seen += 1
            renamed.append("Q18_ca" if q18_seen == 1 else "Q18_advice")
        elif name in {"Q18.1", "Q18_1", "Q18_advice"}:
            renamed.append("Q18_advice")
        else:
            renamed.append(name)
    return renamed


def load_qualtrics(path: str | Path) -> pd.DataFrame:
    """
    Load a Qualtrics export.

    Accepts either the standard 3-row header block or a flat single-header CSV
    (File C). The merge key is embedded ``PROLIFIC_PID`` when present, else ``Q0``.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if raw.shape[0] < 2:
        raise ValueError("Qualtrics export looks empty or missing header rows")

    if _looks_like_qualtrics_three_row_header(raw):
        header = [str(c).strip() for c in raw.iloc[0].tolist()]
        data = raw.iloc[3:].copy()
    else:
        # Flat export: row 0 is the header; all subsequent rows are responses.
        header = [str(c).strip() for c in raw.iloc[0].tolist()]
        data = raw.iloc[1:].copy()

    renamed = _disambiguate_q18_columns(header)
    data.columns = renamed
    data = data.reset_index(drop=True)

    keep = [c for c in QUALTRICS_META + QUALTRICS_ITEMS if c in data.columns]
    out = data[keep].copy()

    # Prefer embedded PROLIFIC_PID; fall back to Q0 free-text Prolific ID.
    pid = out.get("PROLIFIC_PID", pd.Series([""] * len(out))).astype(str).str.strip()
    q0 = out.get("Q0", pd.Series([""] * len(out))).astype(str).str.strip()
    out["participant_id"] = pid.where(pid.ne(""), q0)
    out["participant_id"] = out["participant_id"].replace({"": pd.NA})

    for col in ("LocationLatitude", "LocationLongitude", "Duration (in seconds)"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def join_participant_data(
    prolific: pd.DataFrame,
    qualtrics: pd.DataFrame,
    *,
    how: str = "outer",
) -> pd.DataFrame:
    """Join Prolific demographics to Qualtrics responses on participant_id."""
    left = prolific.copy()
    right = qualtrics.copy()
    # When multiple Qualtrics rows share an ID, keep the most complete CA response.
    if "participant_id" in right.columns:
        right["_ca_completeness"] = right[
            [
                c
                for c in (
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
                    "Q18_ca",
                )
                if c in right.columns
            ]
        ].apply(lambda r: sum(bool(str(v).strip()) for v in r), axis=1)
        right = (
            right.sort_values("_ca_completeness", ascending=False)
            .drop_duplicates("participant_id", keep="first")
            .drop(columns=["_ca_completeness"])
        )

    merged = left.merge(right, on="participant_id", how=how, suffixes=("_prolific", "_qualtrics"))
    return merged


def load_and_prepare(
    prolific_path: str | Path | Sequence[str | Path],
    qualtrics_path: str | Path,
    *,
    how: str = "outer",
    low_max: int = 13,
    high_min: int = 20,
    clean: bool = False,
    wave_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    End-to-end load → join → ground-truth CA scoring.

    Set ``clean=True`` to apply the analytic-sample filters (complete CA,
    research-covariate flags) used for the full-cohort pipeline.
    """
    prolific = load_prolific(prolific_path, wave_labels=wave_labels)
    qualtrics = load_qualtrics(qualtrics_path)
    merged = join_participant_data(prolific, qualtrics, how=how)
    if clean:
        scored, _report = clean_joined_participants(
            merged,
            require_complete_ca=True,
            low_max=low_max,
            high_min=high_min,
        )
        return scored

    # Scoring uses Q18_ca; alias to Q18 for the scorer's expected name.
    if "Q18_ca" in merged.columns and "Q18" not in merged.columns:
        merged = merged.rename(columns={"Q18_ca": "Q18"})
    scored = add_ground_truth_scores(merged, low_max=low_max, high_min=high_min)
    return scored


def load_full_cohort(
    *,
    prolific_paths: Iterable[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    low_max: int = 13,
    high_min: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """
    Load File A + File B + File C from ``../sibling_data/``, clean, and score.

    Returns ``(analytic_frame, cleaning_report_dict)``.
    """
    from ca_personas.paths import default_prolific_paths, default_qualtrics_path

    prolific_list = list(prolific_paths) if prolific_paths is not None else default_prolific_paths()
    qualtrics = Path(qualtrics_path) if qualtrics_path is not None else default_qualtrics_path()

    prolific = load_prolific(
        prolific_list,
        wave_labels=("A", "B") if len(prolific_list) == 2 else None,
    )
    qdf = load_qualtrics(qualtrics)
    joined = join_participant_data(prolific, qdf, how=join_how)

    from ca_personas.clean import complete_ca_mask

    prolific_ids = set(prolific["participant_id"].dropna().astype(str))
    qual_ids = set(qdf["participant_id"].dropna().astype(str))

    analytic, clean_report = clean_joined_participants(
        joined,
        require_complete_ca=True,
        low_max=low_max,
        high_min=high_min,
    )
    clean_report.n_prolific_raw = int(len(prolific))
    clean_report.n_prolific_unique = int(prolific["participant_id"].nunique())
    clean_report.n_qualtrics_raw = int(len(qdf))
    clean_report.n_qualtrics_with_pid = int(qdf["participant_id"].notna().sum())
    clean_report.n_qualtrics_complete_ca = int(complete_ca_mask(qdf).sum())
    clean_report.n_prolific_only = len(prolific_ids - qual_ids)
    clean_report.n_qualtrics_only = len(qual_ids - prolific_ids)
    clean_report.n_dropped_unjoined = clean_report.n_prolific_only + clean_report.n_qualtrics_only
    return analytic, clean_report.to_dict()

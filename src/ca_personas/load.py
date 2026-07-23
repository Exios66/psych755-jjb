"""Load and join Prolific demographics with Qualtrics responses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ca_personas.scoring import add_ground_truth_scores

PROLIFIC_KEEP = [
    "Participant id",
    "Status",
    "Age",
    "Sex",
    "Ethnicity simplified",
    "Country of birth",
    "Country of residence",
    "Nationality",
    "Language",
    "Student status",
    "Employment status",
]

QUALTRICS_META = [
    "ResponseId",
    "LocationLatitude",
    "LocationLongitude",
    "UserLanguage",
    "Finished",
    "Progress",
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


def load_prolific(path: str | Path) -> pd.DataFrame:
    """Load a Prolific export CSV and keep demographic fields."""
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    missing = [c for c in PROLIFIC_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"Prolific file missing columns: {missing}")
    out = df[PROLIFIC_KEEP].copy()
    out = out.rename(columns={"Participant id": "participant_id"})
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    return out


def load_qualtrics(path: str | Path) -> pd.DataFrame:
    """
    Load a Qualtrics export with the standard 3-row header block.

    Row 0: field names, Row 1: question labels, Row 2: ImportIds — data starts at row 3.
    The export contains two columns named Q18 (CA item + open advice). We rename them
    to Q18_ca and Q18_advice by position.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if raw.shape[0] < 4:
        raise ValueError("Qualtrics export looks empty or missing header rows")

    header = [str(c).strip() for c in raw.iloc[0].tolist()]
    # Disambiguate duplicate Q18 columns by order of appearance.
    renamed: list[str] = []
    q18_seen = 0
    for name in header:
        if name == "Q18":
            q18_seen += 1
            renamed.append("Q18_ca" if q18_seen == 1 else "Q18_advice")
        else:
            renamed.append(name)

    data = raw.iloc[3:].copy()
    data.columns = renamed
    data = data.reset_index(drop=True)

    keep = [c for c in QUALTRICS_META + QUALTRICS_ITEMS if c in data.columns]
    out = data[keep].copy()

    # Prefer embedded PROLIFIC_PID; fall back to Q0 free-text Prolific ID.
    pid = out.get("PROLIFIC_PID", pd.Series([""] * len(out))).astype(str).str.strip()
    q0 = out.get("Q0", pd.Series([""] * len(out))).astype(str).str.strip()
    out["participant_id"] = pid.where(pid.ne(""), q0)
    out["participant_id"] = out["participant_id"].replace({"": pd.NA})

    for col in ("LocationLatitude", "LocationLongitude"):
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
            [c for c in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18_ca") if c in right.columns]
        ].apply(lambda r: sum(bool(str(v).strip()) for v in r), axis=1)
        right = (
            right.sort_values("_ca_completeness", ascending=False)
            .drop_duplicates("participant_id", keep="first")
            .drop(columns=["_ca_completeness"])
        )

    merged = left.merge(right, on="participant_id", how=how, suffixes=("_prolific", "_qualtrics"))
    return merged


def load_and_prepare(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    *,
    how: str = "outer",
    low_max: int = 13,
    high_min: int = 20,
) -> pd.DataFrame:
    """End-to-end load → join → ground-truth CA scoring."""
    prolific = load_prolific(prolific_path)
    qualtrics = load_qualtrics(qualtrics_path)
    merged = join_participant_data(prolific, qualtrics, how=how)
    # Scoring uses Q18_ca; alias to Q18 for the scorer's expected name.
    if "Q18_ca" in merged.columns and "Q18" not in merged.columns:
        merged = merged.rename(columns={"Q18_ca": "Q18"})
    scored = add_ground_truth_scores(merged, low_max=low_max, high_min=high_min)
    return scored

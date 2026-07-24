"""Ground-truth PRCA subscale scoring from Qualtrics Likert responses."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

LIKERT_TO_SCORE = {
    "strongly disagree": 1,
    "somewhat disagree": 2,
    "neither agree nor disagree": 3,
    "somewhat agree": 4,
    "strongly agree": 5,
}

# Straight-coded (anxiety) and reverse-coded (comfort) items per PRCA subscale.
GROUP_STRAIGHT = ("Q1", "Q3", "Q5")
GROUP_REVERSE = ("Q2", "Q4", "Q6")
INTERPERSONAL_STRAIGHT = ("Q13", "Q15", "Q18")
INTERPERSONAL_REVERSE = ("Q14", "Q16", "Q17")

ALL_CA_ITEMS = GROUP_STRAIGHT + GROUP_REVERSE + INTERPERSONAL_STRAIGHT + INTERPERSONAL_REVERSE


def likert_to_int(value: object) -> int | None:
    """Map a Qualtrics Likert label to 1–5. Returns None if missing/unmapped."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return LIKERT_TO_SCORE.get(text)


def reverse_score(score: int) -> int:
    """Reverse a 1–5 Likert item: reverse = 6 - score."""
    return 6 - score


def subscale_score(
    row: pd.Series,
    straight: Iterable[str],
    reverse: Iterable[str],
) -> int | None:
    """
    Sum six PRCA items into a 6–30 subscale score.

    Returns None if any required item is missing or unmapped.
    """
    total = 0
    for col in straight:
        mapped = likert_to_int(row.get(col))
        if mapped is None:
            return None
        total += mapped
    for col in reverse:
        mapped = likert_to_int(row.get(col))
        if mapped is None:
            return None
        total += reverse_score(mapped)
    return total


def ca_band(score: int | None, low_max: int = 13, high_min: int = 20) -> str | None:
    """Map a 6–30 score to low / moderate / high bands."""
    if score is None:
        return None
    if score <= low_max:
        return "low"
    if score >= high_min:
        return "high"
    return "moderate"


# Ordinal encoding for band-distance metrics (adjacent miss < opposite miss).
BAND_ORDINAL = {"low": 0, "moderate": 1, "high": 2}
PRCA_SCORE_MIN = 6
PRCA_SCORE_MAX = 30
PRCA_SCORE_RANGE = PRCA_SCORE_MAX - PRCA_SCORE_MIN  # 24
MAX_BAND_DISTANCE = 2


def band_ordinal(band: object) -> int | None:
    """Map a band label to an ordinal rank for distance calculations."""
    if band is None or (isinstance(band, float) and pd.isna(band)):
        return None
    text = str(band).strip().lower()
    return BAND_ORDINAL.get(text)


def band_distance(pred_band: object, gt_band: object) -> int | None:
    """
    Ordinal steps between predicted and ground-truth bands.

    Returns 0 (same band), 1 (adjacent), or 2 (low↔high). None if either missing.
    """
    pred_ord = band_ordinal(pred_band)
    gt_ord = band_ordinal(gt_band)
    if pred_ord is None or gt_ord is None:
        return None
    return abs(pred_ord - gt_ord)


def normalized_score_distance(abs_error: float | None) -> float | None:
    """Scale absolute 6–30 error into [0, 1] (1 = maximum possible miss)."""
    if abs_error is None or (isinstance(abs_error, float) and pd.isna(abs_error)):
        return None
    return float(abs_error) / float(PRCA_SCORE_RANGE)


def normalized_band_distance(distance: int | None) -> float | None:
    """Scale ordinal band distance into [0, 1] (1 = low↔high miss)."""
    if distance is None:
        return None
    return float(distance) / float(MAX_BAND_DISTANCE)


def add_ground_truth_scores(
    df: pd.DataFrame,
    *,
    low_max: int = 13,
    high_min: int = 20,
) -> pd.DataFrame:
    """Attach group/interpersonal CA scores and bands to a participant frame."""
    out = df.copy()
    out["gt_group_ca"] = out.apply(
        lambda r: subscale_score(r, GROUP_STRAIGHT, GROUP_REVERSE),
        axis=1,
    )
    out["gt_interpersonal_ca"] = out.apply(
        lambda r: subscale_score(r, INTERPERSONAL_STRAIGHT, INTERPERSONAL_REVERSE),
        axis=1,
    )
    out["gt_group_band"] = out["gt_group_ca"].map(
        lambda s: ca_band(None if pd.isna(s) else int(s), low_max, high_min)
    )
    out["gt_interpersonal_band"] = out["gt_interpersonal_ca"].map(
        lambda s: ca_band(None if pd.isna(s) else int(s), low_max, high_min)
    )
    return out

"""Join predictions to ground-truth CA scores; score precision + band accuracy."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ca_personas.scoring import ca_band


def _normalize_band(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in {"low", "moderate", "high"}:
        return text
    return None


def derive_band_from_score(
    score: object,
    *,
    low_max: int = 13,
    high_min: int = 20,
) -> str | None:
    """Map a numeric CA prediction to a PRCA band."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return None
    try:
        return ca_band(int(round(float(score))), low_max=low_max, high_min=high_min)
    except (TypeError, ValueError):
        return None


def evaluate_predictions(
    participants: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    low_max: int = 13,
    high_min: int = 20,
) -> pd.DataFrame:
    """
    Merge predictions onto ground-truth scores and compute:

    - signed / absolute score error (precision on the 6–30 scale)
    - exact score match (pred integer equals ground truth)
    - band match (low / moderate / high agreement)
    """
    gt_cols = [
        "participant_id",
        "Age",
        "Sex",
        "Ethnicity simplified",
        "Country of residence",
        "Employment status",
        "LocationLatitude",
        "LocationLongitude",
        "gt_group_ca",
        "gt_interpersonal_ca",
        "gt_group_band",
        "gt_interpersonal_band",
    ]
    available = [c for c in gt_cols if c in participants.columns]
    gt = participants[available].drop_duplicates("participant_id")

    merged = predictions.merge(gt, on="participant_id", how="left")

    for side in ("group", "interpersonal"):
        pred_col = f"pred_{side}_ca"
        gt_col = f"gt_{side}_ca"
        gt_band_col = f"gt_{side}_band"
        pred_band_col = f"pred_{side}_band"

        if pred_col in merged.columns and gt_col in merged.columns:
            merged[f"error_{side}"] = merged[pred_col] - merged[gt_col]
            merged[f"abs_error_{side}"] = merged[f"error_{side}"].abs()
            merged[f"exact_match_{side}"] = (
                merged[pred_col].round().astype("Int64") == merged[gt_col].round().astype("Int64")
            )

        # Prefer model-reported band; otherwise derive from predicted score.
        if pred_band_col in merged.columns:
            reported = merged[pred_band_col].map(_normalize_band)
        else:
            reported = pd.Series([None] * len(merged), index=merged.index)
        derived = merged[pred_col].map(
            lambda s: derive_band_from_score(s, low_max=low_max, high_min=high_min)
        ) if pred_col in merged.columns else reported
        merged[f"pred_{side}_band_resolved"] = reported.where(reported.notna(), derived)

        if gt_band_col in merged.columns:
            pred_b = merged[f"pred_{side}_band_resolved"].map(_normalize_band)
            gt_b = merged[gt_band_col].map(_normalize_band)
            match = pred_b.eq(gt_b)
            # Use nullable boolean so missing bands stay as <NA>, not False.
            match = match.astype("boolean")
            match = match.mask(pred_b.isna() | gt_b.isna())
            merged[f"band_match_{side}"] = match

    return merged


def summarize_errors(evaluation: pd.DataFrame) -> pd.DataFrame:
    """MAE, exact-score accuracy, and band accuracy by tier (and overall)."""
    rows: list[dict[str, Any]] = []
    frames = [("all", evaluation)]
    if "tier" in evaluation.columns:
        for tier, frame in evaluation.groupby("tier"):
            frames.append((str(tier), frame))

    for label, frame in frames:
        score_usable = frame.dropna(
            subset=["abs_error_group", "abs_error_interpersonal"],
            how="any",
        )
        row: dict[str, Any] = {
            "tier": label,
            "n_predictions": int(len(frame)),
            "n_with_ground_truth": int(len(score_usable)),
            "mae_group": float(score_usable["abs_error_group"].mean()) if len(score_usable) else None,
            "mae_interpersonal": (
                float(score_usable["abs_error_interpersonal"].mean()) if len(score_usable) else None
            ),
            "mean_error_group": (
                float(score_usable["error_group"].mean()) if len(score_usable) else None
            ),
            "mean_error_interpersonal": (
                float(score_usable["error_interpersonal"].mean()) if len(score_usable) else None
            ),
        }

        for side in ("group", "interpersonal"):
            exact_col = f"exact_match_{side}"
            band_col = f"band_match_{side}"
            if exact_col in score_usable.columns and len(score_usable):
                row[f"exact_acc_{side}"] = float(score_usable[exact_col].mean())
            else:
                row[f"exact_acc_{side}"] = None

            if band_col in frame.columns:
                band_usable = frame.dropna(subset=[band_col])
                row[f"band_acc_{side}"] = (
                    float(band_usable[band_col].astype(bool).mean()) if len(band_usable) else None
                )
                row[f"n_band_{side}"] = int(len(band_usable))
            else:
                row[f"band_acc_{side}"] = None
                row[f"n_band_{side}"] = 0

        rows.append(row)
    return pd.DataFrame(rows)


def summarize_band_confusion(evaluation: pd.DataFrame, side: str = "group") -> pd.DataFrame:
    """Contingency counts of predicted vs ground-truth bands for one subscale."""
    pred_col = f"pred_{side}_band_resolved"
    gt_col = f"gt_{side}_band"
    if pred_col not in evaluation.columns or gt_col not in evaluation.columns:
        return pd.DataFrame()
    frame = evaluation.dropna(subset=[pred_col, gt_col]).copy()
    frame["pred_band"] = frame[pred_col].map(_normalize_band)
    frame["gt_band"] = frame[gt_col].map(_normalize_band)
    frame = frame.dropna(subset=["pred_band", "gt_band"])
    if frame.empty:
        return pd.DataFrame()
    table = pd.crosstab(frame["gt_band"], frame["pred_band"], dropna=False)
    for band in ("low", "moderate", "high"):
        if band not in table.index:
            table.loc[band] = 0
        if band not in table.columns:
            table[band] = 0
    return table.loc[["low", "moderate", "high"], ["low", "moderate", "high"]]

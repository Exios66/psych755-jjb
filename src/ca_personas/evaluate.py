"""Join LLM predictions to ground-truth CA scores for error analysis."""

from __future__ import annotations

import pandas as pd


def evaluate_predictions(
    participants: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge predictions onto ground-truth scores and compute absolute/signed errors.

    Only rows with both prediction and ground truth contribute numeric errors;
    missing ground truth is retained with null errors for transparency.
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
        if pred_col in merged.columns and gt_col in merged.columns:
            merged[f"error_{side}"] = merged[pred_col] - merged[gt_col]
            merged[f"abs_error_{side}"] = merged[f"error_{side}"].abs()

    return merged


def summarize_errors(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error by tier (and overall) for both subscales."""
    rows: list[dict[str, object]] = []
    frames = [("all", evaluation)]
    if "tier" in evaluation.columns:
        for tier, frame in evaluation.groupby("tier"):
            frames.append((str(tier), frame))

    for label, frame in frames:
        usable = frame.dropna(subset=["abs_error_group", "abs_error_interpersonal"], how="any")
        rows.append(
            {
                "tier": label,
                "n_predictions": int(len(frame)),
                "n_with_ground_truth": int(len(usable)),
                "mae_group": float(usable["abs_error_group"].mean()) if len(usable) else None,
                "mae_interpersonal": (
                    float(usable["abs_error_interpersonal"].mean()) if len(usable) else None
                ),
                "mean_error_group": float(usable["error_group"].mean()) if len(usable) else None,
                "mean_error_interpersonal": (
                    float(usable["error_interpersonal"].mean()) if len(usable) else None
                ),
            }
        )
    return pd.DataFrame(rows)

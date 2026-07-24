"""Compare traditional ML baselines vs LLM persona agents on shared CA metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ca_personas.evaluate import evaluate_predictions, summarize_errors
from ca_personas.llm.base import get_client
from ca_personas.load import load_and_prepare
from ca_personas.ml_baseline import run_stage_one_baselines
from ca_personas.personas import RESEARCH_TIERS, build_persona_prompts
from ca_personas.predict import run_predictions
from ca_personas.scoring import ca_band

COMPARISON_METRICS = [
    "mae_group",
    "mae_interpersonal",
    "exact_acc_group",
    "exact_acc_interpersonal",
    "band_acc_group",
    "band_acc_interpersonal",
    "mean_norm_score_distance_group",
    "mean_norm_score_distance_interpersonal",
    "mean_band_distance_group",
    "mean_band_distance_interpersonal",
    "mean_norm_band_distance_group",
    "mean_norm_band_distance_interpersonal",
]


def ml_long_to_eval_format(ml_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot stage-one long-form ML predictions into the wide schema used by
    ``evaluate_predictions`` (pred_group_ca / pred_interpersonal_ca + bands).
    """
    required = {"participant_id", "tier", "model", "side", "y_pred"}
    missing = required - set(ml_predictions.columns)
    if missing:
        raise ValueError(f"ML predictions missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    group_keys = ["participant_id", "tier", "model"]
    for keys, frame in ml_predictions.groupby(group_keys, dropna=False):
        participant_id, tier, model = keys
        by_side = {str(r["side"]): r for _, r in frame.iterrows()}
        if "group" not in by_side or "interpersonal" not in by_side:
            continue
        group_pred = float(by_side["group"]["y_pred"])
        inter_pred = float(by_side["interpersonal"]["y_pred"])
        # Prefer stored bands when present; otherwise derive from score.
        group_band = by_side["group"].get("pred_band") or ca_band(int(round(group_pred)))
        inter_band = by_side["interpersonal"].get("pred_band") or ca_band(int(round(inter_pred)))
        rows.append(
            {
                "participant_id": participant_id,
                "tier": tier,
                "agent_family": "ml",
                "agent": f"ml:{model}",
                "model": model,
                "pred_group_ca": group_pred,
                "pred_interpersonal_ca": inter_pred,
                "pred_group_band": group_band,
                "pred_interpersonal_band": inter_band,
            }
        )
    return pd.DataFrame(rows)


def llm_predictions_to_eval_format(llm_predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize LLM prediction rows to the shared evaluation schema."""
    out = llm_predictions.copy()
    if "agent_family" not in out.columns:
        out["agent_family"] = "llm"
    if "agent" not in out.columns:
        provider = out["provider"] if "provider" in out.columns else "llm"
        model = out["model"] if "model" in out.columns else "unknown"
        out["agent"] = provider.astype(str) + ":" + model.astype(str)
    return out


def evaluate_agent_predictions(
    participants: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one agent family's predictions; return row-level eval + tier summary."""
    evaluated = evaluate_predictions(participants, predictions)
    # summarize_errors groups only by tier; add agent labels afterwards.
    pieces: list[pd.DataFrame] = []
    if "agent" in evaluated.columns:
        for agent, frame in evaluated.groupby("agent"):
            summary = summarize_errors(frame)
            summary.insert(0, "agent", agent)
            family = frame["agent_family"].iloc[0] if "agent_family" in frame.columns else None
            summary.insert(1, "agent_family", family)
            pieces.append(summary)
    else:
        summary = summarize_errors(evaluated)
        summary.insert(0, "agent", "unknown")
        summary.insert(1, "agent_family", None)
        pieces.append(summary)
    return evaluated, pd.concat(pieces, ignore_index=True)


def build_comparison_table(summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Stack agent summaries for side-by-side comparison.

    Keeps one row per (agent, tier) with the shared metric columns.
    """
    keep = ["agent", "agent_family", "tier", "n_predictions", "n_with_ground_truth"]
    keep += [c for c in COMPARISON_METRICS if c in summaries.columns]
    frame = summaries[keep].copy()
    # Prefer per-tier rows for head-to-head charts; retain "all" as well.
    return frame.sort_values(["tier", "agent_family", "agent"]).reset_index(drop=True)


def comparison_delta(
    comparison: pd.DataFrame,
    *,
    baseline_agent: str | None = None,
    metric: str = "mae_group",
) -> pd.DataFrame:
    """
    For each tier, compute metric − best ML metric (or a named baseline agent).

    Negative deltas mean the agent beats the baseline on error-like metrics.
    """
    rows: list[dict[str, Any]] = []
    tier_frames = comparison[comparison["tier"] != "all"]
    for tier, frame in tier_frames.groupby("tier"):
        ml_frame = frame[frame["agent_family"] == "ml"]
        if baseline_agent:
            base = frame[frame["agent"] == baseline_agent]
        else:
            base = ml_frame
        if base.empty or metric not in base.columns or base[metric].dropna().empty:
            continue
        # For error metrics lower is better — use the best (minimum) ML value.
        if metric.startswith("mae") or "distance" in metric:
            baseline_value = float(base[metric].min())
            better = "lower"
        else:
            baseline_value = float(base[metric].max())
            better = "higher"
        for _, row in frame.iterrows():
            value = row.get(metric)
            if pd.isna(value):
                continue
            delta = float(value) - baseline_value
            rows.append(
                {
                    "tier": tier,
                    "agent": row["agent"],
                    "agent_family": row["agent_family"],
                    "metric": metric,
                    "value": float(value),
                    "baseline_value": baseline_value,
                    "delta_vs_best_ml": delta,
                    "better_direction": better,
                }
            )
    return pd.DataFrame(rows)


def run_ml_vs_llm_comparison(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    *,
    tiers: Iterable[str] = RESEARCH_TIERS,
    llm_provider: str = "mock",
    llm_model: str | None = None,
    join_how: str = "inner",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run RF/KNN baselines and an LLM persona agent, evaluate on shared metrics,
    and optionally write comparison artifacts.
    """
    selected = [t for t in tiers if t != "full"]
    participants, ml_long, _ml_metrics = run_stage_one_baselines(
        prolific_path,
        qualtrics_path,
        tiers=selected,
        join_how=join_how,
    )
    ml_wide = ml_long_to_eval_format(ml_long)
    ml_eval, ml_summary = evaluate_agent_predictions(participants, ml_wide)

    # LLM path uses the same participant frame / tiers.
    prompts = build_persona_prompts(participants, tiers=selected)
    client = get_client(llm_provider, model=llm_model)
    llm_raw = run_predictions(client, prompts)
    llm_wide = llm_predictions_to_eval_format(llm_raw)
    llm_eval, llm_summary = evaluate_agent_predictions(participants, llm_wide)

    summaries = pd.concat([ml_summary, llm_summary], ignore_index=True)
    comparison = build_comparison_table(summaries)
    deltas = pd.concat(
        [
            comparison_delta(comparison, metric="mae_group"),
            comparison_delta(comparison, metric="mae_interpersonal"),
            comparison_delta(comparison, metric="band_acc_group"),
            comparison_delta(comparison, metric="band_acc_interpersonal"),
            comparison_delta(comparison, metric="mean_norm_score_distance_group"),
            comparison_delta(comparison, metric="mean_band_distance_group"),
        ],
        ignore_index=True,
    )

    predictions = pd.concat([ml_wide, llm_wide], ignore_index=True, sort=False)
    evaluation = pd.concat([ml_eval, llm_eval], ignore_index=True, sort=False)

    artifacts: dict[str, Path] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "predictions": out / "agent_predictions.csv",
            "evaluation": out / "agent_evaluation.csv",
            "summary": out / "agent_summary_by_tier.csv",
            "comparison": out / "ml_vs_llm_comparison.csv",
            "deltas": out / "ml_vs_llm_deltas.csv",
        }
        predictions.to_csv(artifacts["predictions"], index=False)
        evaluation.to_csv(artifacts["evaluation"], index=False)
        summaries.to_csv(artifacts["summary"], index=False)
        comparison.to_csv(artifacts["comparison"], index=False)
        deltas.to_csv(artifacts["deltas"], index=False)

    return {
        "participants": participants,
        "predictions": predictions,
        "evaluation": evaluation,
        "summaries": summaries,
        "comparison": comparison,
        "deltas": deltas,
        "artifacts": artifacts,
        "llm_provider": llm_provider,
        "tiers": list(selected),
    }

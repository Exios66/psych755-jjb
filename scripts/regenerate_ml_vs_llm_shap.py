"""Regenerate ML-vs-LLM F1 / SHAP / MAE comparison from real vLLM exports.

Replaces the mock-LLM stand-in in ``memos/feature_predictive_power_ml_llm.md``
and ``docs/ml_vs_llm.md`` with the committed vLLM v1/v2 row-level exports.

LLM side: ``exports/v1/*`` and ``exports/v2/*`` ``02_evaluation_rowlevel.csv``.
ML side: CV Random Forest + KNN on the full matched cohort (sibling data).
Surrogate SHAP: RF fit to the best non-collapsed model's outputs (DeepSeek v2)
at the transit tier, explained with TreeSHAP.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.personas import RESEARCH_TIERS
from ca_personas.scoring import ca_band
from ca_personas.shap_eval import (
    PALETTE,
    llm_metrics_with_f1,
    llm_surrogate_shap,
    load_participants_for_eval,
    ml_metrics_with_f1,
    plot_band_confusion,
    plot_f1_heatmap,
    plot_ml_vs_llm_bars,
    plot_ml_vs_llm_shap_compare,
    plot_shap_bar,
    run_ml_shap_bundle,
    tier_ablation_table,
)

EXPORTS = Path("exports")
ROWLEVEL = "tables/02_evaluation_rowlevel.csv"

# (display tag, family, version) -> glob pattern within exports/
LLM_PACKAGES = [
    ("DeepSeek v1", "deepseek", "v1"),
    ("DeepSeek v2", "deepseek", "v2"),
    ("Llama-3.1 v1", "llama31_8b_instruct", "v1"),
    ("Llama-3.1 v2", "llama31_8b_instruct", "v2"),
    ("Llama-3.2-3B v1", "llama32_3b_instruct", "v1"),
    ("Llama-3.2-3B v2", "llama32_3b_instruct", "v2"),
]
PRIMARY_LLM = "DeepSeek v2"
PRIMARY_ML = "random_forest"
ML_MODELS = ["random_forest", "knn"]

EVAL_COLS = [
    "tier",
    "agent",
    "agent_family",
    "participant_id",
    "gt_group_ca",
    "pred_group_ca",
    "gt_interpersonal_ca",
    "pred_interpersonal_ca",
    "band_match_group",
    "band_match_interpersonal",
    "band_distance_group",
    "band_distance_interpersonal",
]

BAND_CUTS = {"low": (6, 13), "moderate": (14, 19), "high": (20, 30)}
BAND_LABELS = ("low", "moderate", "high")


def resolve_package(tag: str, version: str) -> Path:
    matches = sorted(glob.glob(str(EXPORTS / version / f"psych755_vllm_{tag}*/")))
    if not matches:
        raise FileNotFoundError(f"No {version} package for tag {tag!r} in {EXPORTS}")
    return Path(matches[0])


def load_llm_evaluation(tag: str, family: str, version: str) -> pd.DataFrame:
    package = resolve_package(family, version)
    row = pd.read_csv(package / ROWLEVEL)
    eval_ = pd.DataFrame(
        {
            "tier": row["tier"],
            "agent": tag,
            "agent_family": "llm",
            "participant_id": row["participant_id"],
            "gt_group_ca": row["gt_group_ca"],
            "pred_group_ca": row["pred_group_ca"],
            "gt_interpersonal_ca": row["gt_interpersonal_ca"],
            "pred_interpersonal_ca": row["pred_interpersonal_ca"],
            "band_match_group": row["band_match_group"].astype(bool),
            "band_match_interpersonal": row["band_match_interpersonal"].astype(bool),
            "band_distance_group": row["band_distance_group"],
            "band_distance_interpersonal": row["band_distance_interpersonal"],
        }
    )
    return eval_


def band_score_profile(
    ml_preds: pd.DataFrame,
    llm_eval: pd.DataFrame,
    *,
    ml_model: str = PRIMARY_ML,
    llm_agent: str = PRIMARY_LLM,
) -> pd.DataFrame:
    """Per predicted band, summarize the actual predicted/GT score ranges.

    Answers: "when the model estimates a low/moderate/high band, what numeric
    scores does it actually output, how do those compare to the ground-truth
    scores, and how often is the band correct?".
    """
    rows: list[dict[str, Any]] = []

    ml = ml_preds[ml_preds["model"] == ml_model]
    for tier in RESEARCH_TIERS:
        for side in ("group", "interpersonal"):
            sub = ml[(ml["tier"] == tier) & (ml["side"] == side)].dropna(
                subset=["y_true", "y_pred"]
            )
            for band in BAND_LABELS:
                sel = sub[sub["pred_band"] == band]
                if sel.empty:
                    continue
                gt = sel["y_true"].astype(float)
                pr = sel["y_pred"].astype(float)
                rows.append(
                    {
                        "agent_family": "ml",
                        "model": ml_model,
                        "tier": tier,
                        "side": side,
                        "predicted_band": band,
                        "band_score_range": f"{BAND_CUTS[band][0]}–{BAND_CUTS[band][1]}",
                        "n": int(len(sel)),
                        "pred_score_mean": float(pr.mean()),
                        "pred_score_sd": float(pr.std(ddof=0)),
                        "pred_score_median": float(pr.median()),
                        "pred_score_min": float(pr.min()),
                        "pred_score_max": float(pr.max()),
                        "gt_score_mean": float(gt.mean()),
                        "gt_score_median": float(gt.median()),
                        "mae_mean": float((pr - gt).abs().mean()),
                        "band_precision": float(sel["band_match"].astype(bool).mean()),
                    }
                )

    for tier in RESEARCH_TIERS:
        sub = llm_eval[llm_eval["tier"] == tier]
        for side, gt_col, pred_col in (
            ("group", "gt_group_ca", "pred_group_ca"),
            ("interpersonal", "gt_interpersonal_ca", "pred_interpersonal_ca"),
        ):
            frame = sub.dropna(subset=[gt_col, pred_col]).copy()
            frame["_pred_band"] = [ca_band(int(round(v))) for v in frame[pred_col]]
            band_match_col = f"band_match_{side}"
            for band in BAND_LABELS:
                sel = frame[frame["_pred_band"] == band]
                if sel.empty:
                    continue
                gt = sel[gt_col].astype(float)
                pr = sel[pred_col].astype(float)
                rows.append(
                    {
                        "agent_family": "llm",
                        "model": llm_agent,
                        "tier": tier,
                        "side": side,
                        "predicted_band": band,
                        "band_score_range": f"{BAND_CUTS[band][0]}–{BAND_CUTS[band][1]}",
                        "n": int(len(sel)),
                        "pred_score_mean": float(pr.mean()),
                        "pred_score_sd": float(pr.std(ddof=0)),
                        "pred_score_median": float(pr.median()),
                        "pred_score_min": float(pr.min()),
                        "pred_score_max": float(pr.max()),
                        "gt_score_mean": float(gt.mean()),
                        "gt_score_median": float(gt.median()),
                        "mae_mean": float((pr - gt).abs().mean()),
                        "band_precision": float(sel[band_match_col].astype(bool).mean())
                        if band_match_col in sel.columns
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _ordinal_pairwise_auc(gt_band: pd.Series, y_pred: pd.Series) -> float:
    """Hand & Till pairwise AUC of a continuous score over ordered CA bands.

    Mean over band pairs (low, moderate), (low, high), (moderate, high) of the
    probability that the predicted score correctly orders the two classes
    (ties count 0.5). 0.5 = no ordinal discrimination.
    """
    pairs = [("low", "moderate"), ("low", "high"), ("moderate", "high")]
    aucs = []
    for lo, hi in pairs:
        lo_mask = gt_band == lo
        hi_mask = gt_band == hi
        if lo_mask.sum() == 0 or hi_mask.sum() == 0:
            continue
        x = y_pred[lo_mask].to_numpy(dtype=float)
        y = y_pred[hi_mask].to_numpy(dtype=float)
        concordant = 0
        total = len(x) * len(y)
        for a in x:
            concordant += int((a < y).sum()) + 0.5 * int((a == y).sum())
        aucs.append(concordant / total)
    return float(np.mean(aucs)) if aucs else float("nan")


def band_discrimination_table(
    ml_preds: pd.DataFrame,
    llm_evals: dict[str, pd.DataFrame],
    *,
    tiers: tuple[str, ...] = RESEARCH_TIERS,
    ml_models: tuple[str, ...] = ("random_forest", "knn"),
) -> pd.DataFrame:
    """Band discrimination metrics for ML vs live LLM models.

    Displays how well each model separates the PRCA classroom bands
    (low ≤13 / moderate 14–19 / high ≥20):
      - band accuracy and per-band F1 (low / moderate / high)
      - mean band distance (0 = correct, 1 = adjacent, 2 = opposite)
      - AUC_high_vs_low  : predicted scores separating GT high (≥20) from
                           GT low (≤13) respondents (moderates excluded)
      - AUC_ordinal_ovo  : Hand & Till multiclass AUC over all three bands
    """
    from sklearn.metrics import f1_score, roc_auc_score

    rows: list[dict[str, Any]] = []

    def _push(
        family: str,
        model: str,
        tier: str,
        side: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        band_distance: pd.Series | None,
    ) -> None:
        gt_band = pd.Series([ca_band(int(round(v))) for v in y_true], index=y_true.index)
        pr_band = pd.Series([ca_band(int(round(v))) for v in y_pred], index=y_pred.index)
        valid = gt_band.notna() & pr_band.notna()
        if not valid.all():
            gt_band = gt_band[valid]
            pr_band = pr_band[valid]
            y_true = y_true[valid]
            y_pred = y_pred[valid]
        per_f1 = f1_score(
            gt_band.tolist(),
            pr_band.tolist(),
            labels=list(BAND_LABELS),
            average=None,
            zero_division=0,
        )
        f1s = {f"f1_{b}": float(v) for b, v in zip(BAND_LABELS, per_f1, strict=True)}
        f1_macro = float(
            f1_score(
                gt_band.tolist(),
                pr_band.tolist(),
                labels=list(BAND_LABELS),
                average="macro",
                zero_division=0,
            )
        )
        band_acc = float((gt_band == pr_band).mean())
        mean_bd = (
            float(band_distance[valid].astype(float).mean())
            if band_distance is not None and len(band_distance) > 0
            else float("nan")
        )

        auc_hl = auc_ov = float("nan")
        extreme = gt_band.isin(["low", "high"])
        if extreme.sum() >= 10 and y_true[extreme].nunique() > 1:
            binary = (gt_band[extreme] == "high").astype(int)
            try:
                auc_hl = float(roc_auc_score(binary, y_pred[extreme]))
            except ValueError:
                auc_hl = float("nan")
        if gt_band.nunique() > 1:
            auc_ov = _ordinal_pairwise_auc(gt_band, y_pred)

        rows.append(
            {
                "agent_family": family,
                "model": model,
                "tier": tier,
                "side": side,
                "n": int(len(y_true)),
                "band_accuracy": band_acc,
                "f1_macro": f1_macro,
                **f1s,
                "mean_band_distance": mean_bd,
                "auc_high_vs_low": auc_hl,
                "auc_ordinal_ovo": auc_ov,
            }
        )

    for model in ml_models:
        sub = ml_preds[ml_preds["model"] == model]
        for tier in tiers:
            for side in ("group", "interpersonal"):
                frame = sub[(sub["tier"] == tier) & (sub["side"] == side)].dropna(
                    subset=["y_true", "y_pred"]
                )
                if frame.empty:
                    continue
                _push(
                    "ml",
                    model,
                    tier,
                    side,
                    frame["y_true"],
                    frame["y_pred"],
                    frame["band_distance"] if "band_distance" in frame else None,
                )

    for tag, eval_ in llm_evals.items():
        for tier in tiers:
            frame = eval_[eval_["tier"] == tier]
            for side in ("group", "interpersonal"):
                gt_col, pred_col = f"gt_{side}_ca", f"pred_{side}_ca"
                sub = frame.dropna(subset=[gt_col, pred_col])
                if sub.empty:
                    continue
                _push(
                    "llm",
                    tag,
                    tier,
                    side,
                    sub[gt_col],
                    sub[pred_col],
                    sub[f"band_distance_{side}"] if f"band_distance_{side}" in sub else None,
                )
    return pd.DataFrame(rows)


def plot_band_discrimination_auc(
    disc: pd.DataFrame,
    *,
    llm_agents: tuple[str, ...] = (PRIMARY_LLM,),
    ml_models: tuple[str, ...] = (PRIMARY_ML,),
    path: Path,
) -> Path:
    """Low-vs-high AUC by tier for RF vs the primary live LLM model."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, side in zip(axes, ("group", "interpersonal"), strict=True):
        for model, color, ls, marker in (
            *[(m, PALETTE["random_forest"], "-", "o") for m in ml_models],
            *[(m, PALETTE["llm"], "--", "s") for m in llm_agents],
        ):
            frame = disc[(disc["side"] == side) & (disc["model"] == model)]
            tiers = [t for t in RESEARCH_TIERS if t in set(frame["tier"])]
            ax.plot(
                tiers,
                [frame.loc[frame["tier"] == t, "auc_high_vs_low"].mean() for t in tiers],
                color=color,
                ls=ls,
                marker=marker,
                label=model,
            )
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{side} CA — low (≤13) vs high (≥20)", fontsize=12, pad=8)
        ax.set_ylim(0.3, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Low-vs-high ROC-AUC (predicted score)")
    fig.suptitle("Band discrimination by predicted score across tiers", fontsize=12, y=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_band_score_profiles(
    ml_preds: pd.DataFrame,
    llm_eval: pd.DataFrame,
    *,
    tier: str = "transit",
    side: str = "group",
    ml_model: str = PRIMARY_ML,
    llm_agent: str = PRIMARY_LLM,
    path: Path,
) -> Path:
    """Boxplots of the numeric scores a model outputs within each predicted band.

    Directly answers "what scores do the estimated low / moderate / high bands
    fall into" for RF vs the primary live LLM model, with the band cutoffs
    shaded so boundary behavior is visible.
    """
    from matplotlib import pyplot as plt

    gt_col = f"gt_{side}_ca"
    pred_col = f"pred_{side}_ca"

    ml_sub = ml_preds[
        (ml_preds["tier"] == tier)
        & (ml_preds["side"] == side)
        & (ml_preds["model"] == ml_model)
    ].dropna(subset=["y_true", "y_pred"])
    llm_sub = llm_eval[(llm_eval["tier"] == tier)].dropna(subset=[gt_col, pred_col])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    panels = (
        (axes[0], f"Random Forest", [(b, ml_sub[ml_sub["pred_band"] == b]["y_pred"]) for b in BAND_LABELS]),
        (axes[1], llm_agent, [(b, llm_sub[[ca_band(int(round(v))) == b for v in llm_sub[pred_col]]][pred_col]) for b in BAND_LABELS]),
    )
    for ax, title, series in panels:
        for lo, hi, color in ((6, 13, "#f5c6c9"), (14, 19, "#e8e0c8"), (20, 30, "#c9d8e8")):
            ax.axhspan(lo, hi, color=color, alpha=0.5, zorder=0)
        for band, scores in series:
            if scores.empty:
                ax.boxplot([], positions=[BAND_LABELS.index(band)], widths=0.55)
            else:
                ax.boxplot(
                    [scores.astype(float)],
                    positions=[BAND_LABELS.index(band)],
                    widths=0.55,
                    medianprops=dict(color="#C5050C", linewidth=1.6),
                    zorder=3,
                )
        ax.set_xticks(range(len(BAND_LABELS)))
        ax.set_xticklabels([f"{b}\n({BAND_CUTS[b][0]}–{BAND_CUTS[b][1]})" for b in BAND_LABELS])
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylim(6, 30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Predicted CA score")
    fig.suptitle(
        f"Predicted scores within each estimated band — {side} CA, {tier} tier",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="transit", help="SHAP tier (default transit)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--output-dir", default="outputs/shap_eval", help="Tables + figures output"
    )
    ap.add_argument("--memo-figures-dir", default="memos/figures")
    ap.add_argument(
        "--artifact-dir",
        default="artifacts/posit_full_cohort/ml_vs_llm",
        help="Committed table copy for manuscript renders (rule 4)",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    participants, cleaning_report = load_participants_for_eval(join_how="inner")
    print(f"analytic cohort n = {len(participants)}")

    # --- ML side (RF + KNN) with band F1 ---
    ml_preds, ml_metrics = ml_metrics_with_f1(
        participants, tiers=RESEARCH_TIERS, random_state=args.random_state
    )
    ml_metrics = ml_metrics[ml_metrics["model"].isin(ML_MODELS)]

    # --- LLM side from real vLLM exports ---
    llm_evals: dict[str, pd.DataFrame] = {}
    llm_metric_frames: list[pd.DataFrame] = []
    for tag, family, version in LLM_PACKAGES:
        eval_ = load_llm_evaluation(tag, family, version)
        llm_evals[tag] = eval_
        _, llm_metrics = llm_metrics_with_f1({"evaluation": eval_})
        llm_metric_frames.append(llm_metrics)
        print(f"  loaded {tag}: {eval_['participant_id'].nunique()} participants")

    llm_metrics_all = pd.concat(llm_metric_frames, ignore_index=True, sort=False)
    all_metrics = pd.concat([ml_metrics, llm_metrics_all], ignore_index=True, sort=False)
    ablation = tier_ablation_table(all_metrics)

    # --- ML TreeSHAP (predicting true CA) ---
    ml_shap = run_ml_shap_bundle(participants, tier=args.tier, random_state=args.random_state)

    # --- Surrogate SHAP on the best non-collapsed live model ---
    primary_eval = llm_evals[PRIMARY_LLM]
    llm_shap_group = llm_surrogate_shap(
        participants, primary_eval, tier=args.tier, side="group", random_state=args.random_state
    )
    llm_shap_inter = llm_surrogate_shap(
        participants, primary_eval, tier=args.tier, side="interpersonal", random_state=args.random_state
    )

    # --- Per-band score profiles (what numeric scores each estimated band maps to) ---
    band_profile = band_score_profile(ml_preds, primary_eval)

    # --- Band discrimination (accuracy, per-band F1, band distance, low-vs-high AUC) ---
    discrimination = band_discrimination_table(ml_preds, llm_evals)

    # --- Figures ---
    figure_paths: dict[str, Path] = {}
    figure_paths["shap_bar_ml_group"] = plot_shap_bar(
        ml_shap["gt_group_ca"]["raw_importance"],
        title=f"ML SHAP — predicting true Group CA ({args.tier} tier)",
        path=fig_dir / "fig_shap_bar_ml_group.png",
        color=PALETTE["random_forest"],
    )
    figure_paths["shap_bar_ml_inter"] = plot_shap_bar(
        ml_shap["gt_interpersonal_ca"]["raw_importance"],
        title=f"ML SHAP — predicting true Interpersonal CA ({args.tier} tier)",
        path=fig_dir / "fig_shap_bar_ml_interpersonal.png",
        color=PALETTE["knn"],
    )
    figure_paths["shap_bar_llm_group"] = plot_shap_bar(
        llm_shap_group["raw_importance"],
        title=f"Surrogate SHAP — features tracking {PRIMARY_LLM} Group CA ({args.tier})",
        path=fig_dir / "fig_shap_bar_llm_surrogate_group.png",
        color=PALETTE["llm"],
    )
    figure_paths["shap_compare"] = plot_ml_vs_llm_shap_compare(
        ml_shap["gt_group_ca"]["raw_importance"],
        llm_shap_group["raw_importance"],
        path=fig_dir / "fig_shap_ml_vs_llm_compare.png",
    )
    figure_paths["f1_heatmap"] = plot_f1_heatmap(
        all_metrics,
        path=fig_dir / "fig_f1_heatmap_group.png",
        title="Band classification macro-F1 by agent × tier (Group CA)",
    )

    # Bars compare RF vs the best live LLM only (not a blend of all models).
    bars_metrics = all_metrics[all_metrics["model"].isin([PRIMARY_ML, PRIMARY_LLM])]
    figure_paths["mae_bars"] = plot_ml_vs_llm_bars(
        bars_metrics,
        metric="mae",
        path=fig_dir / "fig_mae_ml_vs_llm.png",
        title="Mean absolute error by tier — RF vs " + PRIMARY_LLM,
        ylabel="MAE (PRCA points on 6–30)",
    )
    figure_paths["f1_bars"] = plot_ml_vs_llm_bars(
        bars_metrics,
        metric="f1_macro",
        path=fig_dir / "fig_f1_ml_vs_llm.png",
        title="Band macro-F1 by tier — RF vs " + PRIMARY_LLM,
        ylabel="Macro F1 (low / moderate / high)",
    )

    ablation_bars = ablation[ablation["model"].isin([PRIMARY_ML, PRIMARY_LLM])]
    figure_paths["ablation"] = plot_ablation_deltas(
        ablation_bars, path=fig_dir / "fig_tier_ablation_delta_mae.png"
    )

    rf_preds = ml_preds[
        (ml_preds["tier"] == args.tier)
        & (ml_preds["model"] == PRIMARY_ML)
        & (ml_preds["side"] == "group")
    ]
    if not rf_preds.empty:
        figure_paths["confusion_ml"] = plot_band_confusion(
            rf_preds["y_true"],
            rf_preds["y_pred"],
            title=f"RF band confusion — Group CA ({args.tier})",
            path=fig_dir / "fig_confusion_rf_group.png",
        )
    llm_tier = primary_eval[primary_eval["tier"] == args.tier].dropna(
        subset=["gt_group_ca", "pred_group_ca"]
    )
    if not llm_tier.empty:
        figure_paths["confusion_llm"] = plot_band_confusion(
            llm_tier["gt_group_ca"],
            llm_tier["pred_group_ca"],
            title=f"{PRIMARY_LLM} band confusion — Group CA ({args.tier})",
            path=fig_dir / "fig_confusion_llm_group.png",
        )
    figure_paths["band_score_profiles"] = plot_band_score_profiles(
        ml_preds,
        primary_eval,
        tier=args.tier,
        side="group",
        path=fig_dir / "fig_band_score_profiles.png",
    )
    figure_paths["band_discrimination_auc"] = plot_band_discrimination_auc(
        discrimination, path=fig_dir / "fig_band_discrimination_auc.png"
    )

    # --- Persist tables + card ---
    tables = {
        "metrics_ml_llm": all_metrics,
        "tier_ablation": ablation,
        "shap_ml_group_raw": ml_shap["gt_group_ca"]["raw_importance"],
        "shap_ml_inter_raw": ml_shap["gt_interpersonal_ca"]["raw_importance"],
        "shap_llm_surrogate_group_raw": llm_shap_group["raw_importance"],
        "shap_llm_surrogate_inter_raw": llm_shap_inter["raw_importance"],
        "band_score_profile_ml_llm": band_profile,
        "band_discrimination_ml_llm": discrimination,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
        print(f"  wrote {path}")

    rf_transit = all_metrics[
        (all_metrics["agent_family"] == "ml")
        & (all_metrics["model"] == PRIMARY_ML)
        & (all_metrics["tier"] == args.tier)
    ]
    llm_transit = all_metrics[
        (all_metrics["agent_family"] == "llm")
        & (all_metrics["model"] == PRIMARY_LLM)
        & (all_metrics["tier"] == args.tier)
    ]

    def _mean(frame: pd.DataFrame, col: str) -> float | None:
        return float(frame[col].mean()) if not frame.empty and col in frame else None

    card = {
        "research_question": (
            "Which demographic and behavioral features have the greatest predictive "
            "power for PRCA group/interpersonal CA under traditional ML and LLM "
            "persona agents, as measured by SHAP values and band-level F1?"
        ),
        "sample": {"n_analytic": int(len(participants)), "shap_tier": args.tier},
        "ml_transit": {
            "mae_mean": _mean(rf_transit, "mae"),
            "f1_macro_mean": _mean(rf_transit, "f1_macro"),
            "top_shap_group": ml_shap["gt_group_ca"]["raw_importance"]
            .head(5)[["raw_feature", "mean_abs_shap"]]
            .to_dict(orient="records"),
        },
        f"llm_{PRIMARY_LLM.lower().replace(' ', '_')}_transit": {
            "mae_mean": _mean(llm_transit, "mae"),
            "f1_macro_mean": _mean(llm_transit, "f1_macro"),
            "surrogate_r2_group": llm_shap_group["surrogate_r2"],
            "top_shap_group": llm_shap_group["raw_importance"]
            .head(5)[["raw_feature", "mean_abs_shap"]]
            .to_dict(orient="records"),
        },
        "cleaning_report": cleaning_report,
    }
    card_path = out / "shap_eval_results_card.json"
    card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    paths["results_card"] = card_path

    # --- Memo composite + copy figures into the memo directory ---
    composite = fig_dir / "fig_memo_feature_power_composite.png"
    _write_memo_composite(
        figure_paths["shap_compare"],
        figure_paths["f1_bars"],
        figure_paths["mae_bars"],
        figure_paths["ablation"],
        composite,
    )
    figure_paths["memo_composite"] = composite

    memo_figs = Path(args.memo_figures_dir)
    memo_figs.mkdir(parents=True, exist_ok=True)
    for fig in figure_paths.values():
        if fig.is_file():
            shutil.copy2(fig, memo_figs / fig.name)
            print(f"  copied figure -> {memo_figs / fig.name}")

    # --- Sync tables into the committed artifact dir (keyless manuscript renders) ---
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        if path.name.endswith(".csv"):
            shutil.copy2(path, artifact_dir / path.name)
            print(f"  synced table -> {artifact_dir / path.name}")


def plot_ablation_deltas(ablation: pd.DataFrame, *, path: Path) -> Path:
    """ΔMAE bars (Group CA) for RF vs the primary live LLM model."""
    from matplotlib import pyplot as plt

    frame = ablation.dropna(subset=["delta_mae_vs_prev"]).copy()
    frame = frame[frame["target"] == "gt_group_ca"]
    rf = frame[(frame["agent_family"] == "ml") & (frame["model"] == PRIMARY_ML)]
    llm = frame[(frame["agent_family"] == "llm") & (frame["model"] == PRIMARY_LLM)]
    tiers = [t for t in RESEARCH_TIERS if t != "demos"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(tiers))
    width = 0.36
    rf_vals = [
        float(rf.loc[rf["tier"] == t, "delta_mae_vs_prev"].mean()) if t in set(rf["tier"]) else 0.0
        for t in tiers
    ]
    llm_vals = [
        float(llm.loc[llm["tier"] == t, "delta_mae_vs_prev"].mean())
        if (not llm.empty and t in set(llm["tier"]))
        else 0.0
        for t in tiers
    ]
    ax.bar(x - width / 2, rf_vals, width, label="Random Forest ΔMAE", color=PALETTE["random_forest"])
    ax.bar(x + width / 2, llm_vals, width, label=PRIMARY_LLM + " ΔMAE", color=PALETTE["llm"])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{t}" for t in tiers])
    ax.legend(frameon=False)
    ax.set_title(
        "Incremental predictive gain from added feature groups (Group CA)",
        fontsize=12,
        pad=8,
    )
    ax.set_xlabel("Feature group added")
    ax.set_ylabel("ΔMAE vs previous tier (negative = better)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_memo_composite(
    p1: Path, p2: Path, p3: Path, p4: Path, dest: Path
) -> None:
    from matplotlib import image as mpimg
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax, path in zip(axes.ravel(), (p1, p2, p3, p4), strict=True):
        if path.is_file():
            ax.imshow(mpimg.imread(path))
        ax.axis("off")
    fig.suptitle(
        "Feature predictive power — SHAP, F1, and ML vs LLM tier gains",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

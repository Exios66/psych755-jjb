"""SHAP values, band F1, and ML-vs-LLM feature predictive-power evaluation.

Evaluates how demographic / employment / geo / transit features drive:

1. Traditional ML (Random Forest / KNN) predictions of PRCA subscales
2. LLM persona-agent predictions across the same cumulative information tiers

Artifacts land under ``outputs/shap_eval/`` (CSVs + figures).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from ca_personas.compare_agents import run_ml_vs_llm_comparison
from ca_personas.load import load_and_prepare, load_full_cohort
from ca_personas.ml_baseline import (
    TARGETS,
    TIER_FEATURES,
    available_feature_columns,
    baseline_models,
    choose_cv,
    make_preprocessor,
    prepare_modeling_frame,
    run_baselines_for_tier,
    split_feature_types,
)
from ca_personas.personas import RESEARCH_TIERS
from ca_personas.scoring import ca_band

try:
    import shap
except ImportError as exc:  # pragma: no cover - exercised when shap absent
    shap = None  # type: ignore[assignment]
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

BAND_LABELS = ("low", "moderate", "high")
UW_RED = "#C5050C"
UW_GRAY = "#333333"
UW_GOLD = "#9B870C"
PALETTE = {
    "random_forest": UW_RED,
    "knn": UW_GRAY,
    "llm": "#006658",
    "demos": "#4C4C4C",
    "employment": UW_GOLD,
    "geo": "#0077C8",
    "transit": UW_RED,
}


def require_shap() -> None:
    if shap is None:
        raise ImportError(
            "The 'shap' package is required for shap_eval. "
            "Install with: pip install 'shap>=0.44' "
            f"(original error: {_SHAP_IMPORT_ERROR})"
        )


def scores_to_bands(scores: Iterable[float]) -> list[str]:
    bands: list[str] = []
    for value in scores:
        band = ca_band(int(round(float(value))))
        if band is None:
            raise ValueError(f"Could not map score {value!r} to a CA band")
        bands.append(band)
    return bands


def band_f1_metrics(
    y_true_scores: Sequence[float] | np.ndarray,
    y_pred_scores: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Macro / weighted / per-band F1 for low/moderate/high PRCA bands."""
    y_true = scores_to_bands(y_true_scores)
    y_pred = scores_to_bands(y_pred_scores)
    macro = float(
        f1_score(y_true, y_pred, labels=list(BAND_LABELS), average="macro", zero_division=0)
    )
    weighted = float(
        f1_score(y_true, y_pred, labels=list(BAND_LABELS), average="weighted", zero_division=0)
    )
    per = f1_score(y_true, y_pred, labels=list(BAND_LABELS), average=None, zero_division=0)
    out = {"f1_macro": macro, "f1_weighted": weighted}
    for label, value in zip(BAND_LABELS, per, strict=True):
        out[f"f1_{label}"] = float(value)
    return out


def _stringify_categoricals(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    _, categorical = split_feature_types(feature_cols)
    for col in categorical:
        out[col] = out[col].astype("string").fillna("<NA>").astype(str)
    return out


def _encoded_to_raw_map(feature_names: list[str], raw_features: list[str]) -> dict[str, str]:
    """Map OneHot/scaled encoded names back to original column names."""
    mapping: dict[str, str] = {}
    for name in feature_names:
        # ColumnTransformer prefixes: num__Age, cat__Sex_Female, etc.
        body = name.split("__", 1)[-1]
        matched = None
        for raw in sorted(raw_features, key=len, reverse=True):
            if body == raw or body.startswith(f"{raw}_"):
                matched = raw
                break
        mapping[name] = matched or body
    return mapping


def fit_rf_pipeline(
    participants: pd.DataFrame,
    *,
    tier: str,
    target: str,
    random_state: int = 42,
    n_estimators: int = 300,
) -> tuple[Pipeline, pd.DataFrame, list[str]]:
    """Fit a Random Forest pipeline for SHAP on one target × tier."""
    model_df = prepare_modeling_frame(participants, tier=tier, targets=TARGETS)
    feature_cols = available_feature_columns(model_df, tier)
    X = _stringify_categoricals(model_df[feature_cols], feature_cols)
    y = model_df[target].astype(float)
    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_cols)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe, model_df, feature_cols


def compute_shap_values(
    pipe: Pipeline,
    model_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_samples: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """TreeExplainer SHAP on the RF; aggregate absolute SHAP to raw features."""
    require_shap()
    X = _stringify_categoricals(model_df[feature_cols], feature_cols)
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_enc = pre.transform(X)
    encoded_names = list(pre.get_feature_names_out())
    rng = np.random.default_rng(random_state)
    if X_enc.shape[0] > max_samples:
        idx = rng.choice(X_enc.shape[0], size=max_samples, replace=False)
        X_enc_s = X_enc[idx]
        ids = model_df["participant_id"].to_numpy()[idx]
    else:
        X_enc_s = X_enc
        ids = model_df["participant_id"].to_numpy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc_s)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_mat = np.asarray(shap_values, dtype=float)

    raw_map = _encoded_to_raw_map(encoded_names, feature_cols)
    # Per-encoded mean |SHAP|
    encoded_importance = pd.DataFrame(
        {
            "encoded_feature": encoded_names,
            "mean_abs_shap": np.abs(shap_mat).mean(axis=0),
            "mean_shap": shap_mat.mean(axis=0),
        }
    )
    encoded_importance["raw_feature"] = encoded_importance["encoded_feature"].map(raw_map)
    encoded_importance = encoded_importance.sort_values("mean_abs_shap", ascending=False)

    raw_importance = (
        encoded_importance.groupby("raw_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_long = pd.DataFrame(shap_mat, columns=encoded_names)
    shap_long.insert(0, "participant_id", ids)

    return {
        "shap_matrix": shap_mat,
        "encoded_names": encoded_names,
        "X_encoded": X_enc_s,
        "encoded_importance": encoded_importance.reset_index(drop=True),
        "raw_importance": raw_importance,
        "shap_long": shap_long,
        "expected_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "raw_map": raw_map,
    }


def ml_metrics_with_f1(
    participants: pd.DataFrame,
    *,
    tiers: Iterable[str] = RESEARCH_TIERS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RF/KNN CV metrics including band F1 for each tier × target."""
    pred_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    selected = [t for t in tiers if t in TIER_FEATURES]
    for tier in selected:
        preds, metrics = run_baselines_for_tier(
            participants, tier=tier, random_state=random_state
        )
        pred_frames.append(preds)
        for _, row in metrics.iterrows():
            subset = preds[
                (preds["model"] == row["model"])
                & (preds["target"] == row["target"])
                & (preds["tier"] == tier)
            ]
            f1s = band_f1_metrics(subset["y_true"], subset["y_pred"])
            metric_rows.append({**row.to_dict(), **f1s, "agent_family": "ml"})
    return pd.concat(pred_frames, ignore_index=True), pd.DataFrame(metric_rows)


def llm_metrics_with_f1(
    comparison: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract LLM rows from an ML-vs-LLM comparison and attach band F1."""
    evaluation = comparison["evaluation"]
    llm_eval = evaluation[evaluation["agent_family"] == "llm"].copy()
    rows: list[dict[str, Any]] = []
    for (tier, agent), frame in llm_eval.groupby(["tier", "agent"]):
        usable = frame.dropna(subset=["gt_group_ca", "pred_group_ca", "gt_interpersonal_ca", "pred_interpersonal_ca"])
        if usable.empty:
            continue
        for side, gt_col, pred_col in (
            ("group", "gt_group_ca", "pred_group_ca"),
            ("interpersonal", "gt_interpersonal_ca", "pred_interpersonal_ca"),
        ):
            f1s = band_f1_metrics(usable[gt_col], usable[pred_col])
            mae = float((usable[pred_col] - usable[gt_col]).abs().mean())
            exact = float(
                (usable[pred_col].round() == usable[gt_col].round()).mean()
            )
            band_acc = float(
                usable[f"band_match_{side}"].astype(bool).mean()
            ) if f"band_match_{side}" in usable.columns else float("nan")
            rows.append(
                {
                    "tier": tier,
                    "model": str(agent),
                    "target": f"gt_{side}_ca",
                    "n_samples": int(len(usable)),
                    "mae": mae,
                    "exact_acc": exact,
                    "band_acc": band_acc,
                    "agent_family": "llm",
                    **f1s,
                }
            )
    return llm_eval, pd.DataFrame(rows)


def tier_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute incremental MAE / F1 gains as tiers accumulate information."""
    order = [t for t in RESEARCH_TIERS if t in set(metrics["tier"])]
    rows: list[dict[str, Any]] = []
    for family, fam_frame in metrics.groupby("agent_family"):
        models = fam_frame["model"].unique()
        for model in models:
            for target in TARGETS:
                sub = fam_frame[
                    (fam_frame["model"] == model) & (fam_frame["target"] == target)
                ].set_index("tier")
                prev_mae = None
                prev_f1 = None
                for tier in order:
                    if tier not in sub.index:
                        continue
                    mae = float(sub.loc[tier, "mae"])
                    f1m = float(sub.loc[tier, "f1_macro"])
                    rows.append(
                        {
                            "agent_family": family,
                            "model": model,
                            "target": target,
                            "tier": tier,
                            "mae": mae,
                            "f1_macro": f1m,
                            "delta_mae_vs_prev": (
                                None if prev_mae is None else mae - prev_mae
                            ),
                            "delta_f1_vs_prev": (
                                None if prev_f1 is None else f1m - prev_f1
                            ),
                        }
                    )
                    prev_mae, prev_f1 = mae, f1m
    return pd.DataFrame(rows)


def llm_surrogate_shap(
    participants: pd.DataFrame,
    llm_evaluation: pd.DataFrame,
    *,
    tier: str = "transit",
    side: str = "group",
    random_state: int = 42,
    max_samples: int = 200,
) -> dict[str, Any]:
    """
    Surrogate SHAP for LLM predictions.

    Fits a Random Forest that predicts the LLM's predicted CA score from the
    same tabular features the LLM was shown (via the persona tier), then
    explains that surrogate with TreeSHAP. This attributes which profile
    features the LLM's outputs track.
    """
    require_shap()
    pred_col = f"pred_{side}_ca"
    llm_tier = llm_evaluation[llm_evaluation["tier"] == tier].dropna(subset=[pred_col])
    if llm_tier.empty:
        raise ValueError(f"No LLM predictions for tier={tier!r} side={side!r}")

    merged = participants.merge(
        llm_tier[["participant_id", pred_col]].drop_duplicates("participant_id"),
        on="participant_id",
        how="inner",
    )
    # Temporarily treat LLM prediction as the regression target.
    work = merged.copy()
    work["gt_group_ca"] = work.get("gt_group_ca", np.nan)
    work["gt_interpersonal_ca"] = work.get("gt_interpersonal_ca", np.nan)
    # prepare_modeling_frame requires both GT columns; keep them, but fit on LLM pred.
    model_df = prepare_modeling_frame(work, tier=tier, targets=TARGETS)
    model_df = model_df.merge(
        merged[["participant_id", pred_col]], on="participant_id", how="inner"
    )
    feature_cols = available_feature_columns(model_df, tier)
    X = _stringify_categoricals(model_df[feature_cols], feature_cols)
    y = model_df[pred_col].astype(float)
    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_cols)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_state,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    # In-sample R² of surrogate (how well tabular features recover LLM outputs).
    y_hat = pipe.predict(X)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    surrogate_r2 = 1.0 - ss_res / ss_tot

    shap_bundle = compute_shap_values(
        pipe,
        model_df,
        feature_cols,
        max_samples=max_samples,
        random_state=random_state,
    )
    shap_bundle["surrogate_r2"] = surrogate_r2
    shap_bundle["side"] = side
    shap_bundle["tier"] = tier
    shap_bundle["n"] = int(len(model_df))
    return shap_bundle


def run_ml_shap_bundle(
    participants: pd.DataFrame,
    *,
    tier: str = "transit",
    random_state: int = 42,
    max_samples: int = 200,
) -> dict[str, dict[str, Any]]:
    """SHAP for RF predicting each CA target at one tier."""
    out: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        pipe, model_df, feature_cols = fit_rf_pipeline(
            participants, tier=tier, target=target, random_state=random_state
        )
        bundle = compute_shap_values(
            pipe,
            model_df,
            feature_cols,
            max_samples=max_samples,
            random_state=random_state,
        )
        bundle["target"] = target
        bundle["tier"] = tier
        bundle["n"] = int(len(model_df))
        out[target] = bundle
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_shap_bar(
    raw_importance: pd.DataFrame,
    *,
    title: str,
    path: Path,
    top_n: int = 12,
    color: str = UW_RED,
) -> Path:
    frame = raw_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(frame["raw_feature"], frame["mean_abs_shap"], color=color)
    _style_axes(ax, title, xlabel="Mean |SHAP| (aggregated to raw feature)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_beeswarm(
    shap_mat: np.ndarray,
    encoded_names: list[str],
    X_encoded: np.ndarray,
    raw_map: dict[str, str],
    *,
    title: str,
    path: Path,
    top_n: int = 12,
) -> Path:
    """Custom beeswarm of top raw features (summed one-hot SHAP)."""
    # Aggregate encoded → raw for plotting by taking the encoded col with
    # largest mean |SHAP| within each raw feature as the display channel,
    # and summing SHAP contributions per raw feature for the x-axis.
    enc_imp = np.abs(shap_mat).mean(axis=0)
    raw_to_idxs: dict[str, list[int]] = {}
    for i, name in enumerate(encoded_names):
        raw_to_idxs.setdefault(raw_map[name], []).append(i)
    raw_mean_abs = {
        raw: float(enc_imp[idxs].sum()) for raw, idxs in raw_to_idxs.items()
    }
    top_raws = [
        r for r, _ in sorted(raw_mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    ][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(0)
    for yi, raw in enumerate(top_raws):
        idxs = raw_to_idxs[raw]
        shap_sum = shap_mat[:, idxs].sum(axis=1)
        # Color by the dominant encoded column's value (scaled rank).
        dom = idxs[int(np.argmax(enc_imp[idxs]))]
        feat_vals = X_encoded[:, dom]
        if np.nanstd(feat_vals) > 0:
            colors = (feat_vals - np.nanmin(feat_vals)) / (
                np.nanmax(feat_vals) - np.nanmin(feat_vals) + 1e-12
            )
        else:
            colors = np.zeros_like(feat_vals)
        jitter = rng.normal(0, 0.08, size=len(shap_sum))
        ax.scatter(
            shap_sum,
            np.full(len(shap_sum), yi) + jitter,
            c=colors,
            cmap="coolwarm",
            s=16,
            alpha=0.75,
            linewidths=0,
        )
    ax.set_yticks(range(len(top_raws)))
    ax.set_yticklabels(top_raws)
    ax.axvline(0, color="gray", linewidth=0.8)
    _style_axes(ax, title, xlabel="SHAP value (impact on model output)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_f1_heatmap(metrics: pd.DataFrame, *, path: Path, title: str) -> Path:
    """Heatmap of macro F1 by model × tier for one target panel pair."""
    frame = metrics.copy()
    frame["label"] = frame["agent_family"] + ":" + frame["model"].astype(str)
    # Focus on group CA for the primary heatmap; interpersonal saved separately if needed.
    group = frame[frame["target"] == "gt_group_ca"]
    if group.empty:
        group = frame
    pivot = group.pivot_table(index="label", columns="tier", values="f1_macro", aggfunc="mean")
    # Order columns by research tier sequence.
    cols = [t for t in RESEARCH_TIERS if t in pivot.columns] + [
        c for c in pivot.columns if c not in RESEARCH_TIERS
    ]
    pivot = pivot.reindex(columns=cols)

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.45 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro F1 (CA bands)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ml_vs_llm_bars(
    metrics: pd.DataFrame,
    *,
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
) -> Path:
    """Grouped bars of a metric by tier for best ML RF vs LLM."""
    frame = metrics.copy()
    rf = frame[(frame["agent_family"] == "ml") & (frame["model"] == "random_forest")]
    llm = frame[frame["agent_family"] == "llm"]
    # Average across targets for a compact view.
    rf_tier = rf.groupby("tier")[metric].mean()
    llm_tier = llm.groupby("tier")[metric].mean()
    tiers = [t for t in RESEARCH_TIERS if t in set(rf_tier.index) | set(llm_tier.index)]
    x = np.arange(len(tiers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x - width / 2,
        [rf_tier.get(t, np.nan) for t in tiers],
        width,
        label="Random Forest",
        color=PALETTE["random_forest"],
    )
    ax.bar(
        x + width / 2,
        [llm_tier.get(t, np.nan) for t in tiers],
        width,
        label="LLM persona agent",
        color=PALETTE["llm"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(frameon=False)
    _style_axes(ax, title, xlabel="Persona / feature tier", ylabel=ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation_deltas(ablation: pd.DataFrame, *, path: Path) -> Path:
    """Bar chart of ΔMAE when adding each successive tier (RF vs LLM)."""
    frame = ablation.dropna(subset=["delta_mae_vs_prev"]).copy()
    frame = frame[frame["target"] == "gt_group_ca"]
    # Prefer RF + one LLM agent.
    rf = frame[(frame["agent_family"] == "ml") & (frame["model"] == "random_forest")]
    llm = frame[frame["agent_family"] == "llm"]
    if not llm.empty:
        llm_model = llm["model"].iloc[0]
        llm = llm[llm["model"] == llm_model]
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
    ax.bar(x - width / 2, rf_vals, width, label="Random Forest ΔMAE", color=UW_RED)
    ax.bar(x + width / 2, llm_vals, width, label="LLM ΔMAE", color=PALETTE["llm"])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{t}" for t in tiers])
    ax.legend(frameon=False)
    _style_axes(
        ax,
        "Incremental predictive gain from added feature groups (Group CA)",
        xlabel="Feature group added",
        ylabel="ΔMAE vs previous tier (negative = better)",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ml_vs_llm_shap_compare(
    ml_raw: pd.DataFrame,
    llm_raw: pd.DataFrame,
    *,
    path: Path,
    top_n: int = 10,
) -> Path:
    """Side-by-side mean |SHAP| for ML target model vs LLM surrogate."""
    ml = ml_raw.set_index("raw_feature")["mean_abs_shap"]
    ll = llm_raw.set_index("raw_feature")["mean_abs_shap"]
    features = list(
        pd.concat([ml, ll], axis=1)
        .fillna(0)
        .max(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    y = np.arange(len(features))
    ax.barh(y - 0.18, [ml.get(f, 0.0) for f in features], 0.35, label="ML → true CA", color=UW_RED)
    ax.barh(
        y + 0.18,
        [ll.get(f, 0.0) for f in features],
        0.35,
        label="Surrogate of LLM output",
        color=PALETTE["llm"],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.legend(frameon=False)
    _style_axes(
        ax,
        "Feature importance: ML (true CA) vs LLM-surrogate SHAP",
        xlabel="Mean |SHAP|",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_band_confusion(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str,
    path: Path,
) -> Path:
    true_b = scores_to_bands(y_true)
    pred_b = scores_to_bands(y_pred)
    table = pd.crosstab(pd.Series(true_b, name="True"), pd.Series(pred_b, name="Pred"))
    table = table.reindex(index=list(BAND_LABELS), columns=list(BAND_LABELS), fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(table.to_numpy(), cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(BAND_LABELS)
    ax.set_yticklabels(BAND_LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(table.to_numpy()[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted band")
    ax.set_ylabel("Ground-truth band")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_participants_for_eval(
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    *,
    join_how: str = "inner",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load cleaned analytic sample (full cohort when multiple Prolific paths)."""
    if prolific_paths is not None and len(list(prolific_paths)) > 1:
        return load_full_cohort(
            prolific_paths=prolific_paths,
            qualtrics_path=qualtrics_path,
            join_how=join_how,
        )
    if prolific_paths is None or qualtrics_path is None:
        from ca_personas.paths import default_prolific_paths, default_qualtrics_path

        prolific_paths = prolific_paths or default_prolific_paths()
        qualtrics_path = qualtrics_path or default_qualtrics_path()
    paths = list(prolific_paths)
    if len(paths) > 1:
        return load_full_cohort(
            prolific_paths=paths,
            qualtrics_path=qualtrics_path,
            join_how=join_how,
        )
    participants = load_and_prepare(paths[0], qualtrics_path, how=join_how, clean=True)
    return participants, None


def run_shap_feature_eval(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    llm_provider: str = "mock",
    llm_model: str | None = None,
    shap_tier: str = "transit",
    output_dir: str | Path = "outputs/shap_eval",
    figures_dir: str | Path | None = None,
    random_state: int = 42,
    max_shap_samples: int = 200,
) -> dict[str, Any]:
    """
    End-to-end ML + LLM feature predictive-power evaluation with SHAP and F1.

    Returns a dict with dataframes, figure paths, and a results card.
    """
    require_shap()
    try:
        from ca_personas.viz_style import apply_memo_style

        apply_memo_style()
    except Exception:
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "font.size": 10,
            }
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir) if figures_dir is not None else out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    participants, cleaning_report = load_participants_for_eval(
        prolific_paths, qualtrics_path, join_how=join_how
    )

    # --- ML metrics + F1 ---
    ml_preds, ml_metrics = ml_metrics_with_f1(participants, random_state=random_state)

    # --- LLM comparison (mock by default) ---
    # compare_agents expects a single prolific path; stack if needed.
    from ca_personas.paths import default_prolific_paths, default_qualtrics_path

    prol_list = list(prolific_paths) if prolific_paths is not None else default_prolific_paths()
    qual = Path(qualtrics_path) if qualtrics_path is not None else default_qualtrics_path()
    if len(prol_list) == 1:
        prolific_arg: Path | list[Path] = Path(prol_list[0])
    else:
        from ca_personas.load import load_prolific

        stacked = load_prolific(prol_list, wave_labels=("A", "B"))
        tmp = out / "_prolific_stacked.csv"
        stacked.rename(columns={"participant_id": "Participant id"}).to_csv(tmp, index=False)
        prolific_arg = tmp

    comparison = run_ml_vs_llm_comparison(
        prolific_arg,
        qual,
        tiers=RESEARCH_TIERS,
        llm_provider=llm_provider,
        llm_model=llm_model,
        join_how=join_how,
        output_dir=out / "ml_vs_llm",
    )
    llm_eval, llm_metrics = llm_metrics_with_f1(comparison)
    all_metrics = pd.concat([ml_metrics, llm_metrics], ignore_index=True, sort=False)
    ablation = tier_ablation_table(all_metrics)

    # --- SHAP: ML predicting true CA ---
    ml_shap = run_ml_shap_bundle(
        participants,
        tier=shap_tier,
        random_state=random_state,
        max_samples=max_shap_samples,
    )

    # --- SHAP: surrogate of LLM outputs ---
    llm_shap_group = llm_surrogate_shap(
        participants,
        llm_eval,
        tier=shap_tier,
        side="group",
        random_state=random_state,
        max_samples=max_shap_samples,
    )
    llm_shap_inter = llm_surrogate_shap(
        participants,
        llm_eval,
        tier=shap_tier,
        side="interpersonal",
        random_state=random_state,
        max_samples=max_shap_samples,
    )

    # --- Figures ---
    figure_paths: dict[str, Path] = {}
    figure_paths["shap_bar_ml_group"] = plot_shap_bar(
        ml_shap["gt_group_ca"]["raw_importance"],
        title=f"ML SHAP — predicting true Group CA ({shap_tier} tier)",
        path=fig_dir / "fig_shap_bar_ml_group.png",
        color=UW_RED,
    )
    figure_paths["shap_bar_ml_inter"] = plot_shap_bar(
        ml_shap["gt_interpersonal_ca"]["raw_importance"],
        title=f"ML SHAP — predicting true Interpersonal CA ({shap_tier} tier)",
        path=fig_dir / "fig_shap_bar_ml_interpersonal.png",
        color=UW_GRAY,
    )
    figure_paths["shap_beeswarm_ml_group"] = plot_shap_beeswarm(
        ml_shap["gt_group_ca"]["shap_matrix"],
        ml_shap["gt_group_ca"]["encoded_names"],
        ml_shap["gt_group_ca"]["X_encoded"],
        ml_shap["gt_group_ca"]["raw_map"],
        title=f"ML SHAP beeswarm — Group CA ({shap_tier})",
        path=fig_dir / "fig_shap_beeswarm_ml_group.png",
    )
    figure_paths["shap_bar_llm_group"] = plot_shap_bar(
        llm_shap_group["raw_importance"],
        title=f"LLM-surrogate SHAP — features tracking LLM Group CA ({shap_tier})",
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
    figure_paths["mae_bars"] = plot_ml_vs_llm_bars(
        all_metrics,
        metric="mae",
        path=fig_dir / "fig_mae_ml_vs_llm.png",
        title="Mean absolute error by tier — RF vs LLM",
        ylabel="MAE (PRCA points on 6–30)",
    )
    figure_paths["f1_bars"] = plot_ml_vs_llm_bars(
        all_metrics,
        metric="f1_macro",
        path=fig_dir / "fig_f1_ml_vs_llm.png",
        title="Band macro-F1 by tier — RF vs LLM",
        ylabel="Macro F1 (low / moderate / high)",
    )
    figure_paths["ablation"] = plot_ablation_deltas(
        ablation, path=fig_dir / "fig_tier_ablation_delta_mae.png"
    )

    # Confusion matrices for RF + LLM at richest tier.
    rf_preds = ml_preds[
        (ml_preds["tier"] == shap_tier)
        & (ml_preds["model"] == "random_forest")
        & (ml_preds["side"] == "group")
    ]
    if not rf_preds.empty:
        figure_paths["confusion_ml"] = plot_band_confusion(
            rf_preds["y_true"],
            rf_preds["y_pred"],
            title=f"RF band confusion — Group CA ({shap_tier})",
            path=fig_dir / "fig_confusion_rf_group.png",
        )
    llm_tier = llm_eval[llm_eval["tier"] == shap_tier].dropna(
        subset=["gt_group_ca", "pred_group_ca"]
    )
    if not llm_tier.empty:
        figure_paths["confusion_llm"] = plot_band_confusion(
            llm_tier["gt_group_ca"],
            llm_tier["pred_group_ca"],
            title=f"LLM band confusion — Group CA ({shap_tier})",
            path=fig_dir / "fig_confusion_llm_group.png",
        )

    # --- Persist tables ---
    paths: dict[str, Path] = {f"fig_{k}": v for k, v in figure_paths.items()}
    tables = {
        "ml_predictions": ml_preds,
        "metrics_ml_llm": all_metrics,
        "tier_ablation": ablation,
        "shap_ml_group_raw": ml_shap["gt_group_ca"]["raw_importance"],
        "shap_ml_group_encoded": ml_shap["gt_group_ca"]["encoded_importance"],
        "shap_ml_inter_raw": ml_shap["gt_interpersonal_ca"]["raw_importance"],
        "shap_llm_surrogate_group_raw": llm_shap_group["raw_importance"],
        "shap_llm_surrogate_inter_raw": llm_shap_inter["raw_importance"],
    }
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    # Results card
    rf_transit = all_metrics[
        (all_metrics["agent_family"] == "ml")
        & (all_metrics["model"] == "random_forest")
        & (all_metrics["tier"] == shap_tier)
    ]
    llm_transit = all_metrics[
        (all_metrics["agent_family"] == "llm") & (all_metrics["tier"] == shap_tier)
    ]

    def _mean_metric(frame: pd.DataFrame, col: str) -> float | None:
        if frame.empty or col not in frame.columns:
            return None
        return float(frame[col].mean())

    card = {
        "research_question": (
            "Which demographic and behavioral features have the greatest predictive "
            "power for PRCA group/interpersonal CA under traditional ML and LLM "
            "persona agents, as measured by SHAP values and band-level F1?"
        ),
        "sample": {
            "n_analytic": int(len(participants)),
            "shap_tier": shap_tier,
            "llm_provider": llm_provider,
        },
        "ml_transit": {
            "mae_mean": _mean_metric(rf_transit, "mae"),
            "f1_macro_mean": _mean_metric(rf_transit, "f1_macro"),
            "top_shap_group": ml_shap["gt_group_ca"]["raw_importance"]
            .head(5)[["raw_feature", "mean_abs_shap"]]
            .to_dict(orient="records"),
        },
        "llm_transit": {
            "mae_mean": _mean_metric(llm_transit, "mae"),
            "f1_macro_mean": _mean_metric(llm_transit, "f1_macro"),
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

    # Composite memo figure (2×2)
    memo_fig = fig_dir / "fig_memo_feature_power_composite.png"
    _write_memo_composite(
        figure_paths["shap_compare"],
        figure_paths["f1_bars"],
        figure_paths["mae_bars"],
        figure_paths["ablation"],
        memo_fig,
    )
    paths["fig_memo_composite"] = memo_fig
    figure_paths["memo_composite"] = memo_fig

    return {
        "participants": participants,
        "ml_predictions": ml_preds,
        "metrics": all_metrics,
        "ablation": ablation,
        "ml_shap": ml_shap,
        "llm_shap_group": llm_shap_group,
        "llm_shap_inter": llm_shap_inter,
        "comparison": comparison,
        "paths": paths,
        "figure_paths": figure_paths,
        "results_card": card,
        "output_dir": out,
        "figures_dir": fig_dir,
    }


def _write_memo_composite(
    p1: Path, p2: Path, p3: Path, p4: Path, dest: Path
) -> None:
    """Tile four existing PNGs into one memo figure."""
    import matplotlib.image as mpimg

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

#!/usr/bin/env python3
"""Regenerate all Posit-site memo/docs PNGs in APA grayscale format.

Reads seeded full-cohort artifacts under ``outputs/`` and
``artifacts/posit_full_cohort/secondary_results.json``. Does not re-fit models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ca_personas.apa_plotting import (  # noqa: E402
    apa_axes,
    apply_apa_style,
    grouped_bars,
    horizontal_bars,
    prevalence_bars,
    roc_curve_apa,
    scatter_by_group,
)

DOCS_FIG = ROOT / "docs" / "figures"
MEMOS_FIG = ROOT / "memos" / "figures"
SECONDARY = json.loads(
    (ROOT / "artifacts" / "posit_full_cohort" / "secondary_results.json").read_text(
        encoding="utf-8"
    )
)
PREVALENCE = float(SECONDARY["prevalence"])


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _auc_lookup() -> dict[str, float]:
    return {r["spec_key"]: float(r["roc_auc"]) for r in SECONDARY["covariate_comparison"]}


def _family_auc(family: str) -> float:
    summary_path = ROOT / "outputs" / "transit_covariate_rf" / family / f"{family}_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cv = summary.get("cv_metrics") or {}
        if "roc_auc" in cv:
            return float(cv["roc_auc"])
    return float(_auc_lookup()[family])


def fig_ml_baseline_mae() -> None:
    metrics = pd.read_csv(ROOT / "outputs" / "ml_baseline" / "ml_baseline_metrics.csv")
    rf = metrics[
        (metrics["model"] == "random_forest") & (metrics["target"] == "gt_group_ca")
    ].copy()
    order = ["demos", "employment", "geo", "transit"]
    rf["tier"] = pd.Categorical(rf["tier"], categories=order, ordered=True)
    rf = rf.sort_values("tier")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    grouped_bars(
        ax,
        rf["tier"].tolist(),
        {"RF group CA": rf["mae"].tolist()},
        ylabel="Mean absolute error",
        xlabel="Persona information tier",
        ylim=(0, 7),
    )
    _save(fig, DOCS_FIG / "ml_baseline_mae_group.png")


def fig_feature_importance() -> None:
    top = pd.read_csv(
        ROOT / "outputs" / "feature_importance" / "top_predictive_features.csv"
    ).head(10)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    horizontal_bars(
        ax,
        top["feature"].tolist(),
        top["permutation_importance_mean"].tolist(),
        xlabel="Permutation importance (mean)",
        highlight_index=0,
        xlim=(0, float(top["permutation_importance_mean"].max()) * 1.18),
    )
    _save(fig, DOCS_FIG / "feature_importance_top.png")


def fig_eda_employment() -> None:
    ca = pd.read_csv(ROOT / "outputs" / "eda" / "ca_by_group.csv")
    emp = ca[ca["group_col"] == "Employment status"].copy()
    order = ["Full-Time", "Part-Time", "Other"]
    emp["group"] = pd.Categorical(emp["group"], categories=order, ordered=True)
    emp = emp.sort_values("group")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    grouped_bars(
        ax,
        emp["group"].astype(str).tolist(),
        {
            "Group CA": emp["mean_group_ca"].tolist(),
            "Interpersonal CA": emp["mean_interpersonal_ca"].tolist(),
        },
        ylabel="Mean PRCA score (6–30)",
        xlabel="Employment status",
        ylim=(0, 22),
    )
    _save(fig, DOCS_FIG / "eda_ca_by_employment.png")


def _ca_means_figure(path: Path) -> None:
    g = SECONDARY["transit_ca"]["group"]
    i = SECONDARY["transit_ca"]["interpersonal"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    grouped_bars(
        ax,
        ["Group CA", "Interpersonal CA"],
        {
            f"Regular (n={g['n_regular']})": [g["mean_regular"], i["mean_regular"]],
            f"Not regular (n={g['n_not_regular']})": [
                g["mean_not_regular"],
                i["mean_not_regular"],
            ],
        },
        ylabel="Mean PRCA score (6–30)",
        ylim=(0, 22),
    )
    _save(fig, path)


def _ca_dist_figure(path: Path) -> None:
    labeled = ROOT / "outputs" / "transit_ca" / "transit_ca_labeled_sample.csv"
    df = pd.read_csv(labeled)
    regular = df["regular_transit"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, col, name in [
        (axes[0], "gt_group_ca", "Group CA"),
        (axes[1], "gt_interpersonal_ca", "Interpersonal CA"),
    ]:
        for mask, style, lab in [
            (regular, {"color": "#111111", "ls": "-"}, "Regular"),
            (~regular, {"color": "#666666", "ls": "--"}, "Not regular"),
        ]:
            vals = df.loc[mask, col]
            ax.hist(
                vals,
                bins=np.arange(5.5, 31.5, 1),
                histtype="step",
                linewidth=1.6,
                label=f"{lab} (n={int(mask.sum())})",
                **style,
            )
        ax.set_xlabel(name)
        ax.set_xlim(6, 30)
        apa_axes(ax)
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_transit_ca() -> None:
    _ca_means_figure(MEMOS_FIG / "transit_riders_ca_memo.png")
    _ca_means_figure(DOCS_FIG / "transit_riders_ca_means.png")
    _ca_dist_figure(DOCS_FIG / "transit_ca_distributions.png")
    _ca_dist_figure(DOCS_FIG / "ca_dist_by_transit.png")


def fig_geo() -> None:
    frame = pd.read_csv(
        ROOT / "outputs" / "geo_transit_rf" / "geo_transit_modeling_frame.csv"
    )
    roc = pd.read_csv(ROOT / "outputs" / "geo_transit_rf" / "geo_transit_rf_roc_curve.csv")
    auc = float(SECONDARY["geo_rf"]["roc_auc"])
    country = float(SECONDARY["geo_rf"]["country_only_roc_auc"])

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    scatter_by_group(
        ax,
        frame["LocationLongitude"],
        frame["LocationLatitude"],
        frame["regular_transit"].astype(bool),
        xlabel="Longitude",
        ylabel="Latitude",
        positive_label="Regular (weekly+)",
        negative_label="Not regular",
    )
    _save(fig, DOCS_FIG / "geo_scatter_latlon_by_transit.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    horizontal_bars(
        ax,
        ["Lat/long RF", "Country-only RF", "Chance"],
        [auc, country, 0.5],
        xlabel="Cross-validated ROC-AUC",
        highlight_index=0,
        vlines={"Chance = .500": 0.5},
        xlim=(0.46, 0.62),
    )
    _save(fig, DOCS_FIG / "geo_rf_auc_vs_baselines.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    scatter_by_group(
        axes[0],
        frame["LocationLongitude"],
        frame["LocationLatitude"],
        frame["regular_transit"].astype(bool),
        xlabel="Longitude",
        ylabel="Latitude",
        positive_label="Regular (weekly+)",
        negative_label="Not regular",
    )
    roc_curve_apa(
        axes[1],
        roc["fpr"],
        roc["tpr"],
        auc=auc,
        label="Lat/lon RF",
        annotate=f"Country-only RF AUC = {country:.3f}",
    )
    _save(fig, MEMOS_FIG / "geo_predicts_transit_memo.png")


def fig_ca_rf() -> None:
    roc = pd.read_csv(ROOT / "outputs" / "ca_transit_rf" / "ca_transit_rf_roc_curve.csv")
    g = SECONDARY["transit_ca"]["group"]
    i = SECONDARY["transit_ca"]["interpersonal"]
    auc = float(SECONDARY["ca_rf"]["roc_auc"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    horizontal_bars(
        ax,
        [
            "Group + interpersonal",
            "Group only",
            "Interpersonal only",
            "Chance",
        ],
        [
            auc,
            float(SECONDARY["ca_rf"]["group_only"]),
            float(SECONDARY["ca_rf"]["interpersonal_only"]),
            0.5,
        ],
        xlabel="Cross-validated ROC-AUC",
        highlight_index=0,
        vlines={"Chance = .500": 0.5},
        xlim=(0.45, 0.65),
    )
    _save(fig, DOCS_FIG / "ca_rf_auc_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    grouped_bars(
        axes[0],
        ["Group CA", "Interpersonal CA"],
        {
            "Regular transit": [g["mean_regular"], i["mean_regular"]],
            "Not regular": [g["mean_not_regular"], i["mean_not_regular"]],
        },
        ylabel="Mean PRCA score (6–30)",
        ylim=(0, 22),
    )
    roc_curve_apa(axes[1], roc["fpr"], roc["tpr"], auc=auc, label="CA Random Forest")
    _save(fig, MEMOS_FIG / "ca_predicts_transit_memo.png")


def fig_covariate_followups() -> None:
    aucs = _auc_lookup()
    order = [
        ("q28_days", "Q28 days"),
        ("q27_q28", "Q27+Q28"),
        ("mobility_bundle", "Mobility bundle"),
        ("rideshare", "Rideshare Q28+Q29"),
        ("car_access", "Car access"),
        ("ca_benchmark", "CA scores"),
        ("q27_intensity", "Q27 intensity"),
        ("geo_benchmark", "Geo lat/long"),
        ("employment", "Employment"),
        ("chance", "Chance"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    horizontal_bars(
        ax,
        [lab for _, lab in order],
        [aucs[k] for k, _ in order],
        xlabel="Cross-validated ROC-AUC",
        highlight_index=0,
        vlines={"Chance=.500": 0.5, "Geo=.551": 0.551, "CA=.590": 0.590},
        xlim=(0.45, 0.85),
    )
    _save(fig, MEMOS_FIG / "transit_covariate_followups_memo.png")

    q_order = [
        ("q28_days", "Q28 ride-share days"),
        ("q27_q28", "Q27 + Q28 joint"),
        ("rideshare", "Q28 + Q29 rideshare family"),
        ("ca_benchmark", "Group + interpersonal CA"),
        ("q27_intensity", "Q27 transit intensity"),
        ("geo_benchmark", "Lat/long geo"),
        ("chance", "Chance"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    horizontal_bars(
        ax,
        [lab for _, lab in q_order],
        [aucs[k] for k, _ in q_order],
        xlabel="Cross-validated ROC-AUC",
        highlight_index=0,
        vlines={"Chance = .500": 0.5, "Geo = .551": 0.551, "CA = .590": 0.590},
        xlim=(0.45, 0.84),
    )
    _save(fig, MEMOS_FIG / "q27_q28_predicts_transit_comparison.png")


def _family_memo_figure(
    *,
    family: str,
    feature_for_prev: str,
    level_order: list[str],
    out_name: str,
    roc_label: str,
) -> None:
    base = ROOT / "outputs" / "transit_covariate_rf" / family
    assoc = pd.read_csv(base / f"{family}_associations.csv")
    roc = pd.read_csv(base / f"{family}_roc_curve.csv")
    auc = _family_auc(family)

    sub = assoc[assoc["feature"] == feature_for_prev].copy()
    sub = sub[sub["level"].isin(level_order)].copy()
    sub["level"] = pd.Categorical(sub["level"], categories=level_order, ordered=True)
    sub = sub.sort_values("level")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    prevalence_bars(
        axes[0],
        sub["level"].astype(str).tolist(),
        sub["pct_regular"].tolist(),
        sub["n"].tolist(),
        sample_prevalence=PREVALENCE,
    )
    roc_curve_apa(axes[1], roc["fpr"], roc["tpr"], auc=auc, label=roc_label)
    _save(fig, MEMOS_FIG / out_name)


def fig_family_memos() -> None:
    order28 = [
        "Never",
        "0-1 days a month",
        "2-4 days a month",
        "4-8 days a month",
        "8 or more days a month",
    ]
    order27 = [
        "1-2 rides in a typical day",
        "3-4 rides in a typical day",
        "5-6 rides in a typical day",
        "7 or more rides in a typical day",
    ]
    order_emp = ["Full-Time", "Part-Time", "Other"]
    order_car = ["No", "Yes"]

    _family_memo_figure(
        family="rideshare",
        feature_for_prev="Q28",
        level_order=order28,
        out_name="rideshare_predicts_transit_memo.png",
        roc_label="Ride-share RF",
    )
    _family_memo_figure(
        family="employment",
        feature_for_prev="Employment status",
        level_order=order_emp,
        out_name="employment_predicts_transit_memo.png",
        roc_label="Employment RF",
    )
    _family_memo_figure(
        family="car_access",
        feature_for_prev="Q21",
        level_order=order_car,
        out_name="car_access_predicts_transit_memo.png",
        roc_label="Car-access RF",
    )
    _family_memo_figure(
        family="q28_days",
        feature_for_prev="Q28",
        level_order=order28,
        out_name="q28_days_predicts_transit_memo.png",
        roc_label="Q28 RF",
    )
    _family_memo_figure(
        family="q27_intensity",
        feature_for_prev="Q27",
        level_order=order27,
        out_name="q27_intensity_predicts_transit_memo.png",
        roc_label="Q27 RF",
    )
    _family_memo_figure(
        family="mobility_bundle",
        feature_for_prev="Q28",
        level_order=order28,
        out_name="mobility_bundle_predicts_transit_memo.png",
        roc_label="Mobility-bundle RF",
    )

    q27 = pd.DataFrame(SECONDARY["q27_prevalence"])
    q28 = pd.DataFrame(SECONDARY["q28_prevalence"])
    q27["level"] = pd.Categorical(q27["level"], categories=order27, ordered=True)
    q28["level"] = pd.Categorical(q28["level"], categories=order28, ordered=True)
    q27 = q27.sort_values("level")
    q28 = q28.sort_values("level")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    prevalence_bars(
        axes[0],
        q27["level"].astype(str).tolist(),
        q27["prevalence"].tolist(),
        q27["n"].tolist(),
        sample_prevalence=PREVALENCE,
    )
    axes[0].text(
        0.98, 0.98, "Q27 (n = 239)", transform=axes[0].transAxes, ha="right", va="top", fontsize=9
    )
    prevalence_bars(
        axes[1],
        q28["level"].astype(str).tolist(),
        q28["prevalence"].tolist(),
        q28["n"].tolist(),
        sample_prevalence=PREVALENCE,
    )
    axes[1].text(
        0.98, 0.98, "Q28 (n = 241)", transform=axes[1].transAxes, ha="right", va="top", fontsize=9
    )
    _save(fig, MEMOS_FIG / "q27_q28_predicts_transit_memo.png")


def main() -> None:
    apply_apa_style()
    fig_ml_baseline_mae()
    fig_feature_importance()
    fig_eda_employment()
    fig_transit_ca()
    fig_geo()
    fig_ca_rf()
    fig_covariate_followups()
    fig_family_memos()
    print("done")


if __name__ == "__main__":
    main()

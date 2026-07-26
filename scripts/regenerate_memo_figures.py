#!/usr/bin/env python3
"""Re-fit models and regenerate *color* memo exploratory figures.

Canonical Posit-site PNGs (APA grayscale) are owned by
``scripts/regenerate_apa_site_figures.py``, which redraws from committed
``artifacts/posit_full_cohort/secondary_results.json`` and ``outputs/``
without re-fitting. Prefer that script after secondary sync; use this
script only when you intentionally want color memo-style re-fits on the
full sibling cohort.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_personas.ca_transit_rf import run_ca_transit_rf_analysis
from ca_personas.followup_experiments import (
    plot_experiment_memo_figure,
    plot_overview_figure,
    run_all_followup_experiments,
)
from ca_personas.geo_transit_rf import run_geo_transit_rf_analysis
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import run_transit_ca_analysis
from ca_personas.transit_covariate_rf import (
    plot_comparison_memo_figure,
    plot_family_memo_figure,
    run_all_followup_analyses,
)
from ca_personas.viz_style import (
    ACCENT,
    INK,
    MUTED,
    PRIMARY,
    SUCCESS,
    WARN,
    add_title_block,
    apply_memo_style,
    plot_auc_bars,
    plot_roc_curve,
    save_figure,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[1]
MEMO_FIG = ROOT / "memos" / "figures"
DOCS_FIG = ROOT / "docs" / "figures"


def plot_geo_memo(analysis: dict, path: Path) -> Path:
    apply_memo_style()
    frame = analysis["frame"]
    roc = analysis["roc"]
    auc = float(analysis["summary"]["cv_metrics"]["roc_auc"])
    country_auc = analysis["summary"]["baselines"].get("country_only_roc_auc")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    reg = frame.loc[frame["y"] == 1]
    non = frame.loc[frame["y"] == 0]
    axes[0].scatter(
        non["LocationLongitude"],
        non["LocationLatitude"],
        s=28,
        c=MUTED,
        alpha=0.55,
        edgecolors="white",
        linewidths=0.3,
        label=f"Not regular (n={len(non)})",
        zorder=2,
    )
    axes[0].scatter(
        reg["LocationLongitude"],
        reg["LocationLatitude"],
        s=36,
        c=ACCENT,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        label=f"Regular weekly+ (n={len(reg)})",
        zorder=3,
    )
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Survey geolocation by transit use", loc="left", fontsize=11, pad=8)
    style_axes(axes[0], grid="both")
    axes[0].legend(loc="best", fontsize=8.5)

    plot_roc_curve(axes[1], roc, auc=auc, title="Lat/long Random Forest", color=PRIMARY)
    if country_auc is not None:
        axes[1].text(
            0.98,
            0.18,
            f"Country-only RF AUC = {country_auc:.3f}",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=WARN,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFFFF", "edgecolor": "#E6E9ED"},
        )
    add_title_block(
        fig,
        "Does survey geography predict regular transit?",
        f"n={len(frame)}  ·  lat/long CV ROC-AUC={auc:.3f}  ·  modest signal near chance",
    )
    fig.subplots_adjust(top=0.80, left=0.08, right=0.98, bottom=0.12, wspace=0.25)
    return save_figure(fig, path)


def plot_ca_memo(analysis: dict, path: Path) -> Path:
    apply_memo_style()
    frame = analysis["frame"]
    roc = analysis["roc"]
    auc = float(analysis["summary"]["cv_metrics"]["roc_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), gridspec_kw={"width_ratios": [1.1, 1.0]})
    for y_val, color, label in [
        (0, MUTED, "Not regular"),
        (1, ACCENT, "Regular weekly+"),
    ]:
        sub = frame.loc[frame["y"] == y_val]
        axes[0].scatter(
            sub["gt_group_ca"],
            sub["gt_interpersonal_ca"],
            s=34,
            c=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.35,
            label=f"{label} (n={len(sub)})",
        )
    axes[0].plot([6, 30], [6, 30], color="#C5CCD4", ls="--", lw=1)
    axes[0].set_xlim(6, 30)
    axes[0].set_ylim(6, 30)
    axes[0].set_xlabel("Group CA (6–30)")
    axes[0].set_ylabel("Interpersonal CA (6–30)")
    axes[0].set_title("CA scores by regular transit", loc="left", fontsize=11, pad=8)
    style_axes(axes[0], grid="both")
    axes[0].legend(loc="upper left", fontsize=8.5)
    plot_roc_curve(axes[1], roc, auc=auc, title="CA → transit Random Forest", color=PRIMARY)

    # Nested bars inset via comparison if available
    metrics = analysis.get("metrics_table")
    add_title_block(
        fig,
        "Do group & interpersonal CA predict regular transit?",
        f"n={len(frame)}  ·  joint RF AUC={auc:.3f}  ·  modest discrimination above chance",
    )
    fig.subplots_adjust(top=0.80, left=0.08, right=0.98, bottom=0.12, wspace=0.25)
    return save_figure(fig, path)


def plot_transit_ca_memo(analysis: dict, path: Path) -> Path:
    apply_memo_style()
    by_q26 = analysis["by_q26"].copy()
    comp = analysis["comparisons"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))
    order = [
        "Never",
        "0-1 days a month",
        "2-4 days a month",
        "4-8 days a month",
        "8 or more days a month",
    ]
    short = {
        "Never": "Never",
        "0-1 days a month": "0–1 d/mo",
        "2-4 days a month": "2–4 d/mo",
        "4-8 days a month": "4–8 d/mo",
        "8 or more days a month": "8+ d/mo",
    }
    # Wide-form descriptives from transit_ca.by_q26
    d = by_q26.copy()
    gcol = next(c for c in ("mean_group_ca", "mean_gt_group_ca", "gt_group_ca") if c in d.columns)
    icol = next(
        c
        for c in ("mean_interpersonal_ca", "mean_gt_interpersonal_ca", "gt_interpersonal_ca")
        if c in d.columns
    )
    d["Q26"] = pd.Categorical(d["Q26"], categories=order, ordered=True)
    d = d.dropna(subset=["Q26"]).sort_values("Q26")
    x = np.arange(len(d))
    w = 0.38
    axes[0].bar(x - w / 2, d[gcol], width=w, color=ACCENT, edgecolor="white", label="Group CA")
    axes[0].bar(x + w / 2, d[icol], width=w, color=PRIMARY, edgecolor="white", label="Interpersonal CA")
    if "n" in d.columns:
        for xi, (_, row) in enumerate(d.iterrows()):
            axes[0].text(
                xi,
                max(float(row[gcol]), float(row[icol])) + 0.35,
                f"n={int(row['n'])}",
                ha="center",
                fontsize=7.5,
                color=MUTED,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([short.get(str(v), str(v)) for v in d["Q26"]], rotation=20, ha="right")
    ymax = max(18.0, float(d[[gcol, icol]].max().max() + 2.5))

    axes[0].set_ylabel("Mean PRCA (6–30)")
    axes[0].set_title("Mean CA by Q26 ridership", loc="left", fontsize=11, pad=8)
    axes[0].set_ylim(0, ymax)
    style_axes(axes[0], grid="y")
    axes[0].legend(loc="upper right", fontsize=8.5)

    scores = ["gt_group_ca", "gt_interpersonal_ca"]
    labels = ["Group CA", "Interpersonal CA"]
    means_reg, means_not, deltas = [], [], []
    for s in scores:
        row = comp.loc[comp["score"] == s].iloc[0]
        means_reg.append(float(row["mean_regular"]))
        means_not.append(float(row["mean_not_regular"]))
        deltas.append(float(row["diff_regular_minus_not_regular"]))
    x2 = np.arange(len(scores))
    w = 0.38
    axes[1].bar(x2 - w / 2, means_reg, width=w, color=ACCENT, edgecolor="white", label="Regular (weekly+)")
    axes[1].bar(x2 + w / 2, means_not, width=w, color=MUTED, edgecolor="white", label="Not regular")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean PRCA (6–30)")
    axes[1].set_title("Regular vs not-regular (Welch contrast)", loc="left", fontsize=11, pad=8)
    axes[1].set_ylim(0, max(means_reg + means_not) + 3.2)
    for i, delta in enumerate(deltas):
        axes[1].text(
            i,
            max(means_reg[i], means_not[i]) + 0.5,
            f"Δ = {delta:+.2f}",
            ha="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
    style_axes(axes[1], grid="y")
    axes[1].legend(loc="upper right", fontsize=8.5)

    add_title_block(
        fig,
        "Do regular transit riders differ in communication apprehension?",
        "Matched cohort  ·  weekly+ Q26 vs rest  ·  lower CA among regular riders",
    )
    fig.subplots_adjust(top=0.80, left=0.08, right=0.98, bottom=0.16, wspace=0.28)
    return save_figure(fig, path)


def main() -> None:
    MEMO_FIG.mkdir(parents=True, exist_ok=True)
    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    # Refuse excerpt fallback — color memo figures must use File A/B/C.
    participants, _ = load_full_cohort(join_how="inner", allow_excerpt_fallback=False)

    print("Wave-2 follow-ups...")
    bundle = run_all_followup_experiments(
        participants, n_perm_repeats=12, n_boot=800, random_state=42
    )
    for key, analysis in bundle["analyses"].items():
        plot_experiment_memo_figure(analysis, output_path=MEMO_FIG / f"{key}_followup_memo.png")
        print(" ", key)
    plot_overview_figure(bundle["overview"], output_path=MEMO_FIG / "followup_experiments_overview.png")

    print("Covariate families...")
    cov = run_all_followup_analyses(
        participants,
        spec_keys=[
            "car_access",
            "employment",
            "rideshare",
            "q27_intensity",
            "q28_days",
            "q27_q28",
            "mobility_bundle",
        ],
        n_perm_repeats=8,
        random_state=42,
    )
    for key, analysis in cov["analyses"].items():
        plot_family_memo_figure(analysis, output_path=MEMO_FIG / f"{key}_predicts_transit_memo.png")
        print(" ", key)
    plot_comparison_memo_figure(
        cov["comparison"],
        output_path=MEMO_FIG / "transit_covariate_followups_memo.png",
    )
    plot_comparison_memo_figure(
        cov["comparison"],
        output_path=MEMO_FIG / "q27_q28_predicts_transit_comparison.png",
        title="Q27 / Q28 and mobility benchmarks — CV ROC-AUC",
    )

    print("Geo / CA / transit→CA memos...")
    geo = run_geo_transit_rf_analysis(participants, n_perm_repeats=10, random_state=42)
    plot_geo_memo(geo, MEMO_FIG / "geo_predicts_transit_memo.png")
    # docs copies
    plot_geo_memo(geo, DOCS_FIG / "geo_scatter_latlon_by_transit.png")
    apply_memo_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    rows = [
        ("Lat/long RF", geo["summary"]["cv_metrics"]["roc_auc"], PRIMARY),
        ("Country-only RF", geo["summary"]["baselines"]["country_only_roc_auc"], WARN),
        ("Chance", 0.5, MUTED),
    ]
    plot_auc_bars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        colors=[r[2] for r in rows],
        benchmarks=("chance",),
        xlim=(0.45, 0.70),
        highlight_best=False,
    )
    add_title_block(fig, "Geography → regular transit", "Lat/long vs country-only vs chance")
    fig.subplots_adjust(top=0.82, left=0.28, right=0.96, bottom=0.14)
    save_figure(fig, DOCS_FIG / "geo_rf_auc_vs_baselines.png")

    ca = run_ca_transit_rf_analysis(participants, n_perm_repeats=10, random_state=42)
    plot_ca_memo(ca, MEMO_FIG / "ca_predicts_transit_memo.png")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    mt = ca["metrics_table"]
    # Prefer readable labels
    label_map = {
        "random_forest_ca": "Group + interpersonal CA",
        "random_forest_group_only": "Group CA only",
        "random_forest_interpersonal_only": "Interpersonal CA only",
        "prevalence_prob": "Chance / prevalence",
        "majority_class": "Majority class",
    }
    if "model" in mt.columns:
        labs = [label_map.get(m, m) for m in mt["model"]]
        plot_auc_bars(
            ax,
            labs,
            mt["roc_auc"].tolist(),
            colors=[PRIMARY if "ca" in str(m) or "group" in str(m) or "interpersonal" in str(m) else WARN for m in mt["model"]],
            benchmarks=("chance", "geo"),
            xlim=(0.45, 0.75),
        )
    add_title_block(fig, "CA scores → regular transit", "Nested RF ablations and null baselines")
    fig.subplots_adjust(top=0.82, left=0.34, right=0.96, bottom=0.14)
    save_figure(fig, DOCS_FIG / "ca_rf_auc_comparison.png")

    tca = run_transit_ca_analysis(participants, n_boot=800, random_state=42)
    plot_transit_ca_memo(tca, MEMO_FIG / "transit_riders_ca_memo.png")
    # docs distribution copy from memo left panel already covered; write means figure
    plot_transit_ca_memo(tca, DOCS_FIG / "transit_riders_ca_means.png")

    print("Done.")


if __name__ == "__main__":
    main()

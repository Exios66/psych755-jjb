"""APA-aligned figure helpers built on seaborn (matplotlib backend).

Figures are rendered with **seaborn** — which sits on matplotlib — and then
styled through ``apply_apa_style()`` to align closely with APA 7 figure
conventions for student manuscripts:

- sans-serif labels (8–14 pt)
- no chartjunk (no background grid, no in-figure title)
- high-contrast grayscale / hatch patterns (print-safe)
- clearly labeled axes; legend only when needed
- captions live outside the image (Quarto ``fig-cap``)
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ca_personas.viz_style import fit_text_within_axes


# Print-safe grayscale fills (dark → light). Distinguishable when photocopied.
APA_FILLS = ["#111111", "#666666", "#999999", "#CCCCCC", "#EEEEEE"]
APA_HATCHES = ["", "///", "\\\\\\", "xxx", "...", "---"]


def apply_apa_style() -> None:
    """Set seaborn's white theme, then layer APA-compatible matplotlib rcParams."""
    sns.set_theme(style="white")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 0,  # titles belong in Quarto captions
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apa_axes(ax: plt.Axes) -> plt.Axes:
    """Strip non-APA chrome from an axes."""
    ax.set_title("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def grouped_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    ylabel: str,
    xlabel: str = "",
    annotate: bool = True,
    ylim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Grouped bar chart (seaborn ``barplot``) with grayscale fills + hatches."""
    apply_apa_style()
    cats = list(categories)
    names = list(series.keys())
    n = max(len(names), 1)
    width = min(0.8 / n, 0.35)

    rows = [
        (cat, name, float(v))
        for name in names
        for cat, v in zip(cats, series[name])
    ]
    df = pd.DataFrame(rows, columns=["category", "series", "value"])

    sns.barplot(
        data=df,
        x="category",
        y="value",
        hue="series",
        hue_order=names,
        order=cats,
        ax=ax,
        palette=APA_FILLS[: len(names)],
        edgecolor="black",
        linewidth=0.8,
        width=width * n,
        errorbar=None,
        legend=False,
    )
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(APA_HATCHES[(i % n) % len(APA_HATCHES)])

    if annotate:
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif series:
        ymax = max(float(np.nanmax(v)) for v in series.values())
        headroom = 1.25 if annotate else 1.18
        ax.set_ylim(0, ymax * headroom)
    if len(names) > 1:
        handles = [
            Patch(
                facecolor=APA_FILLS[i % len(APA_FILLS)],
                hatch=APA_HATCHES[i % len(APA_HATCHES)],
                edgecolor="black",
                linewidth=0.8,
                label=name,
            )
            for i, name in enumerate(names)
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right")
    return apa_axes(ax)


def horizontal_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    xlabel: str,
    highlight_index: int | None = None,
    vlines: dict[str, float] | None = None,
    xlim: tuple[float, float] | None = None,
    legend_loc: str | None = "lower right",
) -> plt.Axes:
    """Horizontal bar chart (seaborn ``barplot``) with value labels.

    ``legend_loc="below"`` places the vline legend under the axes (reserved by
    tight_layout) so no legend box covers bars or value labels.
    """
    apply_apa_style()
    labels_l = list(labels)
    vals = np.asarray(values, dtype=float)
    df = pd.DataFrame({"label": labels_l, "value": vals})

    sns.barplot(
        data=df,
        x="value",
        y="label",
        order=labels_l,
        ax=ax,
        orient="h",
        color="#777777",
        edgecolor="black",
        linewidth=0.8,
        errorbar=None,
        legend=False,
    )
    for i, bar in enumerate(ax.patches):
        if highlight_index is not None and i == highlight_index:
            bar.set_facecolor("#111111")
            bar.set_hatch("")
        else:
            bar.set_facecolor("#777777")
            bar.set_hatch("///")

    ax.set_yticks(np.arange(len(labels_l)))
    ax.set_yticklabels(labels_l)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmax = max(vals) + max(0.15, max(vals) * 0.03)
        ax.set_xlim(0, xmax)
    for bar, v in zip(ax.patches, vals):
        ax.text(
            bar.get_width() + (0.008 if xlim is None else (xlim[1] - xlim[0]) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=8,
        )
    if vlines:
        styles = ["--", ":", "-."]
        for j, (name, xval) in enumerate(vlines.items()):
            ax.axvline(xval, color="#333333", ls=styles[j % len(styles)], lw=1.0, label=name)
        if legend_loc:
            if legend_loc == "below":
                ax.legend(
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=len(vlines),
                    fontsize=8,
                )
            else:
                ax.legend(frameon=False, loc=legend_loc, fontsize=8)
    ax.invert_yaxis()
    return apa_axes(ax)


def scatter_identity(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    lims: tuple[float, float] = (6, 30),
) -> plt.Axes:
    """Scatter with identity line (seaborn ``scatterplot``); grayscale markers."""
    apply_apa_style()
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        ax=ax,
        s=28,
        color="#333333",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.65,
        legend=False,
    )
    ax.plot(lims, lims, ls="--", color="#666666", lw=1.0, label="Identity line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return apa_axes(ax)


def scatter_by_group(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    group: Sequence[bool | int],
    *,
    xlabel: str,
    ylabel: str,
    positive_label: str = "Regular",
    negative_label: str = "Not regular",
) -> plt.Axes:
    """Two-group scatter (seaborn) with marker shape (print-safe grayscale)."""
    apply_apa_style()
    g = np.asarray(group).astype(bool)
    df = pd.DataFrame(
        {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "group": np.where(g, positive_label, negative_label),
        }
    )
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="group",
        hue_order=[negative_label, positive_label],
        style="group",
        markers={negative_label: "o", positive_label: "^"},
        palette={negative_label: "#777777", positive_label: "#111111"},
        s=30,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.75,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    counts = {positive_label: int(g.sum()), negative_label: int((~g).sum())}
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="#777777",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markersize=7,
            label=f"{negative_label} (n={counts[negative_label]})",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            ls="",
            color="#111111",
            markerfacecolor="#111111",
            markeredgecolor="black",
            markersize=8,
            label=f"{positive_label} (n={counts[positive_label]})",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)
    return apa_axes(ax)


def roc_curve_apa(
    ax: plt.Axes,
    fpr: Sequence[float],
    tpr: Sequence[float],
    *,
    auc: float,
    label: str | None = None,
    chance: bool = True,
    annotate: str | None = None,
) -> plt.Axes:
    """APA-friendly ROC curve (seaborn ``lineplot`` + chance diagonal)."""
    apply_apa_style()
    curve_label = label or f"Model (AUC = {auc:.3f})"
    if "AUC" not in curve_label:
        curve_label = f"{curve_label} (AUC = {auc:.3f})"
    df = pd.DataFrame(
        {"fpr": np.asarray(fpr, dtype=float), "tpr": np.asarray(tpr, dtype=float)}
    )
    sns.lineplot(data=df, x="fpr", y="tpr", ax=ax, color="#111111", lw=1.6, legend=False)
    if chance:
        ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0, label="Chance (AUC = .500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    if annotate:
        ax.text(
            0.03,
            0.97,
            annotate,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    ax.legend([curve_label, "Chance (AUC = .500)"] if chance else [curve_label], loc="lower right", fontsize=8)
    return apa_axes(ax)


def prevalence_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    prevalences: Sequence[float],
    ns: Sequence[int],
    *,
    sample_prevalence: float | None = None,
    xlabel: str = "Proportion regular transit (weekly+)",
) -> plt.Axes:
    """Horizontal prevalence bars (seaborn) with n in tick labels."""
    apply_apa_style()
    labels_l = list(labels)
    vals = np.asarray(prevalences, dtype=float)
    df = pd.DataFrame({"label": labels_l, "value": vals})
    sns.barplot(
        data=df,
        x="value",
        y="label",
        order=labels_l,
        ax=ax,
        orient="h",
        color="#444444",
        edgecolor="black",
        linewidth=0.8,
        errorbar=None,
        legend=False,
    )
    ax.set_yticks(np.arange(len(labels_l)))
    ax.set_yticklabels(
        [f"{lab} (n={int(n)})" for lab, n in zip(labels_l, ns)],
        fontsize=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    value_texts = []
    for bar, n in zip(ax.patches, ns):
        v = float(bar.get_width())
        label = f"{v:.1%}  (n={n})"
        value_texts.append(
            ax.text(
                min(v + 0.02, 0.92),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
                clip_on=True,
            )
        )
    if sample_prevalence is not None:
        ax.axvline(
            sample_prevalence,
            color="black",
            ls="--",
            lw=1.0,
            label=f"Sample prevalence = {sample_prevalence:.3f}",
        )
        ax.legend(
            frameon=False,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
        )
    ax.invert_yaxis()
    fit_text_within_axes(ax, value_texts)
    return apa_axes(ax)

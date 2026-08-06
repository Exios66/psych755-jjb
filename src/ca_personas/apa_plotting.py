"""APA-aligned matplotlib helpers for Quarto / Posit figures.

Follows APA 7 figure conventions for student manuscripts:
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


# Print-safe grayscale fills (dark → light). Distinguishable when photocopied.
APA_FILLS = ["#111111", "#666666", "#999999", "#CCCCCC", "#EEEEEE"]
APA_HATCHES = ["", "///", "\\\\\\", "xxx", "...", "---"]


def apply_apa_style() -> None:
    """Set global matplotlib rcParams for APA-like figures."""
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
    """Grouped bar chart with grayscale fills + hatches (APA-friendly)."""
    apply_apa_style()
    cats = list(categories)
    names = list(series.keys())
    x = np.arange(len(cats), dtype=float)
    n = max(len(names), 1)
    width = min(0.8 / n, 0.35)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    for i, name in enumerate(names):
        vals = np.asarray(series[name], dtype=float)
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.95,
            label=name,
            color=APA_FILLS[i % len(APA_FILLS)],
            edgecolor="black",
            linewidth=0.8,
            hatch=APA_HATCHES[i % len(APA_HATCHES)],
        )
        if annotate:
            for rect, v in zip(bars, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
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
        ax.legend(frameon=False, loc="upper right")
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
    """Horizontal bar chart with value labels (APA-friendly grayscale).

    ``legend_loc="below"`` places the vline legend under the axes (reserved by
    tight_layout) so no legend box covers bars or value labels.
    """
    apply_apa_style()
    y = np.arange(len(labels))
    vals = np.asarray(values, dtype=float)
    colors = []
    hatches = []
    for i in range(len(labels)):
        if highlight_index is not None and i == highlight_index:
            colors.append("#111111")
            hatches.append("")
        else:
            colors.append("#777777")
            hatches.append("///")
    bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmax = max(vals) + max(0.15, max(vals) * 0.03)
        ax.set_xlim(0, xmax)
    for yi, v in zip(y, vals):
        ax.text(v + (0.008 if xlim is None else (xlim[1] - xlim[0]) * 0.01), yi, f"{v:.3f}", va="center", fontsize=8)
    if vlines:
        styles = ["--", ":", "-."]
        for j, (name, xval) in enumerate(vlines.items()):
            ax.axvline(xval, color="#333333", ls=styles[j % len(styles)], lw=1.0, label=name)
        if legend_loc:
            if legend_loc == "below":
                ax.legend(
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.09),
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
    """Scatter with identity line; grayscale markers."""
    apply_apa_style()
    ax.scatter(x, y, s=28, c="#333333", edgecolors="black", linewidths=0.3, alpha=0.65)
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
    """Two-group scatter with marker shape (APA grayscale, print-safe)."""
    apply_apa_style()
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    g = np.asarray(group).astype(bool)
    ax.scatter(
        x_arr[~g],
        y_arr[~g],
        s=28,
        c="#777777",
        marker="o",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label=f"{negative_label} (n={int((~g).sum())})",
    )
    ax.scatter(
        x_arr[g],
        y_arr[g],
        s=36,
        c="#111111",
        marker="^",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.8,
        label=f"{positive_label} (n={int(g.sum())})",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best", fontsize=8)
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
    """APA-friendly ROC curve (black solid + dashed chance diagonal)."""
    apply_apa_style()
    curve_label = label or f"Model (AUC = {auc:.3f})"
    if "AUC" not in curve_label:
        curve_label = f"{curve_label} (AUC = {auc:.3f})"
    ax.plot(fpr, tpr, color="#111111", lw=1.6, label=curve_label)
    if chance:
        ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0, label="Chance (AUC = .500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    if annotate:
        ax.text(
            0.98,
            0.02,
            annotate,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
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
    """Horizontal prevalence bars with n in tick labels."""
    apply_apa_style()
    y = np.arange(len(labels))
    vals = np.asarray(prevalences, dtype=float)
    ax.barh(y, vals, color="#444444", edgecolor="black", height=0.7, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{lab} (n={int(n)})" for lab, n in zip(labels, ns)],
        fontsize=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.10)
    for yi, v in zip(y, vals):
        label = f"{v:.1%}"
        if ns is not None:
            label = f"{v:.1%}  (n={ns[yi]})"
        ax.text(min(float(v) + 0.02, 0.92), yi, label, va="center", fontsize=8)
    if sample_prevalence is not None:
        ax.axvline(
            sample_prevalence,
            color="black",
            ls="--",
            lw=1.0,
            label=f"Sample prevalence = {sample_prevalence:.3f}",
        )
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.invert_yaxis()
    return apa_axes(ax)

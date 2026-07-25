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
        ax.set_ylim(0, ymax * 1.18)
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
) -> plt.Axes:
    """Horizontal bar chart with value labels (APA-friendly grayscale)."""
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
    for yi, v in zip(y, vals):
        ax.text(v + (0.008 if xlim is None else (xlim[1] - xlim[0]) * 0.01), yi, f"{v:.3f}", va="center", fontsize=8)
    if vlines:
        styles = ["--", ":", "-."]
        for j, (name, xval) in enumerate(vlines.items()):
            ax.axvline(xval, color="#333333", ls=styles[j % len(styles)], lw=1.0, label=name)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
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

"""Shared publication styling for PSYCH 755 research memo figures.

Design goals: clean academic look, high information density, consistent
benchmarks, and readable type — without generic “AI dashboard” aesthetics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Palette (ink / slate / cardinal accent — UW-adjacent, print-safe)
# ---------------------------------------------------------------------------

INK = "#1B1F24"
MUTED = "#5C6570"
GRID = "#E6E9ED"
PAPER = "#FFFFFF"
PANEL = "#F7F8FA"

PRIMARY = "#1F4E5F"  # deep slate-teal (main series)
PRIMARY_SOFT = "#3D7A8C"
PRIMARY_PALE = "#A8C5CE"
ACCENT = "#C5050C"  # UW cardinal — sparse emphasis only
WARM = "#8B5E3C"  # benchmark / reference
WARM_SOFT = "#C4A484"
COOL_MID = "#2F6F7E"
SUCCESS = "#2F6B4F"  # strong discrimination
WARN = "#A67C52"  # chance / null

BENCHMARKS = {
    "chance": {"value": 0.500, "label": "Chance 0.50", "color": MUTED, "ls": (0, (3, 2))},
    "geo": {"value": 0.551, "label": "Geo 0.55", "color": WARM, "ls": (0, (1, 2))},
    "ca": {"value": 0.590, "label": "CA 0.59", "color": "#6B3E26", "ls": (0, (4, 2, 1, 2))},
    "q28": {"value": 0.762, "label": "Q28 0.76", "color": PRIMARY, "ls": "solid"},
}

FAMILY_COLORS = {
    "q28": PRIMARY,
    "rideshare": PRIMARY,
    "mobility": PRIMARY,
    "car": COOL_MID,
    "country": "#3D5A80",
    "geo": WARM,
    "ca": "#6B3E26",
    "demographics": "#4A6670",
    "employment": "#7A8B99",
    "chance": WARN,
    "other": COOL_MID,
}


def apply_memo_style() -> None:
    """Configure matplotlib rcParams for memo / manuscript figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.edgecolor": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Source Sans 3", "DejaVu Sans", "Noto Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C5CCD4",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "text.color": INK,
        }
    )


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 220, tight_rect: tuple[float, float, float, float] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        # Reserve headroom for suptitle title blocks; otherwise tight_layout()
        # clobbers manual figure margins and the title/subtitle collide with axes.
        rect = (0, 0, 1, 0.88) if fig._suptitle is not None else None
    else:
        rect = tight_rect
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=PAPER, edgecolor="none")
    plt.close(fig)
    return out


def style_axes(ax: Axes, *, grid: str = "x") -> None:
    ax.set_axisbelow(True)
    if grid == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
    elif grid == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.xaxis.grid(False)
    elif grid == "both":
        ax.grid(True, color=GRID, linewidth=0.8)
    else:
        ax.grid(False)
    ax.tick_params(length=0)


def fit_text_within_axes(ax: Axes, texts: Sequence[Any], *, pad_frac: float = 0.015) -> None:
    """Grow the x-limits so every ``text`` stays inside the axes box.

    Value labels are drawn to the right of bars; long labels (e.g. AUC + n)
    would otherwise spill past the axes edge into a neighboring panel or the
    figure margin. We measure the live text extents and expand the x-limits
    with a small proportional pad — applied only on the side(s) that actually
    overflowed, so lower bounds like ``0`` (prevalence) are never pushed
    negative.
    """
    texts = [t for t in texts if t.get_text().strip()]
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    axbox = ax.get_window_extent(renderer)
    x0, x1 = ax.get_xlim()
    lo, hi = x0, x1
    for t in texts:
        ext = t.get_window_extent(renderer)
        if ext.x0 < axbox.x0:
            lo = min(lo, inv.transform((ext.x0, 0))[0])
        if ext.x1 > axbox.x1:
            hi = max(hi, inv.transform((ext.x1, 0))[0])
    span = hi - lo
    pad = pad_frac * span
    new_x0 = x0 - pad if lo < x0 else x0
    new_x1 = x1 + pad if hi > x1 else x1
    if new_x0 != x0 or new_x1 != x1:
        ax.set_xlim(new_x0, new_x1)


def add_title_block(
    fig: Figure,
    title: str,
    subtitle: str | None = None,
    *,
    y: float = 0.98,
) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK, y=y)
    if subtitle:
        fig.text(0.02, y - 0.045, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def color_by_auc(auc: float) -> str:
    """Map discrimination strength to a muted categorical color."""
    if np.isnan(auc):
        return MUTED
    if auc >= 0.70:
        return SUCCESS
    if auc >= 0.60:
        return PRIMARY
    if auc >= 0.55:
        return COOL_MID
    return WARN


def family_color(spec_key: str) -> str:
    key = (spec_key or "").lower()
    if key in {"chance", "chance / prevalence", "prevalence_prob", "majority_class"}:
        return FAMILY_COLORS["chance"]
    if "q28" in key or "rideshare" in key or "mobility" in key:
        return FAMILY_COLORS["q28"]
    if "car" in key or "q20" in key or "q21" in key:
        return FAMILY_COLORS["car"]
    if "country" in key:
        return FAMILY_COLORS["country"]
    if key in {"geo", "geo_benchmark"} or "lat" in key:
        return FAMILY_COLORS["geo"]
    if "ca" in key or "group" in key or "interpersonal" in key:
        return FAMILY_COLORS["ca"]
    if "demo" in key or "age" in key or "sex" in key or "student" in key:
        return FAMILY_COLORS["demographics"]
    if "employ" in key:
        return FAMILY_COLORS["employment"]
    return FAMILY_COLORS["other"]


def add_benchmark_vlines(
    ax: Axes,
    *,
    which: Sequence[str] = ("chance", "geo", "ca"),
    annotate: bool = False,
    ymin: float = 0.0,
    ymax: float = 1.0,
) -> list:
    """Draw benchmark lines; return legend handles when annotate=True."""
    from matplotlib.lines import Line2D

    handles = []
    for name in which:
        meta = BENCHMARKS[name]
        ax.axvline(meta["value"], color=meta["color"], ls=meta["ls"], lw=1.15, zorder=1)
        handles.append(
            Line2D([0], [0], color=meta["color"], ls=meta["ls"], lw=1.4, label=meta["label"])
        )
    if annotate and handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            title="Benchmarks",
            title_fontsize=8.5,
            frameon=True,
            fancybox=False,
            edgecolor=GRID,
            facecolor=PAPER,
            framealpha=0.95,
        )
    return handles


def plot_auc_bars(
    ax: Axes,
    labels: Sequence[str],
    aucs: Sequence[float],
    *,
    colors: Sequence[str] | None = None,
    ns: Sequence[int | None] | None = None,
    xlabel: str = "Stratified CV ROC-AUC",
    xlim: tuple[float, float] | None = None,
    highlight_best: bool = True,
    benchmarks: Sequence[str] = ("chance", "geo", "ca"),
    legend_loc: str | None = "lower right",
) -> list:
    """Horizontal AUC bar chart; returns benchmark legend handles.

    ``legend_loc=None`` skips the in-axes legend so callers can place a
    figure-level legend (e.g. below the axes) without covering bars.
    ``legend_loc="below"`` draws the legend under the axes (tight_layout
    reserves the space), so the longest (bottom) bar and its value label
    stay unobstructed.
    """
    labels = list(labels)
    aucs_arr = np.asarray(list(aucs), dtype=float)
    order = np.argsort(aucs_arr)
    labels = [labels[i] for i in order]
    aucs_arr = aucs_arr[order]
    if colors is None:
        colors_use = [color_by_auc(a) for a in aucs_arr]
    else:
        colors_use = [colors[i] for i in order]
    if highlight_best and len(aucs_arr):
        best_i = int(np.nanargmax(aucs_arr))
        colors_use = list(colors_use)
        colors_use[best_i] = SUCCESS

    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        aucs_arr,
        color=colors_use,
        edgecolor="white",
        linewidth=0.8,
        height=0.72,
        zorder=3,
    )
    xmax = float(np.nanmax(aucs_arr)) if len(aucs_arr) else 0.7
    if xlim is None:
        xlim = (0.45, min(0.92, max(0.72, xmax + 0.08)))
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel)
    style_axes(ax, grid="x")
    bench_handles = add_benchmark_vlines(
        ax, which=benchmarks, annotate=False, ymax=len(labels) - 0.35
    )
    if bench_handles and legend_loc:
        ax.legend(
            handles=bench_handles,
            loc="upper center" if legend_loc == "below" else legend_loc,
            bbox_to_anchor=(0.5, -0.16) if legend_loc == "below" else None,
            ncol=len(bench_handles) if legend_loc == "below" else 1,
            fontsize=7.5,
            title="Benchmarks" if legend_loc != "below" else None,
            title_fontsize=8,
            frameon=legend_loc != "below",
            fancybox=False,
            edgecolor=GRID,
            facecolor=PAPER,
            framealpha=0.95,
        )

    # Omit the repeated "· n=…" suffix when every bar shares the same n —
    # it is redundant noise (n is already in the title block).
    show_n = ns is not None
    if show_n:
        n_vals = [v for v in ns if v is not None]
        if n_vals and all(v == n_vals[0] for v in n_vals):
            show_n = False

    value_texts = []
    for i, (bar, auc) in enumerate(zip(bars, aucs_arr)):
        label = f"{auc:.3f}"
        if show_n:
            n_val = ns[order[i]]
            if n_val is not None and not (isinstance(n_val, float) and np.isnan(n_val)):
                label = f"{auc:.3f}  ·  n={int(n_val)}"
        # Ensure text stays within xlim by dynamically offsetting
        xlim_actual = ax.get_xlim()
        xpos = min(auc + 0.012, xlim_actual[1] - 0.03)
        value_texts.append(
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK,
                fontweight="bold",
                zorder=4,
                clip_on=True,
            )
        )
    fit_text_within_axes(ax, value_texts)
    return bench_handles


def plot_prevalence_bars(
    ax: Axes,
    levels: Sequence[str],
    pcts: Sequence[float],
    *,
    ns: Sequence[int] | None = None,
    sample_prevalence: float | None = None,
    xlabel: str = "Share regular transit (weekly+)",
    title: str | None = None,
    cmap_mode: str = "ramp",
) -> None:
    levels = list(levels)
    pcts_arr = np.asarray(list(pcts), dtype=float)
    order = np.argsort(pcts_arr)
    levels = [levels[i] for i in order]
    pcts_arr = pcts_arr[order]
    if ns is not None:
        ns_arr = [ns[i] for i in order]
    else:
        ns_arr = None

    if cmap_mode == "ramp":
        cmap = LinearSegmentedColormap.from_list("prev", [PRIMARY_PALE, PRIMARY])
        norm = (pcts_arr - pcts_arr.min()) / (np.ptp(pcts_arr) + 1e-9)
        colors = [cmap(0.25 + 0.75 * v) for v in norm]
    else:
        colors = [PRIMARY] * len(pcts_arr)

    y = np.arange(len(levels))
    ax.barh(y, pcts_arr, color=colors, edgecolor="white", height=0.72, zorder=3)
    if sample_prevalence is not None and not np.isnan(sample_prevalence):
        ax.axvline(sample_prevalence, color=ACCENT, ls=(0, (4, 2)), lw=1.2, zorder=2)
        ax.text(
            sample_prevalence,
            len(levels) - 0.35,
            f"cohort {sample_prevalence:.0%}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=7.5,
            color=ACCENT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(levels, fontsize=9)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=8)
    xmax = max(0.72, float(np.nanmax(pcts_arr)) + 0.16)
    ax.set_xlim(0, xmax)
    style_axes(ax, grid="x")
    value_texts = []
    for i, pct in enumerate(pcts_arr):
        txt = f"{pct:.0%}"
        if ns_arr is not None:
            txt = f"{pct:.0%}  (n={ns_arr[i]})"
        value_texts.append(
            ax.text(
                pct + 0.015,
                i,
                txt,
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK,
                clip_on=True,
            )
        )
    fit_text_within_axes(ax, value_texts)
    # Prevalence can never be negative: never let label-fitting push the
    # lower x-limit below 0 (that would draw an out-of-range "-0.2" tick).
    lo, hi = ax.get_xlim()
    ax.set_xlim(max(0.0, lo), hi)


def plot_roc_curve(
    ax: Axes,
    roc: pd.DataFrame,
    *,
    auc: float,
    title: str = "Stratified CV ROC",
    label: str | None = None,
    color: str = PRIMARY,
) -> None:
    fpr = roc["fpr"].to_numpy(dtype=float)
    tpr = roc["tpr"].to_numpy(dtype=float)
    series_label = label or f"Random Forest  ·  AUC {auc:.3f}"
    ax.fill_between(fpr, tpr, alpha=0.12, color=color, zorder=2)
    ax.plot(fpr, tpr, color=color, lw=2.4, label=series_label, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (3, 2)), lw=1.1, label="Chance 0.500", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    style_axes(ax, grid="both")
    # Extra tick pad so the corner "0.0" x-tick clears the "0.0" y-tick label.
    ax.tick_params(axis="x", pad=8)
    ax.legend(loc="lower right", fontsize=8.5)


def plot_forest_diffs(
    ax: Axes,
    strata: pd.DataFrame,
    *,
    outcome: str = "gt_group_ca",
    title: str = "Group CA mean difference by Q28 stratum",
) -> None:
    """Horizontal forest-style plot of mean differences (regular − not)."""
    work = strata.copy()
    if "skipped" in work.columns:
        work = work.loc[~work["skipped"].fillna(False)]
    if "outcome" in work.columns:
        work = work.loc[work["outcome"] == outcome]
    work = work.dropna(subset=["mean_diff"]).copy()
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient stratum sizes", ha="center", va="center", color=MUTED)
        return

    # Prefer a sensible Q28 order when available.
    q28_order = [
        "Never",
        "0-1 days a month",
        "2-4 days a month",
        "4-8 days a month",
        "8 or more days a month",
    ]
    work["Q28"] = pd.Categorical(work["Q28"], categories=q28_order, ordered=True)
    work = work.sort_values("Q28", ascending=False)

    y = np.arange(len(work))
    diffs = work["mean_diff"].to_numpy(dtype=float)
    colors = [SUCCESS if d < 0 else ACCENT for d in diffs]
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    ax.hlines(y, 0, diffs, color=PRIMARY_PALE, lw=2.0, zorder=2)
    ax.scatter(diffs, y, s=55, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    labels = []
    for _, row in work.iterrows():
        p = row.get("welch_p", np.nan)
        star = "" if pd.isna(p) else ("*" if p < 0.05 else "")
        labels.append(f"{row['Q28']}{star}")
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Mean group CA (regular − not regular)")
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    style_axes(ax, grid="x")
    # Auto-expand xlim to fit text labels
    xpad = max(abs(diffs)) * 0.12 + 0.2
    xlim_curr = ax.get_xlim()
    xlim_new = (min(xlim_curr[0], min(diffs) - xpad), max(xlim_curr[1], max(diffs) + xpad))
    ax.set_xlim(*xlim_new)
    # Compact data labels: the p-values / significance stars already live in
    # the accompanying memo tables and on the y-tick labels, so long
    # "−4.8  p=0.006" strings only crowd the panel.
    value_texts = []
    for i, d in enumerate(diffs):
        value_texts.append(
            ax.text(
                d + (0.15 if d >= 0 else -0.15),
                i,
                f"{d:+.1f}",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=8,
                color=INK,
                clip_on=True,
            )
        )
    fit_text_within_axes(ax, value_texts, pad_frac=0.02)


def short_level(text: Any, *, max_len: int = 28) -> str:
    s = str(text)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

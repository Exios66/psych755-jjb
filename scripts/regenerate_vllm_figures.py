#!/usr/bin/env python3
"""Consolidate committed vLLM export metrics and regenerate cross-version memo figures.

Reads every evaluated package under ``exports/{v1,v2,v3,prior_v3_greedy}/``
(``tables/03_metrics_by_tier.csv`` + ``00_run_metadata.json``), writes a
cross-version metrics table to ``outputs/vllm_comparison/``, and regenerates
``memos/figures/vllm_v1_cross_model_memo.png`` and
``memos/figures/vllm_v1_llama31_ip_collapse.png`` to include v2/v3 results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ["v1", "v2", "v3", "prior_v3_greedy"]

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instr",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B (base)",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-8B",
    "meta-llama/Llama-3.3-70B-Instruct-AWQ": "Llama-3.3-70B",
    "casperhansen/llama-3.3-70b-instruct-awq": "Llama-3.3-70B",
}

VERSION_OF_BUCKET = {"v1": "v1", "v2": "v2", "v3": "v3", "prior_v3_greedy": "v3"}


def _short_model(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id.split("/")[-1])


def collect() -> pd.DataFrame:
    rows = []
    for bucket in BUCKETS:
        bdir = ROOT / "exports" / bucket
        if not bdir.is_dir():
            continue
        for pkg in sorted(bdir.iterdir()):
            meta_p = pkg / "00_run_metadata.json"
            metrics_p = pkg / "tables" / "03_metrics_by_tier.csv"
            if not (meta_p.is_file() and metrics_p.is_file()):
                continue
            meta = pd.read_json(meta_p, typ="series").to_dict()
            m = pd.read_csv(metrics_p)
            for _, row in m.iterrows():
                rows.append(
                    {
                        "model_tag": meta.get("model_tag"),
                        "model": meta.get("model"),
                        "bucket": bucket,
                        "version": VERSION_OF_BUCKET.get(bucket, bucket),
                        "tier": row["tier"],
                        "mae_group": row.get("mae_group"),
                        "mae_interpersonal": row.get("mae_interpersonal"),
                        "band_acc_group": row.get("band_acc_group"),
                        "band_acc_interpersonal": row.get("band_acc_interpersonal"),
                        "exact_acc_group": row.get("exact_acc_group"),
                        "exact_acc_interpersonal": row.get("exact_acc_interpersonal"),
                    }
                )
    return pd.DataFrame(rows)


def plot_cross_model(df: pd.DataFrame, out: Path) -> None:
    models = [
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Llama-3.2-3B-Instr", "meta-llama/Llama-3.2-3B-Instruct"),
        ("DeepSeek-R1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
        ("Llama-3.3-70B", ("meta-llama/Llama-3.3-70B-Instruct-AWQ", "casperhansen/llama-3.3-70b-instruct-awq")),
    ]
    versions = ["v1", "v2", "v3"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    x = np.arange(len(models))
    w = 0.22

    colors_mae = {"group": "#4c72b0", "interpersonal": "#c44e52"}
    colors_band = {"group": "#55a868", "interpersonal": "#8172b3"}

    for ax, metric, colors, label in [
        (axes[0], "mae", colors_mae, "MAE (PRCA points)"),
        (axes[1], "band", colors_band, "Band accuracy (%)"),
    ]:
        for vi, version in enumerate(versions):
            vals_g = []
            vals_i = []
            for _, model_ids in models:
                fam = df[df["model"].isin(model_ids) if isinstance(model_ids, tuple) else (df["model"] == model_ids)]
                fam = fam[(fam["version"] == version) & (fam["tier"] == "all")]
                vals_g.append(float(fam["mae_group"].mean()) if metric == "mae" else float(fam["band_acc_group"].mean()) * 100)
                vals_i.append(float(fam["mae_interpersonal"].mean()) if metric == "mae" else float(fam["band_acc_interpersonal"].mean()) * 100)
            off = (vi - 1) * w
            ax.bar(x + off - w / 2, vals_g, w, color=colors["group"], alpha=0.85 if vi < 2 else 1.0, edgecolor="black", linewidth=0.4, label=f"{version} group")
            ax.bar(x + off + w / 2, vals_i, w, color=colors["interpersonal"], alpha=0.85 if vi < 2 else 1.0, edgecolor="black", linewidth=0.4, label=f"{version} interpersonal")
        ax.axhline(4.49 if metric == "mae" else 33.0, color="black", ls="--", lw=1.0)
        ax.text(ax.get_xlim()[1], 4.49 if metric == "mae" else 33.0, "Ridge floor" if metric == "mae" else "~33% chance",
                ha="right", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in models], rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title("Cross-model pooled " + ("MAE by prompt version" if metric == "mae" else "band accuracy by prompt version"))
        ax.legend(fontsize=8, ncol=3, loc="upper left" if metric == "mae" else "upper right")

    fig.suptitle("Full-cohort vLLM digital-twin evaluation: prompt-v1 (v1) vs signal-first v2 vs 8-tier v3 (greedy)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_llama31_collapse(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    llama31 = df[df["model"] == "meta-llama/Llama-3.1-8B-Instruct"]
    deepseek = df[df["model"] == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

    v1_tiers = ["demos", "employment", "geo", "transit", "full"]
    v3_tiers = ["demos", "employment", "geo", "transit", "full", "v3_rideshare", "v3_public_transit", "v3_voice"]

    for ax, (metric, ylabel) in zip(axes, [("mae_interpersonal", "Interpersonal MAE (PRCA points)"), ("band_acc_interpersonal", "Interpersonal band accuracy (%)")]):
        for version in ["v1", "v2", "v3"]:
            series = llama31[(llama31["version"] == version)]
            if series.empty:
                continue
            order = v3_tiers if version == "v3" else v1_tiers
            tiers = [t for t in order if (series["tier"] == t).any()]
            ys = [series.loc[series["tier"] == t, metric].iloc[0] for t in tiers]
            if metric == "band_acc_interpersonal":
                ys = [v * 100 for v in ys]
            ax.plot(range(len(tiers)), ys, marker="o", ms=4, lw=1.8, label=f"Llama-3.1 {version}")
        for version in ["v1", "v2"]:
            series = deepseek[(deepseek["version"] == version)]
            if series.empty:
                continue
            tiers = [t for t in v1_tiers if (series["tier"] == t).any()]
            ys = [series.loc[series["tier"] == t, metric].iloc[0] for t in tiers]
            if metric == "band_acc_interpersonal":
                ys = [v * 100 for v in ys]
            ax.plot(range(len(tiers)), ys, marker="s", ms=4, lw=1.8, ls="--", label=f"DeepSeek {version}")
        ax.set_xticks(range(len(v3_tiers)))
        ax.set_xticklabels(v3_tiers, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Tier")
        ax.axvline(3.5, color="grey", ls=":", lw=1)
        ax.legend(fontsize=8)
        ax.set_title("Llama-3.1 IP " + ("MAE by tier" if metric == "mae_interpersonal" else "band accuracy by tier"))

    fig.suptitle("Llama-3.1 interpersonal collapse persists under v2 packaging; v3 ablations isolate the trigger (greedy decode)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> int:
    df = collect()
    if df.empty:
        raise SystemExit("no export packages found")
    outdir = ROOT / "outputs" / "vllm_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "vllm_cross_version_metrics.csv", index=False)

    figures = ROOT / "memos" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_cross_model(df, figures / "vllm_v1_cross_model_memo.png")
    plot_llama31_collapse(df, figures / "vllm_v1_llama31_ip_collapse.png")

    piv = df[df["tier"] == "all"].pivot_table(
        index=["model", "version"], values=["mae_group", "mae_interpersonal", "band_acc_group", "band_acc_interpersonal"]
    ).round(3)
    print(piv.to_string())
    print(f"\nwrote {outdir / 'vllm_cross_version_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

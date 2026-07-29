#!/usr/bin/env python3
"""Evaluate a vLLM prediction CSV and write a labeled export package."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_personas.evaluate import (
    evaluate_predictions,
    summarize_band_confusion,
    summarize_errors,
    summarize_errors_by_group,
)
from ca_personas.load import load_full_cohort


def _pct(x: object) -> str:
    return "NA" if pd.isna(x) else f"{100 * float(x):.1f}%"


def _f2(x: object) -> str:
    return "NA" if pd.isna(x) else f"{float(x):.2f}"


def infer_export_bucket(tag: str) -> str:
    """Map a model tag onto ``exports/<bucket>/`` for sorted browsing.

    Buckets:
      v1               — published prompt-v1 baselines
      v2               — signal-first 5-tier enhanced runs
      v3               — 8-tier ablation enhanced runs
      prior_v3_greedy  — pre-refresh greedy v3 comparison archives
    """
    t = tag.lower()
    if "prior_greedy" in t or t.endswith("_prior"):
        return "prior_v3_greedy"
    if t.endswith("_v3") or "_v3_" in t:
        return "v3"
    if t.endswith("_v2") or "_v2_" in t:
        return "v2"
    if t.endswith("_v1") or "_v1_" in t:
        return "v1"
    # Legacy tags without version suffix (Jul-26 published baselines) → v1
    return "v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Short model tag for folder names")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--predictions-csv", required=True, type=Path)
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--prompts-csv", type=Path, default=Path("outputs/vllm_prompts/prompts.csv"))
    ap.add_argument("--ground-truth-csv", type=Path, default=Path("outputs/vllm_prompts/ground_truth.csv"))
    ap.add_argument("--throughput", type=float, default=None)
    ap.add_argument("--wall-time-s", type=float, default=None)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--compare-metrics-csv", type=Path, default=None)
    ap.add_argument("--compare-label", default="prior")
    ap.add_argument(
        "--bucket",
        default=None,
        help="Export bucket under exports/ (default: inferred from --tag).",
    )
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    bucket = args.bucket or infer_export_bucket(args.tag)
    export = Path("exports") / bucket / f"psych755_vllm_{args.tag}_full_cohort_{stamp}"
    for sub in ("tables", "figures", "raw"):
        (export / sub).mkdir(parents=True, exist_ok=True)

    participants, cleaning = load_full_cohort(join_how="inner", allow_excerpt_fallback=False)
    preds = pd.read_csv(args.predictions_csv)
    preds["model"] = args.model_id
    results = pd.read_csv(args.results_csv)
    prompts = pd.read_csv(args.prompts_csv)

    n_parsed = int(preds["error"].isna().sum()) if "error" in preds.columns else len(preds)
    parse_rate = float(n_parsed / max(len(preds), 1))

    evaluation = evaluate_predictions(participants, preds)
    summary = summarize_errors(evaluation)

    for gc in [c for c in ["Age", "Sex", "Student status", "Employment status"] if c in evaluation.columns]:
        frame = summarize_errors_by_group(evaluation, gc)
        if frame.empty:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in gc)
        frame.to_csv(export / "tables" / f"stereotyping_by_{safe}.csv", index=False)

    conf_g = summarize_band_confusion(evaluation, "group")
    conf_i = summarize_band_confusion(evaluation, "interpersonal")

    preds.to_csv(export / "tables" / "01_vllm_predictions_parsed.csv", index=False)
    evaluation.to_csv(export / "tables" / "02_evaluation_rowlevel.csv", index=False)
    summary.to_csv(export / "tables" / "03_metrics_by_tier.csv", index=False)
    summary.to_json(export / "tables" / "03_metrics_by_tier.json", orient="records", indent=2)
    conf_g.to_csv(export / "tables" / "04_band_confusion_group.csv")
    conf_i.to_csv(export / "tables" / "05_band_confusion_interpersonal.csv")
    pd.DataFrame([cleaning]).to_json(export / "tables" / "06_cleaning_report.json", orient="records", indent=2)

    parse_by_tier = preds.copy()
    parse_by_tier["parsed_ok"] = parse_by_tier["error"].isna() if "error" in parse_by_tier.columns else True
    if "tier" in parse_by_tier.columns:
        parse_by_tier.groupby("tier")["parsed_ok"].agg(["sum", "mean", "count"]).reset_index().to_csv(
            export / "tables" / "08_parse_success_by_tier.csv", index=False
        )

    shutil.copy2(args.results_csv, export / "raw" / "vllm_results_caseid_generated.csv")
    shutil.copy2(args.prompts_csv, export / "raw" / "vllm_prompts.csv")
    if args.ground_truth_csv.is_file():
        shutil.copy2(args.ground_truth_csv, export / "raw" / "vllm_ground_truth.csv")
    shutil.copy2(args.predictions_csv, export / "raw" / "vllm_predictions.csv")

    core_tiers = ["demos", "employment", "geo", "transit", "full"]
    v3_tiers = ["v3_rideshare", "v3_public_transit", "v3_voice"]
    if "tier" in prompts.columns:
        present = [t for t in [*core_tiers, *v3_tiers] if t in set(prompts["tier"].astype(str))]
    elif "tier" in summary.columns:
        present = [t for t in [*core_tiers, *v3_tiers] if t in set(summary["tier"].astype(str))]
    else:
        present = list(core_tiers)
    tier_order = present or list(core_tiers)

    meta = {
        "export_stamp": stamp,
        "model_tag": args.tag,
        "model": args.model_id,
        "quantization": args.quantization,
        "notes": args.notes,
        "n_participants": int(participants["participant_id"].nunique()),
        "n_prompt_rows": int(len(prompts)),
        "n_result_rows": int(len(results)),
        "n_predictions_parsed": n_parsed,
        "parse_rate": parse_rate,
        "tiers": tier_order,
        "join": "inner",
        "throughput_samples_per_s": args.throughput,
        "wall_time_s": args.wall_time_s,
        "gpu": "NVIDIA RTX A5000",
        "host": "rogers-gpu-1.discovery.wisc.edu",
    }
    (export / "00_run_metadata.json").write_text(json.dumps(meta, indent=2))

    plot_df = summary[summary["tier"].isin(tier_order)].copy()
    if not plot_df.empty:
        plot_df["tier"] = pd.Categorical(plot_df["tier"], categories=tier_order, ordered=True)
        plot_df = plot_df.sort_values("tier")
        x = np.arange(len(plot_df))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, plot_df["mae_group"].fillna(0), w, label="Group CA MAE")
        ax.bar(x + w / 2, plot_df["mae_interpersonal"].fillna(0), w, label="Interpersonal CA MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df["tier"]))
        ax.set_ylabel("MAE (PRCA points)")
        ax.set_title(f"{args.model_id}: MAE by tier")
        ax.legend()
        fig.tight_layout()
        fig.savefig(export / "figures" / "01_mae_by_tier.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, plot_df["band_acc_group"].fillna(0) * 100, w, label="Group band %")
        ax.bar(x + w / 2, plot_df["band_acc_interpersonal"].fillna(0) * 100, w, label="Interpersonal band %")
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df["tier"]))
        ax.set_ylim(0, 100)
        ax.set_ylabel("Band accuracy (%)")
        ax.set_title(f"{args.model_id}: band accuracy by tier")
        ax.legend()
        fig.tight_layout()
        fig.savefig(export / "figures" / "02_band_accuracy_by_tier.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, plot_df["exact_acc_group"].fillna(0) * 100, w, label="Group exact %")
        ax.bar(x + w / 2, plot_df["exact_acc_interpersonal"].fillna(0) * 100, w, label="Interpersonal exact %")
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df["tier"]))
        ax.set_ylabel("Exact match (%)")
        ax.set_title(f"{args.model_id}: exact score match by tier")
        ax.legend()
        fig.tight_layout()
        fig.savefig(export / "figures" / "03_exact_match_by_tier.png", dpi=160)
        plt.close()

        if (plot_df["tier"].astype(str) == "demos").any():
            base = plot_df[plot_df["tier"].astype(str) == "demos"].iloc[0]
            deltas = []
            for _, r in plot_df.iterrows():
                deltas.append(
                    {
                        "tier": str(r["tier"]),
                        "delta_mae_group_vs_demos": (
                            float(r["mae_group"] - base["mae_group"])
                            if pd.notna(r["mae_group"]) and pd.notna(base["mae_group"])
                            else None
                        ),
                        "delta_mae_interpersonal_vs_demos": (
                            float(r["mae_interpersonal"] - base["mae_interpersonal"])
                            if pd.notna(r["mae_interpersonal"]) and pd.notna(base["mae_interpersonal"])
                            else None
                        ),
                        "delta_band_acc_group_vs_demos": (
                            float(r["band_acc_group"] - base["band_acc_group"])
                            if pd.notna(r["band_acc_group"]) and pd.notna(base["band_acc_group"])
                            else None
                        ),
                        "delta_band_acc_interpersonal_vs_demos": (
                            float(r["band_acc_interpersonal"] - base["band_acc_interpersonal"])
                            if pd.notna(r["band_acc_interpersonal"])
                            and pd.notna(base["band_acc_interpersonal"])
                            else None
                        ),
                    }
                )
            pd.DataFrame(deltas).to_csv(export / "tables" / "07_tier_deltas_vs_demos.csv", index=False)

    cmp_txt = ""
    if args.compare_metrics_csv and args.compare_metrics_csv.is_file():
        prior = pd.read_csv(args.compare_metrics_csv)
        rows = []
        for tier in ["all", *tier_order]:
            a = summary[summary["tier"] == tier]
            b = prior[prior["tier"] == tier]
            if a.empty or b.empty:
                continue
            a = a.iloc[0]
            b = b.iloc[0]
            rows.append(
                {
                    "tier": tier,
                    "mae_group_this": a["mae_group"],
                    f"mae_group_{args.compare_label}": b["mae_group"],
                    "mae_ip_this": a["mae_interpersonal"],
                    f"mae_ip_{args.compare_label}": b["mae_interpersonal"],
                    "band_group_this": a["band_acc_group"],
                    f"band_group_{args.compare_label}": b["band_acc_group"],
                    "band_ip_this": a["band_acc_interpersonal"],
                    f"band_ip_{args.compare_label}": b["band_acc_interpersonal"],
                }
            )
        cmp = pd.DataFrame(rows)
        cmp.to_csv(export / "tables" / f"09_compare_vs_{args.compare_label}.csv", index=False)
        if not cmp.empty and (cmp["tier"] == "all").any():
            call = cmp[cmp["tier"] == "all"].iloc[0]
            cmp_txt = f"""
## Comparison vs {args.compare_label}

| Metric (all tiers) | This model | {args.compare_label} |
|---|---:|---:|
| MAE group | {_f2(call['mae_group_this'])} | {_f2(call[f'mae_group_{args.compare_label}'])} |
| MAE interpersonal | {_f2(call['mae_ip_this'])} | {_f2(call[f'mae_ip_{args.compare_label}'])} |
| Band group | {_pct(call['band_group_this'])} | {_pct(call[f'band_group_{args.compare_label}'])} |
| Band interpersonal | {_pct(call['band_ip_this'])} | {_pct(call[f'band_ip_{args.compare_label}'])} |
"""

    allr = summary[summary["tier"] == "all"].iloc[0]
    demos = summary[summary["tier"] == "demos"].iloc[0] if (summary["tier"] == "demos").any() else allr
    emp = summary[summary["tier"] == "employment"].iloc[0] if (summary["tier"] == "employment").any() else allr
    tr = summary[summary["tier"] == "transit"].iloc[0] if (summary["tier"] == "transit").any() else allr
    full = summary[summary["tier"] == "full"].iloc[0] if (summary["tier"] == "full").any() else allr

    report = f"""# PSYCH 755 — vLLM Results Report: `{args.model_id}`

**Model tag:** `{args.tag}`  
**Sample:** N={meta['n_participants']} × {len(tier_order)} tiers = {meta['n_prompt_rows']} prompts  

**Parse success:** {n_parsed}/{meta['n_result_rows']} ({100 * parse_rate:.1f}%)  
**Quantization:** {args.quantization or 'n/a'}  
**Throughput:** {args.throughput} samples/s · wall {args.wall_time_s}s  
**Export:** `{export.name}`

{args.notes}

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | {_f2(allr.mae_group)} |
| MAE interpersonal | {_f2(allr.mae_interpersonal)} |
| Exact acc group | {_pct(allr.exact_acc_group)} |
| Exact acc interpersonal | {_pct(allr.exact_acc_interpersonal)} |
| Band acc group | {_pct(allr.band_acc_group)} |
| Band acc interpersonal | {_pct(allr.band_acc_interpersonal)} |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE {_f2(demos.mae_group)} → {_f2(emp.mae_group)}; IP {_f2(demos.mae_interpersonal)} → {_f2(emp.mae_interpersonal)}
- **RQ2 transit:** group MAE {_f2(demos.mae_group)} → {_f2(tr.mae_group)}; IP {_f2(demos.mae_interpersonal)} → {_f2(tr.mae_interpersonal)}; IP band {_pct(demos.band_acc_interpersonal)} → {_pct(tr.band_acc_interpersonal)}
- **RQ3 full:** group MAE {_f2(full.mae_group)}; group band {_pct(demos.band_acc_group)} → {_pct(full.band_acc_group)}; IP MAE {_f2(demos.mae_interpersonal)} → {_f2(full.mae_interpersonal)}

{cmp_txt}

## Metrics by tier

{summary.to_string(index=False)}

## Band confusion (group)

{conf_g.to_string()}

## Band confusion (interpersonal)

{conf_i.to_string()}

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | {'Yes' if parse_rate >= 0.95 else 'Partial/No'} ({100 * parse_rate:.1f}%) |
| Exact digital-twin recovery | {'No' if (pd.isna(allr.exact_acc_group) or allr.exact_acc_group < 0.2) else 'Partial'} |
| Coarse band recovery | See tier table |
"""
    (export / "REPORT_results_interpretation_and_model_success.md").write_text(report)
    (export / "README_EXPORT.txt").write_text(
        f"Model: {args.model_id}\nTag: {args.tag}\nParse: {100 * parse_rate:.1f}%\nSee REPORT_*.md\n"
    )
    html = (
        "<html><head><meta charset='utf-8'/><title>"
        + args.tag
        + "</title></head><body><pre style='white-space:pre-wrap;font-family:Georgia,serif'>"
        + report.replace("&", "&amp;").replace("<", "&lt;")
        + "</pre>"
        + "<img src='figures/01_mae_by_tier.png'/><img src='figures/02_band_accuracy_by_tier.png'/>"
        + "<img src='figures/03_exact_match_by_tier.png'/></body></html>"
    )
    (export / "REPORT_results_interpretation_and_model_success.html").write_text(html)

    # Keep exports/INDEX.md current for browsing done vs pending.
    try:
        import importlib.util
        import sys

        idx_path = Path(__file__).resolve().parent / "refresh_vllm_export_index.py"
        spec = importlib.util.spec_from_file_location("refresh_vllm_export_index", idx_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            mod.main()
    except Exception as exc:  # noqa: BLE001 — index refresh must not fail packaging
        print(f"[warn] could not refresh exports/INDEX.md: {exc}")

    print(
        json.dumps(
            {
                "export": str(export.resolve()),
                "bucket": bucket,
                "parse_rate": parse_rate,
                "n_parsed": n_parsed,
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

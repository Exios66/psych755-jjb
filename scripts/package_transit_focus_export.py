#!/usr/bin/env python3
"""Evaluate transit-focus vLLM generations and write exports/transit_focus/ packages."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from ca_personas.llm.base import extract_json_object
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import (
    PRIMARY_REGULAR_LABELS,
    Q26_ORDER,
    label_regular_riders,
    normalize_q26,
)
from inference.ca_prompts import parse_caseid
from inference.utils import normalize_caseid


Q26_TO_ORD = {lab: i for i, lab in enumerate(Q26_ORDER)}


def _pct(x: object) -> str:
    return "NA" if pd.isna(x) else f"{100 * float(x):.1f}%"


def _f3(x: object) -> str:
    return "NA" if pd.isna(x) else f"{float(x):.3f}"


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "regular"}:
        return True
    if text in {"false", "0", "no", "not regular", "non-regular"}:
        return False
    return None


def results_to_transit_predictions(result_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(result_csv)
    rows: list[dict] = []
    for _, row in df.iterrows():
        caseid = normalize_caseid(row["caseid"])
        try:
            participant_id, tier = parse_caseid(caseid)
        except ValueError as exc:
            rows.append(
                {
                    "caseid": caseid,
                    "participant_id": None,
                    "tier": None,
                    "raw_response": row.get("generated_text", ""),
                    "error": f"ValueError: {exc}",
                    "pred_regular_transit": None,
                    "pred_q26_days": None,
                    "pred_confidence": None,
                }
            )
            continue
        raw = "" if pd.isna(row.get("generated_text")) else str(row["generated_text"])
        error = None
        pred_reg = None
        pred_q26 = None
        conf = None
        try:
            payload = extract_json_object(raw)
            pred_reg = _parse_bool(payload.get("regular_transit"))
            pred_q26 = normalize_q26(payload.get("q26_days"))
            conf = payload.get("confidence")
            if pred_reg is None and pred_q26 is None:
                raise ValueError("missing regular_transit and q26_days")
            # Consistency: derive regular from q26 when bool missing
            if pred_reg is None and pred_q26 is not None:
                pred_reg = pred_q26 in PRIMARY_REGULAR_LABELS
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "caseid": caseid,
                "participant_id": participant_id,
                "tier": tier,
                "raw_response": raw,
                "error": error,
                "pred_regular_transit": pred_reg,
                "pred_q26_days": pred_q26,
                "pred_confidence": conf,
            }
        )
    return pd.DataFrame(rows)


def evaluate_transit_focus(participants: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    labeled = label_regular_riders(participants)
    gt = labeled[
        ["participant_id", "regular_transit", "Q26"]
    ].drop_duplicates("participant_id").copy()
    gt["participant_id"] = gt["participant_id"].map(normalize_caseid)
    gt["gt_q26_days"] = gt["Q26"].map(normalize_q26)
    gt["gt_regular_transit"] = gt["regular_transit"].astype("boolean")

    merged = preds.merge(gt, on="participant_id", how="left")
    merged["pred_ok"] = merged["error"].isna()
    return merged


def summarize_by_tier(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    tiers = ["all", *sorted(eval_df["tier"].dropna().astype(str).unique())]
    for tier in tiers:
        sub = eval_df if tier == "all" else eval_df[eval_df["tier"].astype(str) == tier]
        parsed = sub[sub["pred_ok"]]
        n = len(sub)
        n_parsed = len(parsed)
        row: dict = {
            "tier": tier,
            "n": n,
            "n_parsed": n_parsed,
            "parse_rate": n_parsed / n if n else float("nan"),
        }
        # Binary regular
        both = parsed.dropna(subset=["pred_regular_transit", "gt_regular_transit"])
        if len(both):
            y_true = both["gt_regular_transit"].astype(bool)
            y_pred = both["pred_regular_transit"].astype(bool)
            row["regular_acc"] = float(accuracy_score(y_true, y_pred))
            row["regular_bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
            row["regular_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            row["n_regular_eval"] = int(len(both))
        else:
            row.update(
                {
                    "regular_acc": float("nan"),
                    "regular_bal_acc": float("nan"),
                    "regular_f1": float("nan"),
                    "n_regular_eval": 0,
                }
            )
        # Q26 ordinal
        q = parsed.dropna(subset=["pred_q26_days", "gt_q26_days"]).copy()
        if len(q):
            q["pred_ord"] = q["pred_q26_days"].map(Q26_TO_ORD)
            q["gt_ord"] = q["gt_q26_days"].map(Q26_TO_ORD)
            q = q.dropna(subset=["pred_ord", "gt_ord"])
            if len(q):
                row["q26_exact_acc"] = float(accuracy_score(q["gt_ord"], q["pred_ord"]))
                row["q26_mae"] = float(np.mean(np.abs(q["pred_ord"] - q["gt_ord"])))
                row["n_q26_eval"] = int(len(q))
            else:
                row.update({"q26_exact_acc": float("nan"), "q26_mae": float("nan"), "n_q26_eval": 0})
        else:
            row.update({"q26_exact_acc": float("nan"), "q26_mae": float("nan"), "n_q26_eval": 0})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--prompts-csv", type=Path, default=Path("outputs/vllm_prompts_transit_focus/prompts.csv"))
    ap.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=Path("outputs/vllm_prompts_transit_focus/ground_truth.csv"),
    )
    ap.add_argument("--quantization", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--throughput", type=float, default=None)
    ap.add_argument("--wall-time-s", type=float, default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    export = Path("exports") / "transit_focus" / f"psych755_vllm_{args.tag}_full_cohort_{stamp}"
    for sub in ("tables", "figures", "raw"):
        (export / sub).mkdir(parents=True, exist_ok=True)

    participants, cleaning = load_full_cohort(join_how="inner", allow_excerpt_fallback=False)
    preds = results_to_transit_predictions(args.results_csv)
    preds["model"] = args.model_id
    evaluation = evaluate_transit_focus(participants, preds)
    summary = summarize_by_tier(evaluation)
    prompts = pd.read_csv(args.prompts_csv) if args.prompts_csv.is_file() else pd.DataFrame()
    results = pd.read_csv(args.results_csv)

    n_parsed = int(preds["error"].isna().sum())
    parse_rate = float(n_parsed / max(len(preds), 1))

    preds.to_csv(export / "tables" / "01_transit_predictions_parsed.csv", index=False)
    evaluation.to_csv(export / "tables" / "02_evaluation_rowlevel.csv", index=False)
    summary.to_csv(export / "tables" / "03_metrics_by_tier.csv", index=False)
    summary.to_json(export / "tables" / "03_metrics_by_tier.json", orient="records", indent=2)
    pd.DataFrame([cleaning]).to_json(export / "tables" / "06_cleaning_report.json", orient="records", indent=2)

    shutil.copy2(args.results_csv, export / "raw" / "vllm_results_caseid_generated.csv")
    if args.prompts_csv.is_file():
        shutil.copy2(args.prompts_csv, export / "raw" / "vllm_prompts_transit_focus.csv")
    if args.ground_truth_csv.is_file():
        shutil.copy2(args.ground_truth_csv, export / "raw" / "vllm_ground_truth.csv")
    preds.to_csv(export / "raw" / "vllm_predictions.csv", index=False)

    tier_order = ["tf_demos", "tf_employment", "tf_geo", "tf_geo_ca"]
    meta = {
        "export_stamp": stamp,
        "model_tag": args.tag,
        "model": args.model_id,
        "task": "transit_focus",
        "quantization": args.quantization,
        "notes": args.notes,
        "n_participants": int(participants["participant_id"].nunique()),
        "n_prompt_rows": int(len(prompts)) if len(prompts) else int(len(results)),
        "n_result_rows": int(len(results)),
        "n_predictions_parsed": n_parsed,
        "parse_rate": parse_rate,
        "tiers": tier_order,
        "join": "inner",
        "throughput_samples_per_s": args.throughput,
        "wall_time_s": args.wall_time_s,
        "export_bucket": "transit_focus",
        "gpu": "NVIDIA RTX A5000",
        "host": "rogers-gpu-1.discovery.wisc.edu",
    }
    (export / "00_run_metadata.json").write_text(json.dumps(meta, indent=2))

    plot_df = summary[summary["tier"].isin(tier_order)].copy()
    if not plot_df.empty:
        plot_df["tier"] = pd.Categorical(plot_df["tier"], categories=tier_order, ordered=True)
        plot_df = plot_df.sort_values("tier")
        x = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, plot_df["regular_bal_acc"].fillna(0) * 100)
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df["tier"]), rotation=15)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Balanced accuracy (%)")
        ax.set_title(f"{args.model_id}: TF1 regular-transit bal. acc.")
        fig.tight_layout()
        fig.savefig(export / "figures" / "01_regular_bal_acc_by_tier.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, plot_df["q26_mae"].fillna(0))
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df["tier"]), rotation=15)
        ax.set_ylabel("Q26 ordinal MAE")
        ax.set_title(f"{args.model_id}: TF2 Q26 day-frequency MAE")
        fig.tight_layout()
        fig.savefig(export / "figures" / "02_q26_mae_by_tier.png", dpi=160)
        plt.close()

    allr = summary[summary["tier"] == "all"].iloc[0]
    report = f"""# PSYCH 755 — Transit-focus vLLM Report: `{args.model_id}`

**Task:** TF1/TF2 — predict regular transit + Q26 with mobility held out  
**Model tag:** `{args.tag}`  
**Sample:** N≈{meta['n_participants']} × 4 tiers = {meta['n_prompt_rows']} prompts  
**Parse success:** {n_parsed}/{meta['n_result_rows']} ({100 * parse_rate:.1f}%)  
**Quantization:** {args.quantization or 'n/a'}  
**Export:** `{export}`

{args.notes}

---

## Overall metrics

| Metric | Value |
|---|---|
| Parse rate | {_pct(allr.parse_rate)} |
| Regular transit accuracy | {_pct(allr.regular_acc)} |
| Regular transit balanced acc | {_pct(allr.regular_bal_acc)} |
| Regular transit F1 | {_f3(allr.regular_f1)} |
| Q26 exact match | {_pct(allr.q26_exact_acc)} |
| Q26 ordinal MAE | {_f3(allr.q26_mae)} |

Tabular ceiling (seed=42): profile→regular AUC ≈ **0.662**; profile+CA ≈ **0.672**.

## Metrics by tier

{summary.to_string(index=False)}
"""
    (export / "REPORT_results_interpretation_and_model_success.md").write_text(report)
    (export / "README_EXPORT.txt").write_text(
        f"Task: transit_focus\nModel: {args.model_id}\nTag: {args.tag}\n"
        f"Parse: {100 * parse_rate:.1f}%\nSee REPORT_*.md\n"
    )

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
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] index refresh: {exc}")

    print(json.dumps({"export": str(export.resolve()), "bucket": "transit_focus", "parse_rate": parse_rate}, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

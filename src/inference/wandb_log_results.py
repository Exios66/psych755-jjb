"""CLI: score an existing vLLM results CSV into a Weights & Biases run.

Example::

    export WANDB_API_KEY=...
    python -m inference.wandb_log_results \\
        --result_csv outputs/vllm_results/results.csv \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --preset v2_enhanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.wandb_tracing import start_wandb_run, wandb_configured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Log vLLM result CSVs to Weights & Biases with CA chunk metrics.",
    )
    ap.add_argument("--result_csv", required=True, type=Path)
    ap.add_argument("--model", type=str, default="unknown")
    ap.add_argument("--preset", type=str, default="unknown")
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable W&B (default: on when WANDB_API_KEY set).",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Rows per logged chunk (default 200, matches vLLM save_freq).",
    )
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.result_csv)
    if "caseid" not in df.columns or "generated_text" not in df.columns:
        raise SystemExit("Result CSV must have columns: caseid, generated_text")

    run = start_wandb_run(
        model=args.model,
        preset=args.preset,
        enabled=args.wandb,
        project=args.project,
        run_name=args.run_name,
        extra_config={"result_csv": str(args.result_csv), "posthoc": True},
    )
    if args.wandb is True and not run.enabled:
        print(json.dumps({"enabled": False, "error": "wandb not started"}))
        return 1

    rows = df.to_dict(orient="records")
    chunk = max(1, int(args.chunk_size))
    chunks = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    parse_ok = 0
    n = 0
    mae_vals: list[float] = []
    for i, part in enumerate(chunks):
        summary = run.log_chunk(
            chunk_idx=i,
            n_chunks=len(chunks),
            rows=part,
            model=args.model,
            preset=args.preset,
        )
        n += int(summary.get("n") or 0)
        pr = summary.get("parse_rate")
        if pr is not None and summary.get("n"):
            parse_ok += int(round(float(pr) * int(summary["n"])))
        if summary.get("mae_mean") is not None:
            mae_vals.append(float(summary["mae_mean"]))

    final = {
        "n": n,
        "n_logged": run.n_logged,
        "parse_rate": (parse_ok / n) if n else None,
        "mae_mean": (sum(mae_vals) / len(mae_vals)) if mae_vals else None,
        "run_name": run.run_name,
        "project": run.project,
        "enabled": run.enabled,
    }
    run.log_summary(final)
    url = run.close()
    if url:
        final["url"] = url
    print(json.dumps(final, indent=2))
    if args.wandb is True and not final.get("enabled"):
        return 1
    if not wandb_configured(enabled=args.wandb):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

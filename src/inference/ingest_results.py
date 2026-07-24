"""CLI: convert vLLM result CSVs into CA prediction tables for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.ca_prompts import results_to_predictions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Parse vLLM results (caseid, generated_text) into CA prediction rows "
            "compatible with ca_personas.evaluate."
        ),
    )
    ap.add_argument("--result_csv", required=True, type=Path)
    ap.add_argument(
        "--predictions_csv",
        type=Path,
        default=Path("outputs/predictions/vllm_predictions.csv"),
    )
    args = ap.parse_args(argv)

    preds = results_to_predictions(args.result_csv)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.predictions_csv, index=False)
    n_ok = int(preds["error"].isna().sum()) if "error" in preds.columns else len(preds)
    print(
        json.dumps(
            {
                "predictions": str(args.predictions_csv),
                "n_rows": int(len(preds)),
                "n_parsed": n_ok,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

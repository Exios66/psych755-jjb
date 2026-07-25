#!/usr/bin/env python3
"""Re-run mock persona predictions from committed Posit participants.

Use when private File A/B/C are unavailable but ``artifacts/posit_full_cohort/
participants.csv`` already holds the full analytic cohort. Rebuilds persona
prompts (including base-demos Student status), mock predictions, evaluation
(with Student status joined for stereotyping slices), and summary tables.

Usage:

    python scripts/rerun_posit_mock_from_participants.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.evaluate import (  # noqa: E402
    evaluate_predictions,
    summarize_band_confusion,
    summarize_errors,
    summarize_errors_by_group,
)
from ca_personas.ground_truth import aggregate_ground_truth  # noqa: E402
from ca_personas.llm.base import get_client  # noqa: E402
from ca_personas.personas import TIERS, build_persona_prompts  # noqa: E402
from ca_personas.predict import run_predictions  # noqa: E402


def main() -> int:
    out = ROOT / "artifacts" / "posit_full_cohort"
    participants_path = out / "participants.csv"
    if not participants_path.is_file():
        raise SystemExit(f"Missing {participants_path}")

    participants = pd.read_csv(participants_path)
    n = len(participants)
    if n < 100:
        raise SystemExit(f"Analytic N={n} looks like an excerpt; aborting.")
    if "Student status" not in participants.columns:
        raise SystemExit("participants.csv missing Student status (base demos layer).")

    prompts = build_persona_prompts(participants, tiers=list(TIERS))
    client = get_client("mock")
    predictions = run_predictions(client, prompts, sleep_seconds=0.0)
    evaluation = evaluate_predictions(participants, predictions)
    if "Student status" not in evaluation.columns:
        raise SystemExit("evaluate_predictions did not join Student status.")

    summary = summarize_errors(evaluation)
    by_student = summarize_errors_by_group(evaluation, "Student status")
    by_employment = summarize_errors_by_group(evaluation, "Employment status")
    aggregates = aggregate_ground_truth(participants)

    evaluation.to_csv(out / "evaluation.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    by_student.to_csv(out / "error_by_student_status.csv", index=False)
    by_employment.to_csv(out / "error_by_employment.csv", index=False)
    aggregates.to_csv(out / "ground_truth_aggregates.csv", index=False)

    for side in ("group", "interpersonal"):
        confusion = summarize_band_confusion(evaluation, side=side)
        if not confusion.empty:
            confusion.to_csv(out / f"band_confusion_{side}.csv")

    source_path = out / "source.json"
    meta = {}
    if source_path.is_file():
        meta = json.loads(source_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "n_analytic": int(n),
            "provider": "mock",
            "tiers": list(TIERS),
            "base_demos_includes_student_status": True,
            "rerun_from": "artifacts/posit_full_cohort/participants.csv",
            "note": (
                "Mock predictions regenerated from committed participants so "
                "persona system-prompt / demos-layer updates (incl. Student "
                "status) stay aligned without private File A/B/C."
            ),
        }
    )
    source_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    demos = summary.loc[summary["tier"] == "demos"].iloc[0]
    transit = summary.loc[summary["tier"] == "transit"].iloc[0]
    card = {
        "n_analytic": int(n),
        "n_prompts": int(len(prompts)),
        "mae_group_demos": float(demos["mae_group"]),
        "mae_group_transit": float(transit["mae_group"]),
        "mae_interpersonal_demos": float(demos["mae_interpersonal"]),
        "mae_interpersonal_transit": float(transit["mae_interpersonal"]),
        "student_scopes": sorted(by_student["group_key"].unique().tolist()),
        "evaluation_has_student_status": True,
    }
    print(json.dumps(card, indent=2))
    print(f"Wrote mock evaluation artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

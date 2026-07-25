#!/usr/bin/env python3
"""Build committed full-cohort mock artifacts for Posit Connect Cloud.

Private File A/B/C exports are never committed. This script runs the offline
mock pipeline on the staged full cohort and writes summary/evaluation tables
under ``artifacts/posit_full_cohort/`` so the Quarto site can display
full-cohort statistics without excerpt fixtures.

Usage:

    python scripts/sync_posit_full_cohort_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.paths import (  # noqa: E402
    cohort_source_label,
    full_cohort_paths,
    sibling_data_available,
)
from ca_personas.personas import TIERS  # noqa: E402
from ca_personas.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    if not sibling_data_available():
        raise SystemExit(
            "Full-cohort File A/B/C not found. Stage exports under "
            "../sibling_data/, /tmp/sibling_data, or CA_SIBLING_DATA."
        )

    prolific, qualtrics = full_cohort_paths()
    source = cohort_source_label()
    if "excerpt" in source.lower():
        raise SystemExit(f"Refusing to sync excerpt data source: {source}")

    out = ROOT / "artifacts" / "posit_full_cohort"
    out.mkdir(parents=True, exist_ok=True)
    render_tmp = ROOT / "outputs" / "quarto_render_full_cohort"

    artifacts = run_pipeline(
        prolific_path=prolific,
        qualtrics_path=qualtrics,
        tiers=list(TIERS),
        provider="mock",
        output_dir=render_tmp,
        join_how="inner",
    )

    participants = Path(artifacts["participants"]).read_bytes()
    evaluation = Path(artifacts["evaluation"]).read_bytes()
    summary = Path(artifacts["summary"]).read_bytes()

    (out / "participants.csv").write_bytes(participants)
    (out / "evaluation.csv").write_bytes(evaluation)
    (out / "summary.csv").write_bytes(summary)

    # Stereotyping slices for base demos (student status) and RQ1 employment.
    for key, name in (
        ("error_by_student_status", "error_by_student_status.csv"),
        ("error_by_employment", "error_by_employment.csv"),
    ):
        src = artifacts.get(key)
        if src is not None and Path(src).is_file():
            (out / name).write_bytes(Path(src).read_bytes())

    import pandas as pd

    n = len(pd.read_csv(out / "participants.csv"))
    if n < 100:
        raise SystemExit(f"Analytic N={n} looks like an excerpt; aborting sync.")

    meta = {
        "data_source": "full_cohort File A/B/C",
        "resolved_source_label": source,
        "n_analytic": int(n),
        "provider": "mock",
        "join_how": "inner",
        "tiers": list(TIERS),
        "n_prolific_files": len(prolific),
        "note": (
            "Committed for Posit Connect Cloud renders when private File A/B/C "
            "are absent. Never generated from data/excerpts/."
        ),
    }
    (out / "source.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote artifacts under {out}")
    secondary = out / "secondary_results.json"
    if not secondary.is_file():
        print("WARNING: secondary_results.json missing; manuscript secondary figures require it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build committed full-cohort mock artifacts for Posit Connect Cloud.

Private File A/B/C exports are never committed. This script runs the offline
mock pipeline on the staged full cohort and writes summary/evaluation tables
under ``artifacts/posit_full_cohort/`` so the Quarto site can display
full-cohort statistics without excerpt fixtures.

Also regenerates ``secondary_results.json`` from seeded secondary RQs so
primary mock MAE tables and secondary AUCs stay in sync.

Usage:

    python scripts/sync_posit_full_cohort_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ca_personas.paths import (  # noqa: E402
    cohort_source_label,
    full_cohort_paths,
    sibling_data_available,
)
from ca_personas.personas import TIERS  # noqa: E402
from ca_personas.pipeline import run_pipeline  # noqa: E402
from ca_personas.posit_secondary import write_secondary_results  # noqa: E402


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

    participants_path = out / "participants.csv"
    participants_path.write_bytes(Path(artifacts["participants"]).read_bytes())
    (out / "evaluation.csv").write_bytes(Path(artifacts["evaluation"]).read_bytes())
    (out / "summary.csv").write_bytes(Path(artifacts["summary"]).read_bytes())

    # Stereotyping slices for base demos (student status) and RQ1 employment.
    for key, name in (
        ("error_by_student_status", "error_by_student_status.csv"),
        ("error_by_employment", "error_by_employment.csv"),
    ):
        src = artifacts.get(key)
        if src is not None and Path(src).is_file():
            (out / name).write_bytes(Path(src).read_bytes())

    participants = pd.read_csv(participants_path)
    n = len(participants)
    if n < 100:
        raise SystemExit(f"Analytic N={n} looks like an excerpt; aborting sync.")

    secondary_path = write_secondary_results(
        participants,
        out / "secondary_results.json",
        random_state=42,
        n_boot=800,
        n_perm_repeats=8,
    )
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    if int(secondary.get("n_analytic", 0)) != n:
        raise SystemExit(
            f"secondary n_analytic={secondary.get('n_analytic')} != participants N={n}"
        )

    meta = {
        "data_source": "full_cohort File A/B/C",
        "resolved_source_label": source,
        "n_analytic": int(n),
        "provider": "mock",
        "join_how": "inner",
        "tiers": list(TIERS),
        "n_prolific_files": len(prolific),
        "secondary_results": str(secondary_path.relative_to(ROOT)),
        "secondary_seed": 42,
        "note": (
            "Committed for Posit Connect Cloud renders when private File A/B/C "
            "are absent. Never generated from data/excerpts/."
        ),
    }
    (out / "source.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote artifacts under {out}")
    print(
        "Secondary AUCs: "
        f"geo={secondary['geo_rf']['roc_auc']:.3f} "
        f"ca={secondary['ca_rf']['roc_auc']:.3f} "
        f"q28={[r['roc_auc'] for r in secondary['covariate_comparison'] if r['spec_key']=='q28_days']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

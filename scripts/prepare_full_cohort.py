#!/usr/bin/env python3
"""Load File A/B/C from ../sibling_data, clean, score, and write EDA artifacts.

Private exports are never committed. Expected layout:

    ../sibling_data/PRCAProlificExport_FileA.csv
    ../sibling_data/PRCAProlificExport_FileB.csv
    ../sibling_data/PRCAQualtricsExport_FileC.csv

Example:

    python scripts/prepare_full_cohort.py
    # or
    ca-personas prepare --join inner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.pipeline import prepare_analytic_sample  # noqa: E402


def main() -> int:
    artifacts = prepare_analytic_sample(
        join_how="inner",
        output_dir=ROOT / "outputs",
    )
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Load File A/B/C from ../sibling_data (or excerpt fallback), clean, score, write EDA.

Private exports are never committed. Expected layout:

    ../sibling_data/PRCAProlificExport_FileA.csv
    ../sibling_data/PRCAProlificExport_FileB.csv
    ../sibling_data/PRCAQualtricsExport_FileC.csv

When sibling data is absent, path resolution falls back to ``data/excerpts/``.

Example:

    python scripts/prepare_full_cohort.py
    python scripts/prepare_full_cohort.py --join inner
    # or
    ca-personas prepare --join inner
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *sys.argv[1:]]))

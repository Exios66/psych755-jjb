#!/usr/bin/env python3
"""Run the CA → regular-transit Random Forest secondary RQ.

Example:

    python scripts/run_ca_transit_rf.py
    python scripts/run_ca_transit_rf.py --join inner --seed 42
    # or
    ca-personas ca-transit-rf --join inner
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["ca-transit-rf", *sys.argv[1:]]))

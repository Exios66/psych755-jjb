#!/usr/bin/env python3
"""Run the secondary transit ↔ CA research-question pipeline.

Example:

    python scripts/run_transit_ca.py
    python scripts/run_transit_ca.py --join inner --n-boot 2000
    # or
    ca-personas transit-ca --join inner
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["transit-ca", *sys.argv[1:]]))

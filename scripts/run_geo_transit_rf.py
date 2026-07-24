#!/usr/bin/env python3
"""Run the geography → regular-transit Random Forest secondary RQ.

Example:

    python scripts/run_geo_transit_rf.py
    # or
    ca-personas geo-transit-rf --join inner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.geo_transit_rf import run_geo_transit_rf_pipeline  # noqa: E402


def main() -> int:
    artifacts = run_geo_transit_rf_pipeline(
        join_how="inner",
        output_dir=ROOT / "outputs" / "geo_transit_rf",
    )
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the secondary transit ↔ CA research-question pipeline.

Example:

    python scripts/run_transit_ca.py
    # or
    ca-personas transit-ca --join inner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca_personas.transit_ca import run_transit_ca_pipeline  # noqa: E402


def main() -> int:
    artifacts = run_transit_ca_pipeline(
        join_how="inner",
        output_dir=ROOT / "outputs" / "transit_ca",
    )
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

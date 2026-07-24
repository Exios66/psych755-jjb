#!/usr/bin/env python3
"""Score and aggregate participant ground-truth PRCA subscales.

Writes evaluation targets used by both ML baselines and LLM persona clones:

- outputs/ground_truth/participants_scored.csv
- outputs/ground_truth/ground_truth.csv
- outputs/ground_truth/ground_truth_aggregates.csv
"""

from __future__ import annotations

import sys

from ca_personas.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["score-gt", *sys.argv[1:]]))

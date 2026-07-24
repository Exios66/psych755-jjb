"""CLI: export CA persona prompts as vLLM ``caseid``/``prompt`` CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.load import load_and_prepare
from ca_personas.personas import RESEARCH_TIERS, TIERS
from inference.ca_prompts import export_vllm_prompt_bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export CA digital-twin persona prompts as a vLLM prompt CSV "
            "(columns: caseid, prompt) plus optional ground-truth answers."
        ),
    )
    ap.add_argument(
        "--prolific",
        type=Path,
        default=Path("data/excerpts/prolific_excerpt.csv"),
    )
    ap.add_argument(
        "--qualtrics",
        type=Path,
        default=Path("data/excerpts/qualtrics_excerpt.csv"),
    )
    ap.add_argument(
        "--join",
        choices=["inner", "outer", "left"],
        default="inner",
    )
    ap.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS),
        default=list(RESEARCH_TIERS) + ["full"],
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vllm_prompts"),
        help="Directory for prompts.csv and ground_truth.csv",
    )
    args = ap.parse_args(argv)

    participants = load_and_prepare(args.prolific, args.qualtrics, how=args.join)
    paths = export_vllm_prompt_bundle(
        participants,
        args.output_dir,
        tiers=args.tiers,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

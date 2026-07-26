"""CLI: export CA persona prompts as vLLM ``caseid``/``prompt`` CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.load import load_and_prepare, load_full_cohort
from ca_personas.paths import default_prolific_paths, default_qualtrics_path
from ca_personas.personas import TIERS
from inference.ca_prompts import export_vllm_prompt_bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export CA digital-twin persona prompts as a vLLM prompt CSV "
            "(columns: caseid, prompt) plus optional ground-truth answers. "
            "Defaults prefer ../sibling_data File A/B/C, else excerpt fixtures."
        ),
    )
    ap.add_argument(
        "--prolific",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Prolific export CSVs. "
            "Default: ../sibling_data File A+B, else data/excerpts/prolific_excerpt.csv"
        ),
    )
    ap.add_argument(
        "--qualtrics",
        type=Path,
        default=None,
        help=(
            "Qualtrics export CSV. "
            "Default: ../sibling_data File C, else data/excerpts/qualtrics_excerpt.csv"
        ),
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
        default=list(TIERS),
        help="Persona tiers to export (default: all 8 core + v3)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vllm_prompts"),
        help="Directory for prompts.csv and ground_truth.csv",
    )
    args = ap.parse_args(argv)

    prolific = list(args.prolific) if args.prolific else default_prolific_paths()
    qualtrics = args.qualtrics or default_qualtrics_path()
    prolific = [Path(p) for p in prolific]
    qualtrics = Path(qualtrics)

    if len(prolific) > 1:
        participants, _report = load_full_cohort(
            prolific_paths=prolific,
            qualtrics_path=qualtrics,
            join_how=args.join,
        )
    else:
        participants = load_and_prepare(
            prolific[0],
            qualtrics,
            how=args.join,
            clean=True,
        )

    paths = export_vllm_prompt_bundle(
        participants,
        args.output_dir,
        tiers=args.tiers,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

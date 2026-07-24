"""Command-line interface for the CA persona framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.ground_truth import export_ground_truth_bundle
from ca_personas.load import load_and_prepare
from ca_personas.personas import RESEARCH_TIERS, TIERS, build_persona_prompts, write_persona_bundle
from ca_personas.pipeline import run_pipeline


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prolific", type=Path, default=None, help="Prolific export CSV")
    parser.add_argument("--qualtrics", type=Path, default=None, help="Qualtrics export CSV")
    parser.add_argument(
        "--join",
        choices=["inner", "outer", "left"],
        default="inner",
        help="Join strategy for Prolific↔Qualtrics (default: inner)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ca-personas",
        description=(
            "Score ground-truth CA, build foolproof persona prompts, predict via "
            "Ollama/OpenRouter, and evaluate exact-score + band accuracy."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # Default / legacy: running with no subcommand still executes the full pipeline.
    run = sub.add_parser("run", help="Full pipeline: GT → personas → LLM predict → evaluate")
    _add_shared_data_args(run)
    run.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS),
        default=None,
        help="Persona tiers (default: research tiers + full)",
    )
    run.add_argument(
        "--provider",
        choices=["ollama", "openrouter", "mock"],
        default=None,
        help="LLM provider (use mock for offline runs)",
    )
    run.add_argument("--model", default=None, help="Model name override")
    run.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--sleep", type=float, default=0.0)

    score = sub.add_parser(
        "score-gt",
        help="Aggregate and score participant ground-truth PRCA subscales",
    )
    _add_shared_data_args(score)
    score.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ground_truth"),
        help="Directory for ground-truth CSV artifacts",
    )

    personas = sub.add_parser(
        "build-personas",
        help="Build foolproof persona prompts from Prolific + Qualtrics characteristics",
    )
    _add_shared_data_args(personas)
    personas.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS),
        default=list(RESEARCH_TIERS) + ["full"],
        help="Which persona tiers to emit",
    )
    personas.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/personas"),
        help="Directory for persona CSV + markdown bundle",
    )

    # Flat args retained so `ca-personas --provider mock` still works.
    _add_shared_data_args(parser)
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=None)
    parser.add_argument("--provider", choices=["ollama", "openrouter", "mock"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser


def _paths_or_defaults(args: argparse.Namespace) -> tuple[Path, Path]:
    prolific = args.prolific or Path("data/excerpts/prolific_excerpt.csv")
    qualtrics = args.qualtrics or Path("data/excerpts/qualtrics_excerpt.csv")
    return Path(prolific), Path(qualtrics)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "score-gt":
        prolific, qualtrics = _paths_or_defaults(args)
        paths = export_ground_truth_bundle(
            prolific,
            qualtrics,
            args.output_dir,
            join_how=args.join,
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return 0

    if command == "build-personas":
        prolific, qualtrics = _paths_or_defaults(args)
        participants = load_and_prepare(prolific, qualtrics, how=args.join)
        prompts = build_persona_prompts(participants, tiers=args.tiers)
        bundle = write_persona_bundle(prompts, args.output_dir)
        print(
            json.dumps(
                {
                    "csv": str(bundle["csv"]),
                    "n_prompts": bundle["n_prompts"],
                    "markdown_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
        return 0

    # Full pipeline
    artifacts = run_pipeline(
        prolific_path=args.prolific,
        qualtrics_path=args.qualtrics,
        tiers=args.tiers,
        provider=args.provider,
        model=args.model,
        config_path=args.config,
        output_dir=args.output_dir,
        sleep_seconds=args.sleep,
        join_how=args.join,
    )
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

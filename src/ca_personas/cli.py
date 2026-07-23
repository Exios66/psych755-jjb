"""Command-line interface for the CA persona framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.personas import TIERS
from ca_personas.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ca-personas",
        description=(
            "Extract participant characteristics, build tiered persona prompts, "
            "predict PRCA scores via Ollama/OpenRouter, and join to ground truth."
        ),
    )
    parser.add_argument(
        "--prolific",
        type=Path,
        default=None,
        help="Path to Prolific export CSV",
    )
    parser.add_argument(
        "--qualtrics",
        type=Path,
        default=None,
        help="Path to Qualtrics export CSV",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS),
        default=None,
        help="Persona information tiers to generate (default: all)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter", "mock"],
        default=None,
        help="LLM provider (default from env/config; use mock for offline runs)",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML config path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for personas/predictions/evaluation artifacts",
    )
    parser.add_argument(
        "--join",
        choices=["inner", "outer", "left"],
        default="inner",
        help="Join strategy for Prolific↔Qualtrics (default: inner)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between LLM calls (seconds)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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

"""Command-line interface for the CA persona framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.compare_agents import run_ml_vs_llm_comparison
from ca_personas.ground_truth import export_ground_truth_bundle
from ca_personas.load import load_and_prepare, load_full_cohort
from ca_personas.paths import default_prolific_paths, default_qualtrics_path
from ca_personas.personas import RESEARCH_TIERS, TIERS, build_persona_prompts, write_persona_bundle
from ca_personas.pipeline import prepare_analytic_sample, run_pipeline
from ca_personas.ca_transit_rf import run_ca_transit_rf_pipeline
from ca_personas.comprehensive_transit_rf import run_comprehensive_transit_rf_pipeline
from ca_personas.geo_transit_rf import run_geo_transit_rf_pipeline
from ca_personas.transit_ca import run_transit_ca_pipeline


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prolific",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Prolific export CSVs (File A + File B). "
            "Default: ../sibling_data/PRCAProlificExport_FileA.csv + FileB, "
            "else data/excerpts/prolific_excerpt.csv"
        ),
    )
    parser.add_argument(
        "--qualtrics",
        type=Path,
        default=None,
        help=(
            "Qualtrics export CSV (File C). "
            "Default: ../sibling_data/PRCAQualtricsExport_FileC.csv, "
            "else data/excerpts/qualtrics_excerpt.csv"
        ),
    )
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
            "Clean & score ground-truth CA, run EDA aligned to the research "
            "questions, build persona prompts, predict via Ollama/OpenRouter, "
            "and evaluate exact-score + band accuracy."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # Default / legacy: running with no subcommand still executes the full pipeline.
    run = sub.add_parser("run", help="Full pipeline: clean → EDA → GT → personas → LLM → evaluate")
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
    run.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip analytic-sample cleaning filters (not recommended for File A/B/C)",
    )
    run.add_argument(
        "--no-eda",
        action="store_true",
        help="Skip writing EDA artifacts under outputs/eda/",
    )

    prepare = sub.add_parser(
        "prepare",
        help="Load sibling-data File A/B/C, clean, score GT, and write EDA (no LLM)",
    )
    _add_shared_data_args(prepare)
    prepare.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs"))

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

    compare = sub.add_parser(
        "compare",
        help="Evaluate RF/KNN vs LLM persona agents on shared CA metrics",
    )
    _add_shared_data_args(compare)
    compare.add_argument(
        "--tiers",
        nargs="+",
        choices=list(RESEARCH_TIERS),
        default=list(RESEARCH_TIERS),
        help="Research tiers to compare (default: all four)",
    )
    compare.add_argument(
        "--provider",
        choices=["ollama", "openrouter", "mock"],
        default="mock",
        help="LLM provider for the persona agent side",
    )
    compare.add_argument("--model", default=None, help="LLM model override")
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml_vs_llm"),
        help="Directory for comparison artifacts",
    )

    transit = sub.add_parser(
        "transit-ca",
        help=(
            "Secondary RQ: test whether regular public-transit riders differ "
            "in CA from the larger matched cohort"
        ),
    )
    _add_shared_data_args(transit)
    transit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transit_ca"),
        help="Directory for transit–CA tables, distributions, and results card",
    )
    transit.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Bootstrap resamples for mean-difference CIs (default: 5000)",
    )
    transit.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")

    geo = sub.add_parser(
        "geo-transit-rf",
        help=(
            "Secondary RQ: Random Forest testing whether lat/long predicts "
            "regular public-transit use"
        ),
    )
    _add_shared_data_args(geo)
    geo.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/geo_transit_rf"),
        help="Directory for RF metrics, importances, OOF predictions, results card",
    )
    geo.add_argument("--splits", type=int, default=5, help="Stratified CV folds")
    geo.add_argument(
        "--perm-repeats",
        type=int,
        default=30,
        help="Permutation-importance repeats",
    )
    geo.add_argument("--seed", type=int, default=42, help="RNG seed")

    ca_rf = sub.add_parser(
        "ca-transit-rf",
        help=(
            "Secondary RQ: Random Forest testing whether group & interpersonal "
            "CA predict regular public-transit use"
        ),
    )
    _add_shared_data_args(ca_rf)
    ca_rf.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ca_transit_rf"),
        help="Directory for RF metrics, importances, OOF predictions, results card",
    )
    ca_rf.add_argument("--splits", type=int, default=5, help="Stratified CV folds")
    ca_rf.add_argument(
        "--perm-repeats",
        type=int,
        default=30,
        help="Permutation-importance repeats",
    )
    ca_rf.add_argument("--seed", type=int, default=42, help="RNG seed")

    comp_rf = sub.add_parser(
        "comprehensive-transit-rf",
        help=(
            "Secondary RQ: feature-importance + tuned Random Forest over "
            "demographics, employment, geo, car access, ride-share, and CA "
            "scores to maximize ROC-AUC for regular transit"
        ),
    )
    _add_shared_data_args(comp_rf)
    comp_rf.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comprehensive_transit_rf"),
        help="Directory for metrics, ablations, importances, and results card",
    )
    comp_rf.add_argument("--splits", type=int, default=5, help="Stratified CV folds")
    comp_rf.add_argument(
        "--perm-repeats",
        type=int,
        default=30,
        help="Permutation-importance repeats",
    )
    comp_rf.add_argument(
        "--tune-iter",
        type=int,
        default=24,
        help="RandomizedSearchCV iterations for ROC-AUC tuning",
    )
    comp_rf.add_argument("--seed", type=int, default=42, help="RNG seed")
    comp_rf.add_argument(
        "--no-upper-bound",
        action="store_true",
        help="Skip the Q27 upper-bound model",
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


def _paths_or_defaults(args: argparse.Namespace) -> tuple[list[Path], Path]:
    prolific = list(args.prolific) if args.prolific else default_prolific_paths()
    qualtrics = args.qualtrics or default_qualtrics_path()
    return [Path(p) for p in prolific], Path(qualtrics)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "prepare":
        artifacts = prepare_analytic_sample(
            prolific_path=args.prolific,
            qualtrics_path=args.qualtrics,
            config_path=args.config,
            join_how=args.join,
            output_dir=args.output_dir,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0

    if command == "score-gt":
        prolific, qualtrics = _paths_or_defaults(args)
        # When multiple Prolific waves are provided, stack via load_full_cohort.
        if len(prolific) > 1:
            participants, report = load_full_cohort(
                prolific_paths=prolific,
                qualtrics_path=qualtrics,
                join_how=args.join,
            )
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            from ca_personas.ground_truth import aggregate_ground_truth, ground_truth_table

            paths = {
                "participants_scored": out / "participants_scored.csv",
                "ground_truth": out / "ground_truth.csv",
                "aggregates": out / "ground_truth_aggregates.csv",
                "cleaning_report": out / "cleaning_report.json",
            }
            participants.to_csv(paths["participants_scored"], index=False)
            ground_truth_table(participants).to_csv(paths["ground_truth"], index=False)
            aggregate_ground_truth(participants).to_csv(paths["aggregates"], index=False)
            paths["cleaning_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            return 0

        paths = export_ground_truth_bundle(
            prolific[0],
            qualtrics,
            args.output_dir,
            join_how=args.join,
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return 0

    if command == "build-personas":
        prolific, qualtrics = _paths_or_defaults(args)
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

    if command == "compare":
        prolific, qualtrics = _paths_or_defaults(args)
        # compare_agents currently takes a single prolific path; stack if needed.
        prolific_arg: Path | list[Path]
        if len(prolific) == 1:
            prolific_arg = prolific[0]
        else:
            from ca_personas.load import load_prolific

            stacked = load_prolific(prolific, wave_labels=("A", "B"))
            tmp = Path("data/processed")
            tmp.mkdir(parents=True, exist_ok=True)
            prolific_arg = tmp / "prolific_stacked.csv"
            export_df = stacked.rename(columns={"participant_id": "Participant id"})
            export_df.to_csv(prolific_arg, index=False)

        result = run_ml_vs_llm_comparison(
            prolific_arg,
            qualtrics,
            tiers=args.tiers,
            llm_provider=args.provider,
            llm_model=args.model,
            join_how=args.join,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {k: str(v) for k, v in result["artifacts"].items()},
                indent=2,
            )
        )
        return 0

    if command == "transit-ca":
        prolific, qualtrics = _paths_or_defaults(args)
        artifacts = run_transit_ca_pipeline(
            prolific_paths=prolific,
            qualtrics_path=qualtrics,
            join_how=args.join,
            output_dir=args.output_dir,
            n_boot=args.n_boot,
            random_state=args.seed,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0

    if command == "geo-transit-rf":
        prolific, qualtrics = _paths_or_defaults(args)
        artifacts = run_geo_transit_rf_pipeline(
            prolific_paths=prolific,
            qualtrics_path=qualtrics,
            join_how=args.join,
            output_dir=args.output_dir,
            n_splits=args.splits,
            n_perm_repeats=args.perm_repeats,
            random_state=args.seed,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0

    if command == "ca-transit-rf":
        prolific, qualtrics = _paths_or_defaults(args)
        artifacts = run_ca_transit_rf_pipeline(
            prolific_paths=prolific,
            qualtrics_path=qualtrics,
            join_how=args.join,
            output_dir=args.output_dir,
            n_splits=args.splits,
            n_perm_repeats=args.perm_repeats,
            random_state=args.seed,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0

    if command == "comprehensive-transit-rf":
        prolific, qualtrics = _paths_or_defaults(args)
        artifacts = run_comprehensive_transit_rf_pipeline(
            prolific_paths=prolific,
            qualtrics_path=qualtrics,
            join_how=args.join,
            output_dir=args.output_dir,
            n_splits=args.splits,
            n_perm_repeats=args.perm_repeats,
            n_tune_iter=args.tune_iter,
            random_state=args.seed,
            include_upper_bound=not args.no_upper_bound,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
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
        clean=not getattr(args, "no_clean", False),
        run_eda_step=not getattr(args, "no_eda", False),
    )
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

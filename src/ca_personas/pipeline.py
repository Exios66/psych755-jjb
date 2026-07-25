"""High-level orchestration for extract → clean → EDA → persona → predict → evaluate."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv

from ca_personas.eda import run_eda
from ca_personas.evaluate import (
    evaluate_predictions,
    summarize_band_confusion,
    summarize_errors,
)
from ca_personas.ground_truth import (
    aggregate_ground_truth,
    ground_truth_table,
)
from ca_personas.llm.base import get_client
from ca_personas.load import load_and_prepare, load_full_cohort
from ca_personas.paths import default_prolific_paths, default_qualtrics_path
from ca_personas.personas import (
    RESEARCH_TIERS,
    TIERS,
    build_persona_prompts,
    write_persona_bundle,
)
from ca_personas.predict import run_predictions

# Re-export for callers / tests that historically imported this symbol here.
__all__ = ["run_pipeline", "prepare_analytic_sample", "load_config", "ensure_dir"]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Raises if the path is set but the file is missing."""
    config_path = Path(path) if path else Path("config/default.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Pass an existing --config path or restore config/default.yaml."
        )
    with config_path.open() as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _existing_files(paths: Sequence[Path]) -> list[Path]:
    return [p for p in paths if p.is_file()]


def _resolve_prolific_paths(
    prolific_path: str | Path | Sequence[str | Path] | None,
    config: dict[str, Any],
) -> list[Path]:
    """Prefer explicit CLI paths, then existing config paths, then sibling/excerpt defaults."""
    if prolific_path is not None:
        if isinstance(prolific_path, (str, Path)):
            return [Path(prolific_path)]
        return [Path(p) for p in prolific_path]

    paths_cfg = config.get("paths", {})
    if "prolific_files" in paths_cfg:
        candidates = [Path(p) for p in paths_cfg["prolific_files"]]
        existing = _existing_files(candidates)
        if existing:
            if len(existing) != len(candidates):
                missing = [str(p) for p in candidates if not p.is_file()]
                warnings.warn(
                    "Some config prolific_files are missing and were skipped: "
                    + ", ".join(missing),
                    stacklevel=2,
                )
            return existing
    if "prolific" in paths_cfg:
        candidate = Path(paths_cfg["prolific"])
        if candidate.is_file():
            return [candidate]
    return default_prolific_paths()


def _resolve_qualtrics_path(
    qualtrics_path: str | Path | None,
    config: dict[str, Any],
) -> Path:
    """Prefer explicit CLI path, then existing config path, then sibling/excerpt default."""
    if qualtrics_path is not None:
        return Path(qualtrics_path)
    paths_cfg = config.get("paths", {})
    if "qualtrics" in paths_cfg:
        candidate = Path(paths_cfg["qualtrics"])
        if candidate.is_file():
            return candidate
    return default_qualtrics_path()


def run_pipeline(
    *,
    prolific_path: str | Path | Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    tiers: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sleep_seconds: float = 0.0,
    join_how: str = "inner",
    clean: bool = True,
    run_eda_step: bool = True,
) -> dict[str, Path]:
    """
    Execute the full research pipeline and write artifacts.

    When ``../sibling_data/`` contains File A/B/C, those exports are preferred
    (stacked Prolific waves + Qualtrics File C). Excerpt fixtures remain the
    fallback for CI / Posit Connect Cloud.

    Returns a dict of artifact paths.
    """
    load_dotenv()
    config = load_config(config_path)
    llm_cfg = config.get("llm", {})
    scoring_cfg = config.get("scoring", {})
    cleaning_cfg = config.get("cleaning", {})
    low_max = int(scoring_cfg.get("band_low_max", 13))
    high_min = int(scoring_cfg.get("band_high_min", 20))
    # CLI --join is authoritative; config documents the project default.
    require_complete_ca = bool(cleaning_cfg.get("require_complete_ca", True))

    prolific_paths = _resolve_prolific_paths(prolific_path, config)
    qualtrics = _resolve_qualtrics_path(qualtrics_path, config)
    selected_tiers = tiers or config.get("tiers", list(RESEARCH_TIERS) + ["full"])
    # Validate tier names early.
    unknown = [t for t in selected_tiers if t not in TIERS]
    if unknown:
        raise ValueError(f"Unknown tiers: {unknown}; expected subset of {TIERS}")

    out_root = Path(output_dir or "outputs")
    personas_dir = ensure_dir(out_root / "personas")
    predictions_dir = ensure_dir(out_root / "predictions")
    evaluation_dir = ensure_dir(out_root / "evaluation")
    gt_dir = ensure_dir(out_root / "ground_truth")
    eda_dir = ensure_dir(out_root / "eda")
    processed_dir = ensure_dir(Path("data/processed"))

    cleaning_report: dict[str, Any] | None = None
    # Prefer the full-cohort loader whenever cleaning is on (audit + consistent filters).
    apply_clean = clean and require_complete_ca
    if apply_clean:
        participants, cleaning_report = load_full_cohort(
            prolific_paths=prolific_paths,
            qualtrics_path=qualtrics,
            join_how=join_how,
            low_max=low_max,
            high_min=high_min,
        )
    else:
        participants = load_and_prepare(
            prolific_paths if len(prolific_paths) > 1 else prolific_paths[0],
            qualtrics,
            how=join_how,
            low_max=low_max,
            high_min=high_min,
            clean=False,
        )

    participants_path = processed_dir / "participants_scored.csv"
    participants.to_csv(participants_path, index=False)

    if cleaning_report is not None:
        report_path = processed_dir / "cleaning_report.json"
        report_path.write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")
    else:
        report_path = None

    eda_artifacts: dict[str, Path] = {}
    if run_eda_step:
        eda_artifacts = run_eda(
            participants,
            eda_dir,
            cleaning_report=cleaning_report,
        )

    # Ground-truth evaluation bundle from the (cleaned) analytic sample.
    gt = ground_truth_table(participants)
    aggregates = aggregate_ground_truth(participants)
    gt_paths = {
        "participants_scored": gt_dir / "participants_scored.csv",
        "ground_truth": gt_dir / "ground_truth.csv",
        "aggregates": gt_dir / "ground_truth_aggregates.csv",
    }
    participants.to_csv(gt_paths["participants_scored"], index=False)
    gt.to_csv(gt_paths["ground_truth"], index=False)
    aggregates.to_csv(gt_paths["aggregates"], index=False)
    # Also mirror compact GT into processed/.
    gt.to_csv(processed_dir / "ground_truth.csv", index=False)
    aggregates.to_csv(processed_dir / "ground_truth_aggregates.csv", index=False)

    prompts = build_persona_prompts(participants, tiers=selected_tiers)
    persona_bundle = write_persona_bundle(prompts, personas_dir)
    prompts_path = Path(persona_bundle["csv"])

    client = get_client(
        provider or llm_cfg.get("provider"),
        model=model,
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 256)),
        timeout_seconds=int(llm_cfg.get("timeout_seconds", 120)),
    )
    predictions = run_predictions(client, prompts, sleep_seconds=sleep_seconds)
    predictions_path = predictions_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    n_errors = int(predictions["error"].notna().sum()) if "error" in predictions.columns else 0
    if len(predictions) and n_errors == len(predictions):
        raise RuntimeError(
            f"All {len(predictions)} LLM predictions failed. "
            f"First error: {predictions.iloc[0].get('error')}"
        )
    if n_errors:
        warnings.warn(
            f"{n_errors}/{len(predictions)} prediction rows failed and will lack scores.",
            stacklevel=2,
        )

    evaluation = evaluate_predictions(
        participants,
        predictions,
        low_max=low_max,
        high_min=high_min,
    )
    evaluation_path = evaluation_dir / "evaluation.csv"
    evaluation.to_csv(evaluation_path, index=False)

    summary = summarize_errors(evaluation)
    summary_path = evaluation_dir / "summary_by_tier.csv"
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        usable = summary.loc[summary["tier"] == "all", "n_with_ground_truth"]
        if len(usable) and int(usable.iloc[0]) == 0:
            raise RuntimeError(
                "Evaluation produced zero rows with ground truth. "
                "Check participant_id alignment between predictions and scored cohort."
            )

    for side in ("group", "interpersonal"):
        confusion = summarize_band_confusion(evaluation, side=side)
        if not confusion.empty:
            confusion.to_csv(evaluation_dir / f"band_confusion_{side}.csv")

    artifacts: dict[str, Path] = {
        "participants": participants_path,
        "ground_truth": gt_paths["ground_truth"],
        "ground_truth_aggregates": gt_paths["aggregates"],
        "prompts": prompts_path,
        "predictions": predictions_path,
        "evaluation": evaluation_path,
        "summary": summary_path,
    }
    if report_path is not None:
        artifacts["cleaning_report"] = report_path
    for key, path in eda_artifacts.items():
        artifacts[f"eda_{key}"] = path
    return artifacts


def prepare_analytic_sample(
    *,
    prolific_path: str | Path | Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    config_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Load → clean → EDA only (no LLM calls). Useful before long prediction runs."""
    config = load_config(config_path)
    scoring_cfg = config.get("scoring", {})
    low_max = int(scoring_cfg.get("band_low_max", 13))
    high_min = int(scoring_cfg.get("band_high_min", 20))

    prolific_paths = _resolve_prolific_paths(prolific_path, config)
    qualtrics = _resolve_qualtrics_path(qualtrics_path, config)

    participants, cleaning_report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics,
        join_how=join_how,
        low_max=low_max,
        high_min=high_min,
    )

    out_root = Path(output_dir or "outputs")
    processed_dir = ensure_dir(Path("data/processed"))
    eda_dir = ensure_dir(out_root / "eda")

    participants_path = processed_dir / "participants_scored.csv"
    participants.to_csv(participants_path, index=False)
    report_path = processed_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")
    eda_artifacts = run_eda(participants, eda_dir, cleaning_report=cleaning_report)

    artifacts = {
        "participants": participants_path,
        "cleaning_report": report_path,
        **{f"eda_{k}": v for k, v in eda_artifacts.items()},
    }
    return artifacts

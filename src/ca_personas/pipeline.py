"""High-level orchestration for extract → persona → predict → evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ca_personas.evaluate import (
    evaluate_predictions,
    summarize_band_confusion,
    summarize_errors,
)
from ca_personas.ground_truth import (
    aggregate_ground_truth,
    export_ground_truth_bundle,
    ground_truth_table,
)
from ca_personas.llm.base import get_client
from ca_personas.load import load_and_prepare
from ca_personas.personas import (
    RESEARCH_TIERS,
    TIERS,
    build_persona_prompts,
    write_persona_bundle,
)
from ca_personas.predict import run_predictions


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else Path("config/default.yaml")
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pipeline(
    *,
    prolific_path: str | Path | None = None,
    qualtrics_path: str | Path | None = None,
    tiers: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sleep_seconds: float = 0.0,
    join_how: str = "inner",
) -> dict[str, Path]:
    """
    Execute the full research pipeline and write artifacts.

    Returns a dict of artifact paths.
    """
    load_dotenv()
    config = load_config(config_path)
    paths = config.get("paths", {})
    llm_cfg = config.get("llm", {})
    scoring_cfg = config.get("scoring", {})
    low_max = int(scoring_cfg.get("band_low_max", 13))
    high_min = int(scoring_cfg.get("band_high_min", 20))

    prolific = Path(prolific_path or paths.get("prolific", "data/excerpts/prolific_excerpt.csv"))
    qualtrics = Path(qualtrics_path or paths.get("qualtrics", "data/excerpts/qualtrics_excerpt.csv"))
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
    processed_dir = ensure_dir(Path("data/processed"))

    participants = load_and_prepare(
        prolific,
        qualtrics,
        how=join_how,
        low_max=low_max,
        high_min=high_min,
    )
    participants_path = processed_dir / "participants_scored.csv"
    participants.to_csv(participants_path, index=False)

    # Standalone ground-truth evaluation bundle (shared by ML + LLM comparisons).
    gt_paths = export_ground_truth_bundle(
        prolific,
        qualtrics,
        gt_dir,
        join_how=join_how,
        low_max=low_max,
        high_min=high_min,
    )
    # Also mirror compact GT into processed/.
    ground_truth_table(participants).to_csv(processed_dir / "ground_truth.csv", index=False)
    aggregate_ground_truth(participants).to_csv(
        processed_dir / "ground_truth_aggregates.csv",
        index=False,
    )

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

    for side in ("group", "interpersonal"):
        confusion = summarize_band_confusion(evaluation, side=side)
        if not confusion.empty:
            confusion.to_csv(evaluation_dir / f"band_confusion_{side}.csv")

    return {
        "participants": participants_path,
        "ground_truth": gt_paths["ground_truth"],
        "ground_truth_aggregates": gt_paths["aggregates"],
        "prompts": prompts_path,
        "predictions": predictions_path,
        "evaluation": evaluation_path,
        "summary": summary_path,
    }

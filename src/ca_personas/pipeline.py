"""High-level orchestration for extract → persona → predict → evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from ca_personas.evaluate import evaluate_predictions, summarize_errors
from ca_personas.llm.base import get_client
from ca_personas.load import load_and_prepare
from ca_personas.personas import TIERS, build_persona_prompts, prompts_to_frame
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

    prolific = Path(prolific_path or paths.get("prolific", "data/excerpts/prolific_excerpt.csv"))
    qualtrics = Path(qualtrics_path or paths.get("qualtrics", "data/excerpts/qualtrics_excerpt.csv"))
    selected_tiers = tiers or config.get("tiers", list(TIERS))
    out_root = Path(output_dir or "outputs")
    personas_dir = ensure_dir(out_root / "personas")
    predictions_dir = ensure_dir(out_root / "predictions")
    evaluation_dir = ensure_dir(out_root / "evaluation")
    processed_dir = ensure_dir(Path("data/processed"))

    participants = load_and_prepare(
        prolific,
        qualtrics,
        how=join_how,
        low_max=int(scoring_cfg.get("band_low_max", 13)),
        high_min=int(scoring_cfg.get("band_high_min", 20)),
    )
    participants_path = processed_dir / "participants_scored.csv"
    participants.to_csv(participants_path, index=False)

    prompts = build_persona_prompts(participants, tiers=selected_tiers)
    prompts_df = prompts_to_frame(prompts)
    prompts_path = personas_dir / "persona_prompts.csv"
    prompts_df.to_csv(prompts_path, index=False)

    # Also write one markdown file per persona for inspection.
    for prompt in prompts:
        md_path = personas_dir / f"{prompt.participant_id}__{prompt.tier}.md"
        md_path.write_text(
            "# System prompt\n\n"
            f"{prompt.system_prompt}\n\n"
            "# User prompt\n\n"
            f"{prompt.user_prompt}\n",
            encoding="utf-8",
        )

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

    evaluation = evaluate_predictions(participants, predictions)
    evaluation_path = evaluation_dir / "evaluation.csv"
    evaluation.to_csv(evaluation_path, index=False)

    summary = summarize_errors(evaluation)
    summary_path = evaluation_dir / "summary_by_tier.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "participants": participants_path,
        "prompts": prompts_path,
        "predictions": predictions_path,
        "evaluation": evaluation_path,
        "summary": summary_path,
    }

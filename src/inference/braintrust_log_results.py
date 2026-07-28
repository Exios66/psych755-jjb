"""CLI: score an existing vLLM results CSV into a Braintrust experiment.

Use after offline GPU runs (or when generation skipped Braintrust) to upload
parse / exact / band / accuracy metrics for prompt comparison.

Example::

    export BRAINTRUST_API_KEY=...
    python -m inference.braintrust_log_results \\
        --result_csv outputs/vllm_results/results.csv \\
        --prompt_csv outputs/vllm_prompts/prompts.csv \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --preset v2_enhanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_personas.personas import SYSTEM_PROMPT
from inference.braintrust_tracing import log_results_csv, resolve_system_prompt

DEFAULT_SYSTEM_MSG = SYSTEM_PROMPT.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Log vLLM result CSVs to Braintrust with CA PRCA scorers.",
    )
    ap.add_argument("--result_csv", required=True, type=Path)
    ap.add_argument(
        "--prompt_csv",
        type=Path,
        default=None,
        help="Optional prompts CSV (caseid, prompt[, answer]) for richer inputs.",
    )
    ap.add_argument("--model", type=str, default="unknown")
    ap.add_argument("--preset", type=str, default="unknown")
    ap.add_argument(
        "--project",
        type=str,
        default=None,
        help="Braintrust project (default: BRAINTRUST_PROJECT / psych755-ca-personas).",
    )
    ap.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: auto from model/preset/time).",
    )
    ap.add_argument(
        "--braintrust",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable Braintrust (default: on when BRAINTRUST_API_KEY set).",
    )
    ap.add_argument(
        "--load-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load system prompt from Braintrust registry when configured.",
    )
    args = ap.parse_args(argv)

    system_msg = DEFAULT_SYSTEM_MSG
    prompt_meta: dict = {"source": "local"}
    if args.load_prompt:
        system_msg, prompt_meta = resolve_system_prompt(
            fallback=DEFAULT_SYSTEM_MSG,
            use_braintrust=args.braintrust,
            project=args.project,
        )

    summary = log_results_csv(
        result_csv=args.result_csv,
        prompt_csv=args.prompt_csv,
        model=args.model,
        preset=args.preset,
        system_msg=system_msg,
        enabled=args.braintrust,
        project=args.project,
        experiment=args.experiment,
        system_prompt_meta=prompt_meta,
    )
    print(json.dumps(summary, indent=2))
    if args.braintrust is True and not summary.get("enabled"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

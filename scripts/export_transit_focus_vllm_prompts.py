#!/usr/bin/env python3
"""Export transit-focus (TF1/TF2) persona prompts as a vLLM prompt CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ca_personas.load import load_full_cohort
from ca_personas.paths import default_prolific_paths, default_qualtrics_path
from ca_personas.transit_ca import label_regular_riders, normalize_q26
from ca_personas.transit_focus import (
    TRANSIT_FOCUS_SYSTEM_PROMPT,
    TRANSIT_FOCUS_TIERS,
    assert_no_mobility_leak,
    build_transit_focus_prompts,
)
from inference.ca_prompts import make_caseid
from inference.utils import normalize_caseid


def _answer_payload(row: pd.Series) -> str | None:
    payload: dict = {}
    if "regular_transit" in row and pd.notna(row.get("regular_transit")):
        payload["gt_regular_transit"] = bool(row["regular_transit"])
    q26 = normalize_q26(row.get("Q26")) if "Q26" in row.index else None
    if q26:
        payload["gt_q26_days"] = q26
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"))


def export_transit_focus_vllm_bundle(
    participants: pd.DataFrame,
    output_dir: str | Path,
    *,
    tiers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    selected = list(tiers) if tiers is not None else list(TRANSIT_FOCUS_TIERS)
    labeled = label_regular_riders(participants)
    prompts = build_transit_focus_prompts(labeled, tiers=selected)

    gt_by_pid: dict[str, pd.Series] = {}
    for _, row in labeled.drop_duplicates("participant_id").iterrows():
        gt_by_pid[normalize_caseid(row["participant_id"])] = row

    rows: list[dict] = []
    for prompt in prompts:
        assert_no_mobility_leak(prompt.user_prompt)
        caseid = make_caseid(prompt.participant_id, prompt.tier)
        row: dict = {
            "caseid": caseid,
            "prompt": prompt.user_prompt,
            "participant_id": prompt.participant_id,
            "tier": prompt.tier,
            "task": "transit_focus",
        }
        gt = gt_by_pid.get(normalize_caseid(prompt.participant_id))
        if gt is not None:
            answer = _answer_payload(gt)
            if answer is not None:
                row["answer"] = answer
        rows.append(row)

    frame = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "prompts.csv"
    frame.to_csv(prompt_path, index=False)

    sys_path = out / "system_prompt.md"
    sys_path.write_text(TRANSIT_FOCUS_SYSTEM_PROMPT.strip() + "\n", encoding="utf-8")

    paths: dict[str, Path] = {"prompts": prompt_path, "system_prompt": sys_path}
    if "answer" in frame.columns and frame["answer"].notna().any():
        truth = frame.loc[frame["answer"].notna(), ["caseid", "answer"]].copy()
        truth_path = out / "ground_truth.csv"
        truth.to_csv(truth_path, index=False)
        paths["ground_truth"] = truth_path
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export TF1/TF2 transit-focus persona prompts for vLLM "
            "(mobility held out; predict regular_transit + Q26)."
        ),
    )
    ap.add_argument("--join", choices=["inner", "outer", "left"], default="inner")
    ap.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TRANSIT_FOCUS_TIERS),
        default=list(TRANSIT_FOCUS_TIERS),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vllm_prompts_transit_focus"),
    )
    args = ap.parse_args(argv)

    participants, _ = load_full_cohort(
        prolific_paths=default_prolific_paths(),
        qualtrics_path=default_qualtrics_path(),
        join_how=args.join,
    )
    paths = export_transit_focus_vllm_bundle(
        participants,
        args.output_dir,
        tiers=args.tiers,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    n = len(pd.read_csv(paths["prompts"]))
    print(f"n_prompts={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

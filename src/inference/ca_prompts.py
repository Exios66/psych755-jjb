"""Bridge CA persona prompts ↔ vLLM digital-twin CSV schema (caseid, prompt).

Persona prompts in ``ca_personas`` use ``participant_id`` + ``tier`` with separate
system/user strings.  The vLLM launcher expects a flat CSV::

    caseid,prompt[,answer]

where ``caseid`` is a unique row key (``{participant_id}__{tier}``) and ``prompt``
is the user-facing persona text.  The CA system instruction lives in
``predict_vllm.DEFAULT_SYSTEM_MSG`` (applied at generation time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ca_personas.llm.base import extract_json_object, validate_prediction
from ca_personas.personas import PersonaPrompt, build_persona_prompts
from inference.utils import find_duplicate_caseids, normalize_caseid


def make_caseid(participant_id: str, tier: str) -> str:
    """Stable unique key for one participant × tier prompt row."""
    pid = normalize_caseid(participant_id)
    t = str(tier).strip()
    if not pid or not t:
        raise ValueError(f"Invalid participant_id/tier for caseid: {participant_id!r}, {tier!r}")
    return f"{pid}__{t}"


def parse_caseid(caseid: str) -> tuple[str, str]:
    """Split ``{participant_id}__{tier}`` back into components."""
    cid = normalize_caseid(caseid)
    if "__" not in cid:
        raise ValueError(f"caseid must look like '<participant_id>__<tier>': {caseid!r}")
    pid, tier = cid.rsplit("__", 1)
    if not pid or not tier:
        raise ValueError(f"Malformed caseid: {caseid!r}")
    return pid, tier


def _answer_payload(row: pd.Series) -> str | None:
    """Serialize ground-truth CA scores into the optional ``answer`` column."""
    group = row.get("gt_group_ca")
    interpersonal = row.get("gt_interpersonal_ca")
    if pd.isna(group) and pd.isna(interpersonal):
        return None
    payload: dict[str, Any] = {}
    if not pd.isna(group):
        payload["gt_group_ca"] = int(group)
    if not pd.isna(interpersonal):
        payload["gt_interpersonal_ca"] = int(interpersonal)
    if "gt_group_band" in row and not pd.isna(row.get("gt_group_band")):
        payload["gt_group_band"] = str(row["gt_group_band"])
    if "gt_interpersonal_band" in row and not pd.isna(row.get("gt_interpersonal_band")):
        payload["gt_interpersonal_band"] = str(row["gt_interpersonal_band"])
    return json.dumps(payload, separators=(",", ":"))


def personas_to_prompt_frame(
    prompts: list[PersonaPrompt],
    participants: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert ``PersonaPrompt`` objects into a vLLM prompt CSV frame."""
    gt_by_pid: dict[str, pd.Series] = {}
    if participants is not None and not participants.empty:
        for _, row in participants.drop_duplicates("participant_id").iterrows():
            gt_by_pid[normalize_caseid(row["participant_id"])] = row

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        caseid = make_caseid(prompt.participant_id, prompt.tier)
        row: dict[str, Any] = {
            "caseid": caseid,
            "prompt": prompt.user_prompt,
            "participant_id": prompt.participant_id,
            "tier": prompt.tier,
        }
        gt_row = gt_by_pid.get(normalize_caseid(prompt.participant_id))
        if gt_row is not None:
            answer = _answer_payload(gt_row)
            if answer is not None:
                row["answer"] = answer
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["caseid", "prompt", "participant_id", "tier"])

    dupes = find_duplicate_caseids(frame["caseid"])
    if dupes:
        sample = ", ".join(dupes[:5])
        raise ValueError(f"Duplicate caseids in prompt export: {sample}")
    return frame


def export_vllm_prompt_bundle(
    participants: pd.DataFrame,
    output_dir: str | Path,
    *,
    tiers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Build persona prompts and write vLLM-ready CSVs under *output_dir*.

    Writes
    ------
    prompts.csv
        ``caseid``, ``prompt`` (+ metadata columns ``participant_id``, ``tier``).
    ground_truth.csv
        ``caseid``, ``answer`` (JSON with gt CA scores) when scores are available.
    """
    from ca_personas.personas import RESEARCH_TIERS

    selected = list(tiers) if tiers is not None else list(RESEARCH_TIERS) + ["full"]
    prompts = build_persona_prompts(participants, tiers=selected)
    frame = personas_to_prompt_frame(prompts, participants)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "prompts.csv"
    # Keep metadata for debugging; vLLM only requires caseid + prompt.
    frame.to_csv(prompt_path, index=False)

    paths: dict[str, Path] = {"prompts": prompt_path}
    if "answer" in frame.columns and frame["answer"].notna().any():
        truth = frame.loc[frame["answer"].notna(), ["caseid", "answer"]].copy()
        truth_path = out / "ground_truth.csv"
        truth.to_csv(truth_path, index=False)
        paths["ground_truth"] = truth_path
    return paths


def results_to_predictions(result_csv: str | Path) -> pd.DataFrame:
    """Parse a vLLM result CSV (``caseid``, ``generated_text``) into CA predictions."""
    df = pd.read_csv(result_csv)
    if "caseid" not in df.columns or "generated_text" not in df.columns:
        raise ValueError("Result CSV must have columns: caseid, generated_text")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        caseid = normalize_caseid(row["caseid"])
        try:
            participant_id, tier = parse_caseid(caseid)
        except ValueError as exc:
            rows.append(
                {
                    "caseid": caseid,
                    "participant_id": None,
                    "tier": None,
                    "provider": "vllm",
                    "model": None,
                    "raw_response": row.get("generated_text", ""),
                    "error": f"ValueError: {exc}",
                    "pred_group_ca": None,
                    "pred_interpersonal_ca": None,
                    "pred_group_band": None,
                    "pred_interpersonal_band": None,
                }
            )
            continue

        raw = "" if pd.isna(row.get("generated_text")) else str(row["generated_text"])
        error: str | None = None
        parsed = {
            "pred_group_ca": None,
            "pred_interpersonal_ca": None,
            "pred_group_band": None,
            "pred_interpersonal_band": None,
        }
        try:
            payload = extract_json_object(raw)
            parsed = validate_prediction(payload)
        except Exception as exc:  # noqa: BLE001 - capture per-row parse failures
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "caseid": caseid,
                "participant_id": participant_id,
                "tier": tier,
                "provider": "vllm",
                "model": None,
                "raw_response": raw,
                "error": error,
                **parsed,
            }
        )
    return pd.DataFrame(rows)

"""Shared helpers for digital-twin batch inference (caseid / prompt CSV schema)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_caseid(value: object) -> str:
    """Normalize a caseid for stable join / checkpoint-resume keys.

    Float-like values such as ``123.0`` (common after CSV round-trips) become
    ``"123"``.  Missing values become an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "<na>"}:
        return ""
    if text.endswith(".0"):
        head = text[:-2]
        if head.isdigit() or (head.startswith("-") and head[1:].isdigit()):
            return head
    return text


def find_duplicate_caseids(caseids: pd.Series | list[str]) -> list[str]:
    """Return caseids that appear more than once (stable order of first sighting)."""
    series = pd.Series(caseids).map(normalize_caseid)
    series = series[series != ""]
    if series.empty:
        return []
    counts = series.value_counts()
    dupes = counts[counts > 1].index.tolist()
    # Preserve first-seen order from the original series.
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in series.tolist():
        if cid in dupes and cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def convert_prompt_to_messages(
    prompt: str,
    system_msg: str | None = None,
) -> list[dict[str, str]]:
    """Wrap a raw prompt string as chat messages for ``apply_chat_template``.

    Parameters
    ----------
    prompt:
        User-facing persona / digital-twin prompt text.
    system_msg:
        Optional system instruction prepended as a ``system`` role message.
    """
    messages: list[dict[str, str]] = []
    if system_msg:
        messages.append({"role": "system", "content": str(system_msg)})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def read_completed_caseids(result_csv: str | Path) -> set[str]:
    """Validate a checkpoint CSV and return already-completed caseids.

    Raises
    ------
    ValueError
        If the file is empty, missing a ``caseid`` column, or contains
        duplicate caseids.
    """
    path = Path(result_csv)
    if not path.is_file():
        return set()

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Existing result CSV is empty (no header row): {path}") from exc

    if df.empty and list(df.columns) == []:
        raise ValueError(f"Existing result CSV is empty (no header row): {path}")

    if "caseid" not in df.columns:
        raise ValueError(
            f"Existing result CSV has no 'caseid' column: {path}. "
            "Delete or repair the file before resuming."
        )

    normalized = df["caseid"].map(normalize_caseid)
    if (normalized == "").any():
        raise ValueError(
            f"Existing result CSV contains blank caseid values: {path}. "
            "Delete or repair the file before resuming."
        )

    dupes = find_duplicate_caseids(normalized)
    if dupes:
        sample = ", ".join(dupes[:5])
        suffix = " ..." if len(dupes) > 5 else ""
        raise ValueError(
            "Existing result CSV contains duplicate caseid values "
            f"({sample}{suffix}): {path}"
        )

    return set(normalized.tolist())


def resolve_hf_token(token_file: str | None = None) -> str | None:
    """Resolve a Hugging Face token from a file or environment variables.

    Lookup order:
    1. ``token_file`` if it exists and is non-empty (also checked relative to
       the repository root).
    2. ``HF_TOKEN``
    3. ``HUGGINGFACE_HUB_TOKEN``
    4. ``HUGGING_FACE_HUB_TOKEN``
    """
    candidates: list[Path] = []
    if token_file:
        raw = Path(token_file)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(_REPO_ROOT / raw)

    for path in candidates:
        try:
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except OSError:
            continue

    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value
    return None

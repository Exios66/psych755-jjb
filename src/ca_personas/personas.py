"""Tiered persona prompt construction from participant characteristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

TIERS = ("demos", "employment", "geo", "transit")

SYSTEM_PROMPT = """You are taking part in a research simulation. You will be assigned an
identity — a specific person's demographic and behavioral profile. Fully adopt this
identity and answer as if you ARE this person, in first person.

You will then rate your own communication apprehension using McCroskey's PRCA scale
logic: for each of two contexts (group discussions, and one-on-one conversations with
new people), report how anxious/apprehensive YOU (in this identity) would say you feel,
as an integer from 6 (very low apprehension) to 30 (very high apprehension).

Do not break character or mention that you are an AI. Do not add caveats about
uncertainty in your output — give your best first-person self-report, as a real survey
respondent would.

Respond with ONLY a JSON object, no other text:
{
  "self_reported_group_ca": <integer 6-30>,
  "self_reported_interpersonal_ca": <integer 6-30>,
  "self_reported_band_group": "low" | "moderate" | "high",
  "self_reported_band_interpersonal": "low" | "moderate" | "high"
}
"""


@dataclass(frozen=True)
class PersonaPrompt:
    participant_id: str
    tier: str
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "participant_id": self.participant_id,
            "tier": self.tier,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "na"}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "unknown"
        # Keep lat/long readable without excessive precision.
        return f"{value:.4f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text if text else "unknown"


def _line(label: str, value: Any) -> str | None:
    if not _present(value):
        return None
    return f"- {label}: {_fmt(value)}"


def demos_block(row: pd.Series) -> list[str]:
    lines = [
        _line("Age", row.get("Age")),
        _line("Sex", row.get("Sex")),
        _line("Ethnicity", row.get("Ethnicity simplified")),
        _line("Country of birth", row.get("Country of birth")),
        _line("Country of residence", row.get("Country of residence")),
        _line("Nationality", row.get("Nationality")),
        _line("Primary language", row.get("Language")),
        _line("Student status", row.get("Student status")),
    ]
    return [line for line in lines if line]


def employment_block(row: pd.Series) -> list[str]:
    line = _line("Employment status", row.get("Employment status"))
    return [line] if line else []


def geo_block(row: pd.Series) -> list[str]:
    lines = [
        _line("Country of residence", row.get("Country of residence")),
        _line("Approximate latitude", row.get("LocationLatitude")),
        _line("Approximate longitude", row.get("LocationLongitude")),
    ]
    return [line for line in lines if line]


def transit_block(row: pd.Series) -> list[str]:
    lines = [
        _line(
            "Public transportation days (last 3 months)",
            row.get("Q26"),
        ),
        _line("Typical public-transit rides per day", row.get("Q27")),
        _line("Ride-share days (last 3 months)", row.get("Q28")),
        _line("Typical ride-share rides per day", row.get("Q29")),
        _line("Has a driver's license", row.get("Q20")),
        _line("Has access to a car", row.get("Q21")),
    ]
    return [line for line in lines if line]


def build_profile_sections(row: pd.Series, tier: str) -> list[tuple[str, list[str]]]:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS}")

    sections: list[tuple[str, list[str]]] = [("Demographics", demos_block(row))]
    if tier in {"employment", "geo", "transit"}:
        sections.append(("Employment", employment_block(row)))
    if tier in {"geo", "transit"}:
        sections.append(("Geographic location", geo_block(row)))
    if tier == "transit":
        sections.append(("Transportation use", transit_block(row)))
    return [(title, lines) for title, lines in sections if lines]


def build_user_prompt(row: pd.Series, tier: str) -> str:
    pid = _fmt(row.get("participant_id"))
    sections = build_profile_sections(row, tier)
    body_parts: list[str] = []
    for title, lines in sections:
        body_parts.append(f"{title}:\n" + "\n".join(lines))
    profile = "\n\n".join(body_parts) if body_parts else "- [No profile fields available]"

    return (
        f"Adopt the following identity (participant {pid}). "
        "Use only this profile; do not invent extra biography beyond what is listed.\n\n"
        f"{profile}\n\n"
        "Now, as this person, estimate your communication apprehension subscale scores:\n"
        "1) Group discussion apprehension (integer 6–30)\n"
        "2) Interpersonal / one-on-one conversation apprehension (integer 6–30)\n\n"
        "Return ONLY the JSON object specified in the system instructions."
    )


def build_persona_prompt(row: pd.Series, tier: str) -> PersonaPrompt:
    pid = str(row.get("participant_id", "")).strip()
    if not pid:
        raise ValueError("Row is missing participant_id")
    return PersonaPrompt(
        participant_id=pid,
        tier=tier,
        system_prompt=SYSTEM_PROMPT.strip(),
        user_prompt=build_user_prompt(row, tier),
    )


def build_persona_prompts(
    df: pd.DataFrame,
    tiers: list[str] | tuple[str, ...] = TIERS,
    *,
    require_demographics: bool = True,
) -> list[PersonaPrompt]:
    """Build persona prompts for each participant × tier."""
    prompts: list[PersonaPrompt] = []
    for _, row in df.iterrows():
        pid = row.get("participant_id")
        if not _present(pid):
            continue
        if require_demographics and not demos_block(row):
            continue
        for tier in tiers:
            prompts.append(build_persona_prompt(row, tier))
    return prompts


def prompts_to_frame(prompts: list[PersonaPrompt]) -> pd.DataFrame:
    return pd.DataFrame([p.to_dict() for p in prompts])

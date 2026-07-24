"""Tiered / full persona prompt construction from participant characteristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Cumulative research tiers, plus a foolproof "full" profile that includes
# every available Qualtrics + Prolific characteristic used for personification.
TIERS = ("demos", "employment", "geo", "transit", "full")
RESEARCH_TIERS = ("demos", "employment", "geo", "transit")

SYSTEM_PROMPT = """You are taking part in a research simulation. You will be assigned an
identity — a specific person's demographic and behavioral profile. Fully adopt this
identity and answer as if you ARE this person, in first person.

Stay in character for the entire response. Speak and reason from this person's lived
context (age, work situation, place, travel habits, and any self-described attitudes
included in the profile). Do not invent biography that contradicts the profile; you may
only elaborate lightly in ways that are consistent with the listed facts.

You will then rate your own communication apprehension using McCroskey's PRCA scale
logic: for each of two contexts (group discussions, and one-on-one conversations with
new people), report how anxious/apprehensive YOU (in this identity) would say you feel,
as an integer from 6 (very low apprehension) to 30 (very high apprehension).

Also classify each score into the standard classroom bands:
- low: 6–13
- moderate: 14–19
- high: 20–30

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

    def to_markdown(self) -> str:
        return (
            f"# Persona — {self.participant_id} ({self.tier})\n\n"
            "## System prompt\n\n"
            f"{self.system_prompt}\n\n"
            "## User prompt\n\n"
            f"{self.user_prompt}\n"
        )


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
        return f"{value:.4f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text if text else "unknown"


def _line(label: str, value: Any) -> str | None:
    if not _present(value):
        return None
    return f"- {label}: {_fmt(value)}"


def _paragraph(label: str, value: Any) -> str | None:
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
        _line("Survey language", row.get("UserLanguage")),
    ]
    return [line for line in lines if line]


def transit_block(row: pd.Series) -> list[str]:
    lines = [
        _line("Public transportation days (last 3 months)", row.get("Q26")),
        _line("Typical public-transit rides per day", row.get("Q27")),
        _line("Ride-share days (last 3 months)", row.get("Q28")),
        _line("Typical ride-share rides per day", row.get("Q29")),
        _line("Has a driver's license", row.get("Q20")),
        _line("Has access to a car", row.get("Q21")),
    ]
    return [line for line in lines if line]


def voice_block(row: pd.Series) -> list[str]:
    """Open-text Qualtrics items that help the model personify attitudes."""
    lines = [
        _paragraph(
            "Advice I would give a friend who gets nervous talking to new people",
            row.get("Q18_advice"),
        ),
        _paragraph(
            "My ideal way to get around my city (and why)",
            row.get("Q19"),
        ),
    ]
    return [line for line in lines if line]


def build_profile_sections(row: pd.Series, tier: str) -> list[tuple[str, list[str]]]:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS}")

    if tier == "full":
        sections: list[tuple[str, list[str]]] = [
            ("Demographics", demos_block(row)),
            ("Employment", employment_block(row)),
            ("Geographic location", geo_block(row)),
            ("Transportation use", transit_block(row)),
            ("Self-described attitudes (from survey free responses)", voice_block(row)),
        ]
        return [(title, lines) for title, lines in sections if lines]

    sections = [("Demographics", demos_block(row))]
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

    personify = (
        "Fully personify this individual. Answer as this person would — using the "
        "listed facts as constraints — and estimate how communication-anxious you "
        "feel in group discussions and in one-on-one conversations with new people."
    )
    if tier == "full":
        personify += (
            " Weight the free-response attitudes heavily when they are present; they "
            "are this person's own words about social nervousness and daily mobility."
        )

    return (
        f"Adopt the following identity (participant {pid}). "
        "Use only this profile; do not invent extra biography beyond what is listed.\n\n"
        f"{profile}\n\n"
        f"{personify}\n\n"
        "Report:\n"
        "1) Group discussion apprehension (integer 6–30) and its band "
        "(low / moderate / high)\n"
        "2) Interpersonal / one-on-one conversation apprehension (integer 6–30) "
        "and its band (low / moderate / high)\n\n"
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
    tiers: list[str] | tuple[str, ...] = RESEARCH_TIERS,
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


def write_persona_bundle(
    prompts: list[PersonaPrompt],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write CSV index + per-persona markdown files for inspection / LLM intake."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = prompts_to_frame(prompts)
    csv_path = out / "persona_prompts.csv"
    frame.to_csv(csv_path, index=False)
    md_paths: list[Path] = []
    for prompt in prompts:
        md_path = out / f"{prompt.participant_id}__{prompt.tier}.md"
        md_path.write_text(prompt.to_markdown(), encoding="utf-8")
        md_paths.append(md_path)
    return {"csv": csv_path, "markdown_files": md_paths, "n_prompts": len(prompts)}

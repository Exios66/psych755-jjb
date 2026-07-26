"""Tiered / full persona prompt construction from participant characteristics.

Prompt framing follows the AI Terrarium / ICA 2026 digital-twin practice:
natural-language, second-person narratives that *convey* a persona to the
model ("You are a …"), rather than structured questionnaire bullet lists or
meta-instructions to "adopt" / "personify" a profile. Cumulative context
depth mirrors the project's progressive information tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Cumulative research tiers, plus a foolproof "full" profile that includes
# every available Qualtrics + Prolific characteristic used for personification.
TIERS = ("demos", "employment", "geo", "transit", "full")
RESEARCH_TIERS = ("demos", "employment", "geo", "transit")

# Base demographics layer shared by every tier (File A/B core set).
# Optional Prolific fields (ethnicity / nationality / language / birth country)
# are appended when present in the export.
BASE_DEMO_FIELDS = (
    "Age",
    "Sex",
    "Country of residence",
    "Student status",
)
OPTIONAL_DEMO_FIELDS = (
    "Ethnicity simplified",
    "Country of birth",
    "Nationality",
    "Language",
)

# Digital-twin system framing: inhabit the conveyed persona; emit CA JSON.
# Kept intentionally short — the user prompt *is* the persona narrative.
SYSTEM_PROMPT = """You inhabit the identity described to you. Answer as that
person, in first person, from their lived context (age, student status, work
situation, place, travel habits, and any self-described attitudes conveyed in
the profile). Do not invent biography that contradicts the profile; you may
only elaborate lightly in ways consistent with what was told to you.

Use the profile as context, but do not treat any single life circumstance as
deterministically fixing your apprehension. Rate the two communication contexts
independently — group discussion anxiety and one-on-one conversation anxiety
can differ.

Rate your own communication apprehension using McCroskey's PRCA scale logic:
for each of two contexts (group discussions, and one-on-one conversations with
new people), report how anxious/apprehensive YOU feel, as an integer from 6
(very low apprehension) to 30 (very high apprehension).

Also classify each score into the standard classroom bands:
- low: 6–13
- moderate: 14–19
- high: 20–30

Do not break character or mention that you are an AI. Do not add caveats about
uncertainty — give your best first-person self-report, as a real survey
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


def _coord_1dp(value: Any) -> str | None:
    """Format a coordinate to one decimal place (approx. place cue, less noise)."""
    if not _present(value):
        return None
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return None


def _is_never(value: Any) -> bool:
    return _present(value) and str(value).strip().lower() == "never"


def _sex_noun(sex: Any) -> str | None:
    if not _present(sex):
        return None
    text = str(sex).strip().lower()
    if text == "female":
        return "woman"
    if text == "male":
        return "man"
    return str(sex).strip()


def _yes_no(value: Any) -> bool | None:
    if not _present(value):
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None


def _place(value: Any) -> str:
    """Format a place name for fluent English (e.g. 'the United States')."""
    text = _fmt(value)
    specials = {
        "united states": "the United States",
        "united states of america": "the United States",
        "usa": "the United States",
        "us": "the United States",
        "united kingdom": "the United Kingdom",
        "uk": "the United Kingdom",
        "netherlands": "the Netherlands",
        "czech republic": "the Czech Republic",
    }
    return specials.get(text.lower(), text)


def _rides_phrase(value: Any) -> str:
    """Normalize Qualtrics rides-per-day choice text for narrative use."""
    text = _fmt(value)
    # Choice labels often end with "in a typical day"; the surrounding sentence
    # already supplies that frame, so drop the redundant tail.
    lowered = text.lower()
    suffix = " in a typical day"
    if lowered.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _employment_sentence(status: Any) -> str | None:
    if not _present(status):
        return None
    text = str(status).strip()
    lowered = text.lower()
    if lowered in {"full-time", "full time"}:
        return "You work full-time."
    if lowered in {"part-time", "part time"}:
        return "You work part-time."
    if "unemployed" in lowered and "job seeking" in lowered:
        return "You are unemployed and job seeking."
    if lowered.startswith("unemployed"):
        return "You are unemployed."
    if "not in paid work" in lowered:
        return (
            "You are not in paid work (for example, a homemaker, retired, "
            "or disabled)."
        )
    if lowered in {"retired"}:
        return "You are retired."
    if lowered in {"student"}:
        return "You are a student for employment purposes."
    return f"Your employment status is {text}."


# Qualtrics transportation instrument stems (File C / excerpt header labels).
# Used when paraphrasing transit items into natural-language digital-twin prose.
TRANSIT_ITEM_LABELS = {
    "Q26": (
        "In the last three months, on how many days did you use public "
        "transportation (bus, train, tram, etc.)"
    ),
    "Q27": "On a typical day of public transportation use, how many rides do you take",
    "Q28": (
        "In the last three months, on how many days did you use ride share "
        "platforms (Lyft, Uber, DiDi, etc.)"
    ),
    "Q29": "On a typical day of ride share use, how many rides do you take",
    "Q20": "Do you have a license to drive a car",
    "Q21": "Do you have access to a car you can use for transportation",
}


def demos_sentences(row: pd.Series) -> list[str]:
    """Natural-language demographics sentences (AI Terrarium narrative style)."""
    sentences: list[str] = []

    age = row.get("Age")
    sex_noun = _sex_noun(row.get("Sex"))
    ethnicity = _fmt(row.get("Ethnicity simplified")) if _present(row.get("Ethnicity simplified")) else None
    residence = _place(row.get("Country of residence")) if _present(row.get("Country of residence")) else None

    opener_bits: list[str] = []
    if _present(age):
        try:
            age_num = int(float(age))
            age_phrase = f"{age_num}-year-old"
        except (TypeError, ValueError):
            age_phrase = f"{_fmt(age)}-year-old"
        opener_bits.append(age_phrase)
    if ethnicity:
        opener_bits.append(ethnicity)
    if sex_noun:
        opener_bits.append(sex_noun)

    if opener_bits and residence:
        sentences.append(f"You are a {' '.join(opener_bits)} living in {residence}.")
    elif opener_bits:
        sentences.append(f"You are a {' '.join(opener_bits)}.")
    elif residence:
        sentences.append(f"You live in {residence}.")

    if _present(row.get("Country of birth")):
        sentences.append(f"You were born in {_place(row.get('Country of birth'))}.")
    if _present(row.get("Nationality")):
        sentences.append(f"Your nationality is {_fmt(row.get('Nationality'))}.")
    if _present(row.get("Language")):
        sentences.append(f"Your primary language is {_fmt(row.get('Language'))}.")

    student = _yes_no(row.get("Student status"))
    if student is True:
        sentences.append("You are a student.")
    elif student is False:
        sentences.append("You are not a student.")
    elif _present(row.get("Student status")):
        sentences.append(f"Your student status is {_fmt(row.get('Student status'))}.")

    return sentences


def demos_block(row: pd.Series) -> list[str]:
    """Compatibility alias: narrative demographic sentences for the demos layer."""
    return demos_sentences(row)


def employment_sentences(row: pd.Series) -> list[str]:
    sentence = _employment_sentence(row.get("Employment status"))
    return [sentence] if sentence else []


def employment_block(row: pd.Series) -> list[str]:
    return employment_sentences(row)


def geo_sentences(row: pd.Series) -> list[str]:
    """Approximate place cue; avoid repeating country already stated in demos.

    Cumulative tiers always include the demos paragraph first, so country of
    residence is not restated here. When coordinates are missing, this layer
    contributes nothing (country remains in demos).
    """
    lat = _coord_1dp(row.get("LocationLatitude"))
    lon = _coord_1dp(row.get("LocationLongitude"))
    if lat is None or lon is None:
        return []
    # Rounded coords keep geographic signal without false-precision noise.
    return [
        f"Your approximate survey location is near latitude {lat} "
        f"and longitude {lon}."
    ]


def geo_block(row: pd.Series) -> list[str]:
    return geo_sentences(row)


def _rides_intensity_sentence(value: Any, *, mode: str) -> str | None:
    """Build a rides-per-day sentence; *mode* is 'public transportation' or 'ride share'."""
    if not _present(value):
        return None
    rides = _rides_phrase(value)
    if "ride" in rides.lower():
        return f"On a typical day of {mode} use, you take {rides}."
    return f"On a typical day of {mode} use, you take {rides} rides."


def transit_sentences(row: pd.Series) -> list[str]:
    """Paraphrase transit items signal-first: Q26 → Q28 → intensity → car access.

    Skips rides-per-day (Q27/Q29) when the matching frequency is Never — those
    intensity items are nonsensical without use and add token noise.
    """
    sentences: list[str] = []

    # 1) Public-transit frequency (Q26) — primary mobility cue for CA.
    q26_never = _is_never(row.get("Q26"))
    if _present(row.get("Q26")):
        freq = _fmt(row.get("Q26"))
        if q26_never:
            sentences.append(
                "In the last three months, you never used public transportation "
                "(bus, train, tram, etc.)."
            )
        else:
            sentences.append(
                "In the last three months, you used public transportation "
                f"(bus, train, tram, etc.) {freq}."
            )
            intensity = _rides_intensity_sentence(row.get("Q27"), mode="public transportation")
            if intensity:
                sentences.append(intensity)

    # 2) Ride-share frequency (Q28) — strongest tabular CA covariate in this cohort.
    q28_never = _is_never(row.get("Q28"))
    if _present(row.get("Q28")):
        freq = _fmt(row.get("Q28"))
        if q28_never:
            sentences.append(
                "In the last three months, you never used ride share platforms "
                "(Lyft, Uber, DiDi, etc.)."
            )
        else:
            sentences.append(
                "In the last three months, you used ride share platforms "
                f"(Lyft, Uber, DiDi, etc.) {freq}."
            )
            intensity = _rides_intensity_sentence(row.get("Q29"), mode="ride share")
            if intensity:
                sentences.append(intensity)

    # 3) License / car access after frequency (weaker CA signal; RQ2 “used sensibly”).
    license_yn = _yes_no(row.get("Q20"))
    if license_yn is True:
        sentences.append("You have a license to drive a car.")
    elif license_yn is False:
        sentences.append("You do not have a license to drive a car.")
    elif _present(row.get("Q20")):
        sentences.append(
            f"Regarding a license to drive a car, your answer is {_fmt(row.get('Q20'))}."
        )

    car_yn = _yes_no(row.get("Q21"))
    if car_yn is True:
        sentences.append("You have access to a car you can use for transportation.")
    elif car_yn is False:
        sentences.append(
            "You do not have access to a car you can use for transportation."
        )
    elif _present(row.get("Q21")):
        sentences.append(
            "Regarding access to a car you can use for transportation, your "
            f"answer is {_fmt(row.get('Q21'))}."
        )

    return sentences


def transit_block(row: pd.Series) -> list[str]:
    return transit_sentences(row)


def voice_sentences(row: pd.Series) -> list[str]:
    """Open-text Qualtrics items paraphrased as the twin's own words."""
    sentences: list[str] = []
    if _present(row.get("Q18_advice")):
        advice = _fmt(row.get("Q18_advice"))
        sentences.append(
            "When a friend gets nervous talking to new people, you would advise: "
            f'"{advice}"'
        )
    if _present(row.get("Q19")):
        ideal = _fmt(row.get("Q19"))
        sentences.append(
            f'Your ideal way to get around your city is: "{ideal}"'
        )
    return sentences


def voice_block(row: pd.Series) -> list[str]:
    return voice_sentences(row)


def build_narrative_sections(row: pd.Series, tier: str) -> list[str]:
    """Return ordered narrative paragraphs for the requested context depth."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS}")

    paragraphs: list[str] = []

    demos = demos_sentences(row)
    if demos:
        paragraphs.append(" ".join(demos))

    include_employment = tier in {"employment", "geo", "transit", "full"}
    include_geo = tier in {"geo", "transit", "full"}
    include_transit = tier in {"transit", "full"}
    include_voice = tier == "full"

    if include_employment:
        emp = employment_sentences(row)
        if emp:
            paragraphs.append(" ".join(emp))
    if include_geo:
        geo = geo_sentences(row)
        if geo:
            paragraphs.append(" ".join(geo))
    if include_transit:
        transit = transit_sentences(row)
        if transit:
            paragraphs.append(" ".join(transit))
    if include_voice:
        voice = voice_sentences(row)
        if voice:
            paragraphs.append(" ".join(voice))

    return paragraphs


def build_profile_sections(row: pd.Series, tier: str) -> list[tuple[str, list[str]]]:
    """Legacy section view used by tests / inspection; wraps narrative sentences."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {TIERS}")

    if tier == "full":
        sections: list[tuple[str, list[str]]] = [
            ("Demographics", demos_sentences(row)),
            ("Employment", employment_sentences(row)),
            ("Geographic location", geo_sentences(row)),
            ("Transportation use", transit_sentences(row)),
            ("Self-described attitudes (from survey free responses)", voice_sentences(row)),
        ]
        return [(title, lines) for title, lines in sections if lines]

    sections = [("Demographics", demos_sentences(row))]
    if tier in {"employment", "geo", "transit"}:
        sections.append(("Employment", employment_sentences(row)))
    if tier in {"geo", "transit"}:
        sections.append(("Geographic location", geo_sentences(row)))
    if tier == "transit":
        sections.append(("Transportation use", transit_sentences(row)))
    return [(title, lines) for title, lines in sections if lines]


def build_user_prompt(row: pd.Series, tier: str) -> str:
    """Build a natural-language digital-twin user prompt for one tier.

    The body conveys the persona in second person (AI Terrarium narrative
    framing). A short self-report ask follows — not a long instruction block.
    """
    paragraphs = build_narrative_sections(row, tier)
    persona = "\n\n".join(paragraphs) if paragraphs else "You are a survey respondent."

    # Calibrated ask: independent subscales + mid-scale anchor (reduces bleed from
    # mobility cues into interpersonal over-prediction). No demographic
    # anti-stereotype rails — stereotyping RQs must remain valid.
    ask = (
        "Rate your communication apprehension in two contexts independently:\n"
        "(1) group discussions, and (2) one-on-one conversations with new people.\n\n"
        "For each context, report an integer from 6 (very low) to 30 (very high) "
        "and its band (low / moderate / high). Mid-scale scores are common; no "
        "single life circumstance determines your apprehension.\n\n"
        "Return ONLY the JSON object specified in the system instructions."
    )
    return f"{persona}\n\n{ask}"


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
    import warnings

    prompts: list[PersonaPrompt] = []
    n_skipped_missing_id = 0
    n_skipped_no_demos = 0
    for _, row in df.iterrows():
        pid = row.get("participant_id")
        if not _present(pid):
            n_skipped_missing_id += 1
            continue
        if require_demographics and not demos_sentences(row):
            n_skipped_no_demos += 1
            continue
        for tier in tiers:
            prompts.append(build_persona_prompt(row, tier))
    n_skipped = n_skipped_missing_id + n_skipped_no_demos
    if n_skipped:
        warnings.warn(
            f"Skipped {n_skipped} participant rows while building personas "
            f"({n_skipped_missing_id} missing participant_id, "
            f"{n_skipped_no_demos} empty demographics).",
            stacklevel=2,
        )
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

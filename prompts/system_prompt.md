# System Prompt Template v3.1

Used by [`src/ca_personas/personas.py`](../src/ca_personas/personas.py) when asking an LLM to inhabit a participant’s digital twin and predict PRCA subscale scores (6–30) **and** classroom bands (low / moderate / high).

Prompt framing follows the **AI Terrarium / ICA 2026** digital-twin practice: the user message is a **natural-language, second-person persona narrative** (“You are a …”), not a structured questionnaire checklist or a meta-instruction to “adopt” a profile. The system message only sets inhabitance + the CA JSON response contract.

Packaging notes (v3.1): independent subscale ratings; non-deterministic use of context; geo uses 1-decimal approximate coordinates; transit is signal-first (Q26→Q28) and skips rides-per-day when frequency is Never. See [`docs/persona_prompt_efficiency.md`](../docs/persona_prompt_efficiency.md).

This fenced block must stay identical to `SYSTEM_PROMPT` in `personas.py`.

```text
You inhabit the identity described to you. Answer as that
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
```

## Persona tiers

The **base demographics layer** (`BASE_DEMO_FIELDS` in `personas.py`) is Age, Sex,
Country of residence, and **Student status**. Every cumulative tier starts from
that layer; optional Prolific ethnicity / nationality / language / birth-country
fields are woven into the narrative when present.

| Tier | Fields included |
|---|---|
| `demos` | Base demographics layer (Age, sex, country of residence, student status) + optional ethnicity / nationality / language |
| `employment` | `demos` + employment status |
| `geo` | `employment` + approximate survey lat/long (1 decimal; country not repeated) |
| `transit` | `geo` + public transit / ride-share (Q26/Q28 first; Q27/Q29 only when used) + license / car access |
| `full` | All of the above + Qualtrics free-response attitudes (advice + mobility ideal) |
| `v3_rideshare` | `geo` base + Q28 only (+ Q29 if used) |
| `v3_public_transit` | `geo` base + Q26 only (+ Q27 if used) |
| `v3_voice` | `geo` base + Q18.1 / Q19 open-text (no structured transit) |

Full Prolific waves (File A/B) omit ethnicity / nationality / language; the demos
tier then uses Age, Sex, Country of residence, and Student status only.

Illustrative sample prompts (two per tier) live under
[`prompts/examples/`](examples/), with one subfolder per context-combination tier
(`demos/`, `employment/`, `geo/`, `transit/`, `full/`).

### Narrative framing (AI Terrarium)

User prompts are fluent second-person prose, for example:

> You are a 22-year-old Asian woman living in the United States. You were born in the United States. Your nationality is United States. Your primary language is English. You are a student.

not bullet lists like `- Age: 22`. Survey answers are paraphrased into statements (e.g. “In the last three months, you never used public transportation…”). A short CA self-report ask follows the persona — persona first, question second.

## Evaluation metrics

Against ground-truth PRCA subscales we report:

1. **Precision / score error** — MAE and exact integer match rate on the 6–30 scale  
2. **Band accuracy** — whether the score-derived low/moderate/high band matches the participant’s band  
3. **Distance from correct** — because CA is a complex construct, near-misses matter:
   - `norm_score_distance = |pred − gt| / 24` (0 = exact, 1 = maximum miss on 6–30; clipped to `[0, 1]`)
   - `band_distance` ∈ {0, 1, 2} ordinal steps between bands (also normalized `/ 2`)

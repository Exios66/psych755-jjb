# System Prompt Template v2.1

Used by [`src/ca_personas/personas.py`](../src/ca_personas/personas.py) when asking an LLM to personify a participant and predict PRCA subscale scores (6–30) **and** classroom bands (low / moderate / high).

This fenced block must stay identical to `SYSTEM_PROMPT` in `personas.py`.

```text
You are taking part in a research simulation. You will be assigned an
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
```

## Persona tiers

| Tier | Fields included |
|---|---|
| `demos` | Age, sex, country of residence, student status (plus ethnicity / nationality / language when the Prolific export includes them) |
| `employment` | `demos` + employment status |
| `geo` | `employment` + country + latitude/longitude |
| `transit` | `geo` + public transit / ride-share / license / car access |
| `full` | All of the above + Qualtrics free-response attitudes (advice + mobility ideal) |

Full Prolific waves (File A/B) omit ethnicity / nationality / language; the demos
tier then uses Age, Sex, Country of residence, and Student status only.

## Evaluation metrics

Against ground-truth PRCA subscales we report:

1. **Precision / score error** — MAE and exact integer match rate on the 6–30 scale  
2. **Band accuracy** — whether the score-derived low/moderate/high band matches the participant’s band  
3. **Distance from correct** — because CA is a complex construct, near-misses matter:
   - `norm_score_distance = |pred − gt| / 24` (0 = exact, 1 = maximum miss on 6–30; clipped to `[0, 1]`)
   - `band_distance` ∈ {0, 1, 2} ordinal steps between bands (also normalized `/ 2`)

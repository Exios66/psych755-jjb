# Example persona prompts by context-combination tier

Two illustrative samples per progressive / cumulative tier, built with
`ca_personas.personas.build_persona_prompt` in the **AI Terrarium** digital-twin
style: natural-language, second-person persona narratives (“You are a …”), not
structured questionnaire checklists.

See `prompts/system_prompt.md` and the ICA 2026 extended abstract on prompt
framing vs contextual depth.

| Folder | Tier | Cumulative fields |
|---|---|---|
| [`demos/`](demos/) | `demos` | Base demographics (Age, Sex, Country, Student status) + optional ethnicity / nationality / language |
| [`employment/`](employment/) | `employment` | `demos` + employment status |
| [`geo/`](geo/) | `geo` | `employment` + approximate survey lat/long (1 decimal; country not repeated) |
| [`transit/`](transit/) | `transit` | `geo` + Q26/Q28 first; Q27/Q29 only when used; license / car access |
| [`full/`](full/) | `full` | All of the above + Qualtrics free-response attitudes (Q18 advice + Q19 mobility ideal) |
| [`v3_rideshare/`](v3_rideshare/) | `v3_rideshare` | `geo` base + Q28 only (+ Q29 if used) |
| [`v3_public_transit/`](v3_public_transit/) | `v3_public_transit` | `geo` base + Q26 only (+ Q27 if used) |
| [`v3_voice/`](v3_voice/) | `v3_voice` | `geo` base + open-text attitudes only |

See [`docs/persona_prompt_efficiency.md`](../../docs/persona_prompt_efficiency.md) for the v3.1 packaging rationale and v3 ablation design.

Each folder contains `sample_01.md` and `sample_02.md` (contrasting student /
regular-transit vs full-time / car-oriented profiles). Participant ids are
illustrative labels for documentation — not live Prolific IDs.

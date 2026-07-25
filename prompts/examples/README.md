# Example persona prompts by context-combination tier

Two illustrative samples per progressive / cumulative tier, built with
`ca_personas.personas.build_persona_prompt` so the system + user text matches
the research pipeline (`prompts/system_prompt.md`, `src/ca_personas/personas.py`).

| Folder | Tier | Cumulative fields |
|---|---|---|
| [`demos/`](demos/) | `demos` | Base demographics (Age, Sex, Country, Student status) + optional ethnicity / nationality / language |
| [`employment/`](employment/) | `employment` | `demos` + employment status |
| [`geo/`](geo/) | `geo` | `employment` + country + latitude/longitude |
| [`transit/`](transit/) | `transit` | `geo` + transit / ride-share / license / car access |
| [`full/`](full/) | `full` | All of the above + Qualtrics free-response attitudes (Q18 advice + Q19 mobility ideal) |

Each folder contains `sample_01.md` and `sample_02.md` (contrasting student /
regular-transit vs full-time / car-oriented profiles). Participant ids are
illustrative labels for documentation — not live Prolific IDs.

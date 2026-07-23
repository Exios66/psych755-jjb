# psych755-jjb - A Summer Semester Final Project

**Group:** Jack J. Burleson 
**Course:** PSYCH 755, Summer Semester 2026 - UW-Madison | Madison, WI
**Professor:** Dr. Adam Ross Nelson

This is a final semester project for the course PSYCH 755 with Dr. Adam Ross Nelson; It is a showcase of the effective usage of mainstream data science software & utilities.

## Members

| Name | GitHub Username |
|---|---|
| Jack J. Burleson | @Exios66 |
| Jack J. Burleson | @jjburleson |

## Research Question

When an LLM is given a dynamically-constructed, first-person “embodiment” prompt built from an individual respondent’s demographic and behavioral attributes, and instructed to predict that person’s own communication-apprehension (CA) score, how accurately does the model recover the respondent’s true PRCA subscale scores — and where accuracy fails, does the error pattern correlate systematically with demographic group membership (i.e., stereotyping) rather than random noise?

## Research Focus

1. Does employment status improve prediction accuracy over demographics alone, and does it change the bias pattern (e.g., does the model now stereotype “unemployed” respondents as higher-CA, correctly or not)?
2. Does transportation-use data improve prediction accuracy, and does the model use it sensibly (e.g., inferring low transit use → higher avoidance → higher CA) or does it ignore it/misuse it?
3. Does combining both help beyond either alone, or do they carry redundant signal (e.g., employment and transit use may correlate with each other in your sample, so Tier 3 may not beat Tier 1 or 2 individually)?

## Suggested Project Structure + Contents

| Path | What it is |
|---|---|
| `index.qmd` | The primary manuscript. Start here. |
| `contributions.md` | Who owned what. |
| `memos/` | Individual research memos, one per member. |
| `references.bib` | Shared BibTeX file for the manuscript and memos. |
| `data/` | Data used in this project. |

## Reproducing this project

```bash
# from the root of the repo
quarto render index.qmd
```

## Notes

[Anything a reader needs to know that does not belong in the manuscript. Delete this section if you have nothing to say here.]

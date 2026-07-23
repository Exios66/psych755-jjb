# psych755-jjb - A Summer Semester Final Project

**Group:** Jack J. Burleson 
**Course:** PSYCH 755, Summer Semester 2026 - UW-Madison | Madison, WI
**Professor:** Dr. Adam Ross Nelson

<a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://wisc.edu/"><img src="https://img.shields.io/badge/UW--Madison-Research-C5050C?style=for-the-badge&logo=google-scholar&logoColor=white" alt="UW-Madison"></a>

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

## Persona / LLM prediction framework

Python package under [`src/ca_personas/`](src/ca_personas/) extracts Prolific + Qualtrics fields, scores ground-truth PRCA subscales (6–30), builds **tiered** persona prompts (`demos` → `employment` → `geo` → `transit`), calls **Ollama** or **OpenRouter**, and writes prediction error tables.

See [`docs/framework.md`](docs/framework.md) for architecture details.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set Ollama or OpenRouter credentials

# Offline dry run (deterministic mock model)
ca-personas --provider mock --join inner

# Local Ollama
ca-personas --provider ollama --model llama3.2

# OpenRouter
ca-personas --provider openrouter --model meta-llama/llama-3.2-3b-instruct:free

pytest
```

Artifacts land in `outputs/personas/`, `outputs/predictions/`, and `outputs/evaluation/`.

## Reproducing this project

```bash
# from the root of the repo
quarto render index.qmd
```

## Notes

Excerpt fixtures live in `data/excerpts/`. Generated `data/processed/` and `outputs/` are gitignored. Never commit API keys; use `.env` locally.
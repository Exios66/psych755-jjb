# Project data

| Path | Purpose |
|---|---|
| `excerpts/` | Small Prolific + Qualtrics excerpt files for development, tests, and Posit Connect Cloud renders |
| `processed/` | Generated joined/scored participant tables + cleaning reports (**gitignored**) |

## Full cohort exports (private — do not commit)

Place the three source exports in a **sibling** folder next to this repository:

```text
../sibling_data/PRCAProlificExport_FileA.csv   # Prolific wave 1
../sibling_data/PRCAProlificExport_FileB.csv   # Prolific wave 2 (same columns; stack with A)
../sibling_data/PRCAQualtricsExport_FileC.csv  # Qualtrics responses; merge on Q0
```

From Python / notebooks:

```python
import pandas as pd

a = pd.read_csv("../sibling_data/PRCAProlificExport_FileA.csv")
b = pd.read_csv("../sibling_data/PRCAProlificExport_FileB.csv")
c = pd.read_csv("../sibling_data/PRCAQualtricsExport_FileC.csv")
```

Or via the package (preferred):

```bash
ca-personas prepare --join inner
# or the full offline pipeline
ca-personas run --provider mock --join inner
```

**Never commit** File A/B/C (or any other full Prolific/Qualtrics dump) into git. The repo `.gitignore` already blocks `data/*` except the approved excerpt fixtures.

### Merge key

Prolific `Participant id` ↔ Qualtrics `Q0` (File C) or `PROLIFIC_PID` when present in richer exports.

### Expected merge coverage (File A + File B vs File C)

| Bucket | N | Use in analysis? |
|---|---:|---|
| Matched Prolific ∩ Qualtrics | **252** | Yes (then apply complete-CA filters) |
| Qualtrics-only (incl. blank `Q0` test rows) | **21** | No — disregard |
| Prolific-only | **10** | No — disregard |

Sanity check: `252 + 10 = 262` Prolific IDs; `252 + 21 = 273` Qualtrics rows.  
See `ca_personas.load.merge_coverage_audit` and `docs/qualtrics_data_dictionary.csv`.

### Schema notes (full cohort vs excerpts)

| Source | Notes |
|---|---|
| File A + File B | Demographics: Age, Sex, Country of residence, Student status, Employment status. **No** ethnicity / nationality / language columns. |
| File C | Flat single-header Qualtrics CSV (not the 3-row header block). Open advice is `Q18.1` (mapped to `Q18_advice`). PRCA items `Q1–Q6`, `Q13–Q18`; transit `Q26–Q29`, `Q20`, `Q21`. |

### Transportation instrument (verified against excerpt + File C)

| Item | Survey stem | Closed choices |
|---|---|---|
| `Q26` | In the last three months on how many days did you use public transportation (bus, train, tram, etc.)? | `Never`, `0-1 days a month`, `2-4 days a month`, `4-8 days a month`, `8 or more days a month` |
| `Q27` | On a typical day of public transportation use, how many rides do you take? | `1-2` / `3-4` / `5-6` / `7 or more` rides in a typical day |
| `Q28` | In the last three months on how many days did you use ride share platforms (Lyft, Uber, DiDi, etc.)? | Same day-frequency scale as `Q26` |
| `Q29` | On a typical day of ride share use, how many rides do you take? | Same rides-per-day scale as `Q27` |
| `Q20` | Do you have a license to drive a car? | `Yes`, `No` |
| `Q21` | Do you have access to a car you can use for transportation? | `Yes`, `No`, (`Not Sure` in full File C) |

Note: `Q26`/`Q28` stems ask about days in the **last three months**, but the answer labels are worded as **days a month**. Secondary-RQ “regular” riders use the choice text as written (`4-8` or `8+` days a month).

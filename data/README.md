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

### Schema notes (full cohort vs excerpts)

| Source | Notes |
|---|---|
| File A + File B | Demographics: Age, Sex, Country of residence, Student status, Employment status. **No** ethnicity / nationality / language columns. |
| File C | Flat single-header Qualtrics CSV (not the 3-row header block). Open advice is `Q18.1` (mapped to `Q18_advice`). PRCA items `Q1–Q6`, `Q13–Q18`; transit `Q26–Q29`, `Q20`, `Q21`. |

---
title: "Merge coverage sanity check"
subtitle: "File A/B ↔ File C matching totals"
---

# Merge coverage sanity check (File A/B ↔ File C)

## Expected totals (data owners)

| Bucket | N | Action |
|---|---:|---|
| Matched observations (Prolific **and** Qualtrics) | **252** | Analysis sample (before complete-CA filters) |
| Qualtrics-only | **21** | Disregard (test responses / unmatched) |
| Prolific-only | **10** | Disregard |

Identities:

- Prolific File A + File B = \(252 + 10 = 262\) unique participant IDs  
- Qualtrics File C = \(252 + 21 = 273\) rows  

Of the 21 Qualtrics-only rows, **18** have blank `Q0` and **3** have a non-empty `Q0` that does not appear in Prolific.

## Bug found & fixed

File C is a **flat** single-header export. An earlier detector treated long `ResponseId` / hash strings in the first data row as Qualtrics “question label” rows and skipped the first two responses. That dropped one valid match (251 instead of 252) and under-counted Qualtrics rows (271 instead of 273).

**Fix:** require ImportId markers in row 2 before treating an export as the 3-row Qualtrics header block (`src/ca_personas/load.py`). The excerpt fixture still uses the 3-row format and continues to load correctly.

## Data dictionary alignment

[`docs/qualtrics_data_dictionary.csv`](qualtrics_data_dictionary.csv) lists field names and survey stems. File C includes the instrument columns (`Q0`–`Q6`, `Q13`–`Q18`, `Q26`–`Q29`, `Q20`, `Q21`, `Q18.1`, `Q19`, lat/long, `ResponseId`, …). Optional Qualtrics meta columns (`Status`, `IPAddress`, `PROLIFIC_PID`, …) appear in the dictionary / excerpt but are absent from the flat File C export — the loader already treats them as optional.

Transit stems in the dictionary match the scales used by `transit_ca` / persona prompts:

- **Q26** public transportation days  
- **Q27** rides on a typical public-transit day  
- **Q28** ride-share days  
- **Q29** rides on a typical ride-share day  
- **Q20** / **Q21** license / car access  

## How to re-verify

```bash
ca-personas prepare --join inner
python - <<'PY'
from ca_personas.load import load_full_cohort
_, report = load_full_cohort(join_how="inner")
assert report["n_matched_both"] == 252
assert report["n_qualtrics_only"] == 21
assert report["n_prolific_only"] == 10
print("merge coverage OK", report["n_matched_both"], report["n_analytic"])
PY
pytest tests/test_merge_coverage.py
```

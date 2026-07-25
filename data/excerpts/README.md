# Excerpt fixtures

Public fixtures used by tests, Quarto / Posit Connect Cloud renders, and as a
**fallback** when private full-cohort exports are absent.

| File | Role |
|---|---|
| `prolific_excerpt.csv` | Prolific demographics export (richer column set than File A/B) |
| `qualtrics_excerpt.csv` | Qualtrics survey export with the standard 3-row header block |

## Path resolution

`config/default.yaml` lists preferred sibling-data paths:

```text
../sibling_data/PRCAProlificExport_FileA.csv
../sibling_data/PRCAProlificExport_FileB.csv
../sibling_data/PRCAQualtricsExport_FileC.csv
```

`ca_personas.pipeline` and `ca_personas.paths` use those paths **only when the
files exist**. Otherwise they fall back to the excerpt CSVs in this directory.
CLI flags `--prolific` / `--qualtrics` always override both.

Join key: Prolific `Participant id` ↔ Qualtrics `PROLIFIC_PID` / `Q0`.

# Excerpt fixtures

Public fixtures used by **unit tests and offline CI only**. They are **not**
displayed on the Posit Connect Cloud site.

| File | Role |
|---|---|
| `prolific_excerpt.csv` | Prolific demographics export (richer column set than File A/B) |
| `qualtrics_excerpt.csv` | Qualtrics survey export with the standard 3-row header block |

## Path resolution

`ca_personas.paths` prefers full-cohort File A/B/C when staged:

```text
../sibling_data/PRCAProlificExport_FileA.csv
../sibling_data/PRCAProlificExport_FileB.csv
../sibling_data/PRCAQualtricsExport_FileC.csv
```

Cloud agents may also stage the same filenames under `/tmp/sibling_data` or
`$CA_SIBLING_DATA`. Posit Connect renders use those full-cohort exports when
present, otherwise the committed full-cohort mock tables in
`artifacts/posit_full_cohort/` (N = 241). Excerpt fixtures remain a last-resort
fallback for tests when no full cohort is available.

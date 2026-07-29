---
name: posit-connect-publish
description: >-
  Update Quarto (.qmd/.md) manuscript content, re-run full-cohort analyses on
  sibling File A/B/C data, render the site, publish to the JackJBurleson Posit
  Connect Cloud deployment, and verify the live site. Use when the user asks to
  publish/render/deploy the Quarto site, update Posit Connect Cloud, push the
  manuscript online, or refresh JackJBurleson / jackjburleson Connect content.
---

# Posit Connect Cloud publish (JackJBurleson)

End-to-end workflow for this repo (`psych755-jjb`): **full-data analyses → update write-ups → `quarto render` → publish → verify**.

## Canonical deployment

| Field | Value |
| --- | --- |
| Account | `jackjburleson` |
| Content ID | `019f9a10-ebb9-d1d5-839f-97e794bfd0ca` |
| Dashboard | <https://connect.posit.cloud/jackjburleson/content/019f9a10-ebb9-d1d5-839f-97e794bfd0ca> |
| Public share URL | <https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/> |
| Config | `_publish.yml`, `_quarto.yml` |

Do **not** publish to the older `jjb-morningstar` content id unless the user explicitly asks and auth has write access to that account.

## Hard rule: full data before publish

Publishing **requires** the private full cohort (File A + File B + File C). Excerpt-only publishes are forbidden unless the user explicitly overrides.

Expected paths (prefer in order):

1. `../sibling_data/PRCA{ProlificExport_FileA,ProlificExport_FileB,QualtricsExport_FileC}.csv`
2. `/tmp/sibling_data/` with the same three filenames (cloud-agent staging)
3. Explicit `--prolific` / `--qualtrics` CLI paths

Gate with `ca_personas.paths.sibling_data_available()` or the equivalent three-file check. If files are only under `/tmp/sibling_data`, symlink or copy them to `../sibling_data/` **or** pass explicit paths into every CLI invocation.

**Full-cohort sanity targets (seed=42, inner join)** — fail publish verification if these regress without explanation:

- Matched analytic PRCA sample **n ≈ 241** (101 regular / 140 not)
- Geo → transit RF AUC ≈ **0.551**
- CA → transit RF AUC ≈ **0.590**
- Q28 → transit RF AUC ≈ **0.762**
- Q27 → transit RF AUC ≈ **0.589**

## Workflow (agent checklist)

### 1. Update content

Edit the relevant `index.qmd`, `docs/*.md`, and/or `memos/*.md`. Keep manuscript numbers tied to seeded full-cohort artifacts under `outputs/` and the research memos.

If `_quarto.yml` `project.render` list is missing a new page, add it (and navbar/sidebar entries when user-facing).

### 2. Re-run analyses on the **full** cohort

Install deps if needed: `pip install -e .` and Quarto ≥ 1.10 (`posit-connect-cloud` provider).

```bash
# Prefer sibling_data; otherwise pass /tmp/sibling_data paths.
export PROLIFIC_A=../sibling_data/PRCAProlificExport_FileA.csv
export PROLIFIC_B=../sibling_data/PRCAProlificExport_FileB.csv
export QUALTRICS=../sibling_data/PRCAQualtricsExport_FileC.csv

ca-personas prepare --join inner \
  --prolific "$PROLIFIC_A" "$PROLIFIC_B" --qualtrics "$QUALTRICS"

ca-personas transit-ca --join inner --seed 42 \
  --prolific "$PROLIFIC_A" "$PROLIFIC_B" --qualtrics "$QUALTRICS"

ca-personas geo-transit-rf --join inner --seed 42 \
  --prolific "$PROLIFIC_A" "$PROLIFIC_B" --qualtrics "$QUALTRICS"

ca-personas ca-transit-rf --join inner --seed 42 \
  --prolific "$PROLIFIC_A" "$PROLIFIC_B" --qualtrics "$QUALTRICS"

ca-personas covariate-transit-rf --join inner --seed 42 \
  --specs q27_intensity q28_days q27_q28 rideshare car_access employment mobility_bundle \
  --figures-dir memos/figures \
  --prolific "$PROLIFIC_A" "$PROLIFIC_B" --qualtrics "$QUALTRICS"
```

Sync any changed AUCs / N into memos and `index.qmd` before rendering. Commit content updates on a `cursor/<name>-ecf3` branch when this is a cloud agent.

### 3. Render

```bash
quarto check
quarto render    # writes _site/
```

Note: `index.qmd` live Python cells use the **full matched cohort + mock LLM**
for Posit-safe MAE figures (or committed `artifacts/posit_full_cohort/` when
File A/B/C are absent). Secondary RQ numbers in prose/memos must still reflect
the seeded full-cohort RF runs above.

### 4. Authenticate to Posit Connect Cloud

Prefer env vars when available:

- `POSIT_CONNECT_CLOUD_ACCESS_TOKEN`
- `POSIT_CONNECT_CLOUD_REFRESH_TOKEN`
- `POSIT_CONNECT_CLOUD_ACCOUNT_ID` (JackJBurleson account id)

Otherwise run **device-code OAuth**:

1. `POST https://login.posit.cloud/oauth/device/authorize` with `client_id=quarto-cli&scope=vivid`
2. Show the user `verification_uri_complete` and `user_code`
3. Poll `https://login.posit.cloud/oauth/token` until authorized
4. Confirm the authorized account name is **`jackjburleson`** (or has write on the content id). If the wrong account authorizes, stop and re-auth — do not silently publish elsewhere.

### 5. Publish

Preferred one-shot helper:

```bash
python scripts/publish_posit_jackjburleson.py
# or skip re-analysis if just done:
python scripts/publish_posit_jackjburleson.py --skip-analysis
```

The script updates content `019f9a10-ebb9-d1d5-839f-97e794bfd0ca` via Connect Cloud API (new bundle → upload `_site` tar.gz → publish) and polls until `publish_result=success`.

Fallback:

```bash
export POSIT_CONNECT_CLOUD_ACCESS_TOKEN=...
export POSIT_CONNECT_CLOUD_REFRESH_TOKEN=...
export POSIT_CONNECT_CLOUD_ACCOUNT_ID=...
quarto publish posit-connect-cloud --no-prompt --no-browser --id 019f9a10-ebb9-d1d5-839f-97e794bfd0ca
```

### 6. Verify (required)

After publish, fetch the **share URL** (not only the dashboard SPA):

`https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/`

Assert:

- HTTP 200 and title contains the manuscript / site title
- Body includes the updated section(s) (e.g. `Q27`, `0.762`, memo links as applicable)
- Full-cohort markers present (`n = 241` or current analytic N; key AUCs)
- Screenshot to `/opt/cursor/artifacts/connect-cloud-published.png` when in cloud agents

Report the share URL + dashboard URL to the user.

## Common failure modes

| Symptom | Fix |
| --- | --- |
| Only excerpt fixtures found | Stage File A/B/C; refuse publish |
| Device auth → wrong account / 403 on content | Re-auth as JackJBurleson owner |
| Quarto lacks `posit-connect-cloud` | Install Quarto ≥ 1.10 |
| Silent `quarto publish` exit (no TTY) | Use device-code + API helper script |
| Live page is Connect login SPA | Verify via `.share.connect.posit.cloud` URL |

## Related paths

- Manuscript: `index.qmd`
- Site config: `_quarto.yml`
- Publish target: `_publish.yml`
- Helper: `scripts/publish_posit_jackjburleson.py`
- Full-data paths: `src/ca_personas/paths.py`
- Secondary RQ CLI: `ca-personas {transit-ca,geo-transit-rf,ca-transit-rf,covariate-transit-rf}`

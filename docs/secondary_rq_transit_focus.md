---
title: "Secondary focus: predict transit with mobility held out"
subtitle: "TF1 regular-rider classification and TF2 intensity estimation from profile (+ CA)"
---

**Code:** [`src/ca_personas/transit_focus.py`](../src/ca_personas/transit_focus.py)  
**CLI:** `ca-personas transit-focus --join inner --seed 42`  
**Memo:** [`memos/transit_focus_regular_and_intensity.qmd`](../memos/transit_focus_regular_and_intensity.qmd)  
**Artifacts:** `outputs/transit_focus/`  
**Persona prompts:** `outputs/transit_focus/persona_prompts/`

---

## Research role

Primary RQ1–RQ3 ask whether **transit cues improve CA prediction**. Secondary S1–S5 ask whether **CA / place / mobility predict transit** when those fields are available as features.

This **secondary focus** flips the persona task again:

1. **TF1.** Can a model estimate **regular (weekly+) public-transit use** from demographics, employment, and geography — with **all transit / ride-share / car self-reports held out** of the prompt and feature set?
2. **TF2.** Can the same held-out profile, plus **PRCA CA scores**, estimate **how much** public transit the person uses (Q26 day-frequency; Q27 rides/day among reporters)?

These questions support an LLM “digital twin” that infers mobility habits from non-mobility context, without leaking Q26–Q29 / Q20 / Q21 / Q19 into the prompt.

## Hold-out rule

Never used as predictors or persona text:

| Held out | Items |
|---|---|
| Public transit | `Q26`, `Q27` |
| Ride-share | `Q28`, `Q29` |
| Car access | `Q20`, `Q21` |
| Mobility free-text | `Q19` |

Allowed profile layers: demos (Age, Sex, Country, Student), employment, survey lat/long. TF1 also has a **profile + CA** arm; TF2 always includes group + interpersonal CA.

## Specs

| Spec key | Task | Features | Outcome |
|---|---|---|---|
| `tf1_profile_regular` | Binary | demos + employment + geo | `regular_transit` (Q26 weekly+) |
| `tf1_profile_ca_regular` | Binary | + `gt_group_ca`, `gt_interpersonal_ca` | `regular_transit` |
| `tf2_intensity_q26` | Ordinal / multiclass | profile + CA | Q26 day-frequency |
| `tf2_intensity_q27` | Ordinal / multiclass | profile + CA | Q27 rides/day |

**Models.** Balanced Random Forests with mixed numeric/categorical preprocessing; stratified 5-fold CV (`random_state=42`).

**LLM twin.** Separate system prompt + tiers `tf_demos` → `tf_employment` → `tf_geo` → `tf_geo_ca` emit JSON `{regular_transit, q26_days, confidence}` — not wired into the primary CA `run` pipeline.

## Seeded full-cohort results (N complete-case = 224; seed = 42)

From committed analytic participants (`artifacts/posit_full_cohort/participants.csv`):

| Spec | n | Primary metric | Takeaway |
|---|---:|---:|---|
| Profile → regular | 224 | ROC-AUC **0.662** | Beats chance; Age + longitude lead permutation importance |
| Profile + CA → regular | 224 | ROC-AUC **0.672** | CA adds only **+0.010** AUC over profile alone |
| Profile + CA → Q26 | 224 | Bal. acc. **0.283** (ordinal MAE **1.22**) | Beats majority accuracy (0.228) but weak absolute recovery |
| Profile + CA → Q27 | 222 | Bal. acc. **0.371** (ordinal MAE **0.38**) | Does **not** beat majority accuracy (0.805; class imbalance) |

Compare: Wave-2 Age+Sex+Student → regular transit AUC = **0.618** (n=224). Adding employment + lat/long lifts the tabular ceiling to **0.662** without any mobility self-report.

## Reproducibility

```bash
# Prefer staged File A/B/C; otherwise the module can be driven from scored participants.
ca-personas transit-focus --join inner --seed 42 \
  --output-dir outputs/transit_focus

# Subset
ca-personas transit-focus --specs tf1_profile_regular tf2_intensity_q26
```

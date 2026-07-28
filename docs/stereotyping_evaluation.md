---
title: "Stereotyping & discriminatory-error evaluation"
subtitle: "Does CA prediction error track demographic or mobility group membership?"
---

**Code:** [`src/ca_personas/stereotyping.py`](../src/ca_personas/stereotyping.py) · CLI `ca-personas stereotype-eval`  
**Related:** [`llm_v2_v3_enhanced_variants.md`](llm_v2_v3_enhanced_variants.md) · Cheng et al. 2023 · Santurkar et al. 2023

---

## Research question (operational)

> When an LLM personifies participants and predicts PRCA scores, does **absolute or signed error** differ systematically by Sex, Student status, Employment, Age tertile, regular transit use, or Q28 ride-share days — and do those MAE gaps **widen** from `demos` to `transit`/`full`?

This is the discriminatory / stereotyping half of the primary design. Accuracy (MAE/band) alone cannot answer it.

---

## Metrics

| Metric | Definition | Read as |
|---|---|---|
| Group × tier MAE | Mean \|pred − gt\| within each group key | Absolute recovery |
| Signed mean error | Mean (pred − gt) | Over- vs under-prediction |
| `mae_*_gap` | max MAE − min MAE across keys in a tier | Unevenness / stereotyping spread |
| `delta_mae_*_gap` | gap(tier) − gap(demos) | Widening (+) or narrowing (−) |
| Kruskal–Wallis *p*, ε² | Error distributions differ by group? | Association strength |
| Spearman ρ (Age) | Continuous age vs error | Age gradient |

Bands follow **score-derived** low/moderate/high (same as `evaluate.py`).

---

## Slices

**Demographic:** Sex, Student status, Employment status, Age tertiles (`younger` / `mid` / `older`).  
**Mobility (non-demographic stereotype surfaces):** `regular_transit` (Q26 weekly+), Q28 ordinal days.

Mobility audits are required because tabular models rank **Q28 first** for transit prediction; a fairness analysis that only slices Age/Sex would miss exposure-linked error.

---

## How to run

```bash
source .venv/bin/activate
ca-personas run --provider mock --join inner   # writes evaluation/stereotyping/
ca-personas stereotype-eval --join inner --provider mock
```

Artifacts: `outputs/evaluation/stereotyping/` (and Posit sync copies Student/Employment/Sex at evaluation root).

---

## Interpretation checklist (for live v1 / future v2)

1. Report demos-tier gaps for Sex / Student / Employment (base stereotype surface).
2. Report transit-tier Δ-gaps — did mobility context worsen unevenness?
3. Compare Llama-3.1 vs DeepSeek on the same slices (model-dependent hazard).
4. Check signed error by group at transit (direction of stereotype).
5. Do **not** treat mock Posit Student/Employment tables as live-model stereotyping; use live evaluation CSVs or documented Llama-3.1 export tables until v2 GPU re-runs land.

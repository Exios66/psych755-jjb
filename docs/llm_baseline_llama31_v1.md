---
title: "Llama-3.1 full-cohort baseline (prompt v1)"
subtitle: "vLLM digital-twin results on the original 5-tier Terrarium prompts"
---

**Model:** `meta-llama/Llama-3.1-8B-Instruct` · **Prompt generation:** pre–v3.1 packaging (version 1 core ladder)  
**Companion:** [Cross-model memo](../memos/vllm_v1_cross_model_comparison.md) · [Llama-3.2-3B](llm_baseline_llama32_instruct_v1.md) · [DeepSeek](llm_baseline_deepseek_r1_distill_v1.md) · [Llama-3.3-70B](llm_baseline_llama33_70b_v1.md) · [Prompt version map](persona_prompt_versions.md) · [v2/v3 packaging](persona_prompt_efficiency.md)

> This page is the published baseline for **prompt version 1** (five cumulative tiers: `demos` → `full`). Later packaging (v2 / v3.1) and ablation tiers (v3) are **designed** against these failure modes and implemented in code; GPU re-evaluation is **deferred future work** ([`persona_prompt_versions.md`](persona_prompt_versions.md)).

---

**Model:** `meta-llama/Llama-3.1-8B-Instruct`  
**Quantization / hardware:** fp8 (Marlin weight-only on NVIDIA RTX A5000); NVIDIA RTX A5000 (GPU 0)  
**Sample:** N = 241 matched Prolific↔Qualtrics participants (inner join), **5 persona tiers** → **1205** prompts  
**Parse success:** 1205/1205 (100.0%)  
**Throughput:** 12.28 samples/s (wall ~98.1s after engine warmup)  
**Export stamp:** `20260726_0221`  
**Host:** `rogers-gpu-1.discovery.wisc.edu`

---

## 1. Research questions (primary LLM persona track)

1. **RQ1 — Employment context:** Does adding employment status to demographic personas reduce absolute PRCA prediction error?
2. **RQ2 — Transit context:** Do transportation-use cues improve CA prediction (or change error structure)?
3. **RQ3 — Cumulative / full context:** Does stacking demographics → employment → geography → transit (and the `full` bundle) improve recovery of ground-truth PRCA and/or change demographic error patterns (stereotyping)?

**Tracked metrics** (project standard):
- **MAE** on group & interpersonal PRCA subscales (scores 6–30)
- **Exact score match** (predicted integer == ground truth)
- **Band accuracy** (low / moderate / high; bands resolved from predicted score)
- **Band distance** (ordinal 0–2) and normalized distances
- **Signed mean error** (bias: positive = over-prediction of CA)

Secondary observational RQs (transit Random Forests, etc.) are out of scope for this vLLM export.

---

## 2. Executive interpretation

### Overall success vs naive baselines
Across all 1,205 predictions (241 × 5 tiers):

| Metric | Observed | Naive baseline | Verdict |
|---|---:|---:|---|
| Exact score match (group) | 6.1% | ~4.0% (1/25 integers) | Slightly above chance |
| Exact score match (interpersonal) | 9.0% | ~4.0% | Above chance |
| Band accuracy (group) | 28.8% | ~33.3% (3 bands) | **Below** uniform chance |
| Band accuracy (interpersonal) | 40.2% | ~33.3% | Modestly above chance |
| MAE group | 5.92 pts | — | Large vs classical ML floors (~4.5 in manuscript) |
| MAE interpersonal | 5.82 pts | — | Large absolute error on a 24-point span |

**Bottom line:** Llama-3.1-8B-Instruct emits valid PRCA JSON for every prompt (100% parse rate) and recovers interpersonal **bands** better than chance in demos/employment/geo tiers, but it is **not** a high-fidelity digital twin of participants’ PRCA scores. Exact score recovery remains rare (~6–13%). Adding transit/full context often **hurts** interpersonal prediction while only marginally helping group MAE/band metrics.

### RQ verdicts

**RQ1 (employment):** Employment does **not** meaningfully improve CA prediction vs demos-only. Group MAE 6.05 → 6.03 (Δ -0.02); interpersonal MAE 4.67 → 4.73 (Δ 0.07). Band accuracies stay flat.

**RQ2 (transit):** Transit cues are mixed. Group MAE improves slightly (6.05 → 5.68, Δ -0.37), but interpersonal MAE worsens sharply (4.67 → 8.17, Δ 3.51). Interpersonal band accuracy falls 51.9% → 15.4%. Signed interpersonal error at transit is 6.52 (systematic over-prediction).

**RQ3 (full / cumulative):** Full context is not uniformly better. Group MAE 5.84 vs demos 6.05; group band accuracy rises 25.7% → 37.8%. Interpersonal MAE worsens (4.67 → 6.92) and interpersonal band accuracy falls (51.9% → 29.0%).

### Practical interpretation
The model behaves more like a **stereotype / prior engine** than an individual-difference instrument:
- With only demos/employment/geo, interpersonal bands are recovered at ~52–53% — better than chance — while exact scores remain rare.
- Injecting transit (and full) information appears to **shift the model toward higher interpersonal CA**, increasing absolute error and ordinal band distance.
- Group-CA band accuracy is highest in the `full` tier (37.8%), suggesting coarse categorical recovery can improve even when continuous MAE stays large.

Relative to the manuscript’s classical tabular floor (best transit group MAE ≈ 4.49 Ridge), Llama-3.1-8B-Instruct’s best group MAE here is **5.68**, still worse than that ML floor.

---

## 3. Metrics by tier

| tier | n_with_ground_truth | mae_group | mae_interpersonal | exact_acc_group | exact_acc_interpersonal | band_acc_group | band_acc_interpersonal | mean_band_distance_group | mean_band_distance_interpersonal | mean_error_group | mean_error_interpersonal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1205 | 5.924 | 5.825 | 6.1% | 9.0% | 28.8% | 40.2% | 0.766 | 0.851 | 1.915 | 0.472 |
| demos | 241 | 6.050 | 4.668 | 6.2% | 12.9% | 25.7% | 51.9% | 0.755 | 0.660 | 3.444 | -2.212 |
| employment | 241 | 6.033 | 4.734 | 5.8% | 12.4% | 25.3% | 51.9% | 0.751 | 0.672 | 3.394 | -2.162 |
| geo | 241 | 6.017 | 4.627 | 5.8% | 12.9% | 25.3% | 53.1% | 0.747 | 0.651 | 3.378 | -2.311 |
| transit | 241 | 5.676 | 8.174 | 5.8% | 2.1% | 29.9% | 15.4% | 0.801 | 1.257 | 0.697 | 6.523 |
| full | 241 | 5.842 | 6.921 | 6.6% | 5.0% | 37.8% | 29.0% | 0.776 | 1.012 | -1.336 | 2.523 |

### Deltas vs `demos` (negative MAE Δ = improvement)

| tier | delta_mae_group_vs_demos | delta_mae_interpersonal_vs_demos | delta_band_acc_group_vs_demos | delta_band_acc_interpersonal_vs_demos |
| --- | --- | --- | --- | --- |
| demos | 0.000 | 0.000 | 0.0% | 0.0% |
| employment | -0.017 | 0.066 | -0.4% | 0.0% |
| geo | -0.033 | -0.041 | -0.4% | 1.2% |
| transit | -0.373 | 3.506 | 4.1% | -36.5% |
| full | -0.207 | 2.253 | 12.0% | -22.8% |

---

## 4. Band confusion (all tiers pooled)

### Group CA
| gt_band   |   low |   moderate |   high |
|:----------|------:|-----------:|-------:|
| low       |    92 |        554 |      4 |
| moderate  |    51 |        252 |      2 |
| high      |    61 |        186 |      3 |

### Interpersonal CA
| gt_band   |   low |   moderate |   high |
|:----------|------:|-----------:|-------:|
| low       |   422 |         58 |    160 |
| moderate  |   230 |         29 |     86 |
| high      |   145 |         41 |     34 |

Rows = ground-truth band; columns = predicted band (resolved from predicted score).

---

## 5. Stereotyping / demographic error slices

Error summaries were computed by demographic field × tier for: Age, Sex, Student status, Employment status.

Highlights at the **demos** tier (group-CA MAE spread):

- **Age** (demos): group-CA MAE 0.00 (`75.0`, n=1) to 14.00 (`65.0`, n=1); spread 14.00.
- **Employment status** (demos): group-CA MAE 5.60 (`Other`, n=62) to 6.23 (`Part-Time`, n=31); spread 0.63.
- **Sex** (demos): group-CA MAE 5.62 (`Female`, n=120) to 6.47 (`Male`, n=121); spread 0.85.
- **Student status** (demos): group-CA MAE 5.12 (`Yes`, n=34) to 6.25 (`No`, n=190); spread 1.13.

Full slice tables are in `tables/stereotyping_by_*.csv`. These support RQ3’s bias question: if MAE gaps across Sex / Employment / Student status widen or shrink as tiers accumulate, that is evidence of context-sensitive stereotyping rather than uniform noise.

---

## 6. What “success” means for this task

| Criterion | Result |
|---|---|
| Engineering success (schema-valid CA JSON) | **Yes** — 100% parseable generations |
| Digital-twin success (exact PRCA recovery) | **No** — single-digit exact-match rates |
| Coarse clinical/band success | **Partial** — interpersonal bands OK without transit; group bands best at `full` |
| RQ1 confirmed (employment helps) | **No** |
| RQ2 confirmed (transit helps) | **Mixed / mostly no** for interpersonal; weak yes for group MAE |
| RQ3 confirmed (more context helps & clarifies stereotyping) | **Mixed** — some group-band gains; interpersonal degradation; inspect stereotyping CSVs |

---

## 7. Methods notes (reproducibility)

- Prompts: cumulative persona tiers `demos → employment → geo → transit → full` via `ca-personas` / `inference.export_prompts`.
- Inference: vLLM 0.26 offline engine; system prompt = project PRCA JSON schema; `max_output_tokens=256`; batch size 16.
- FP8 on A5000 uses Marlin **weight-only** FP8 (GPU lacks native FP8 compute).
- Evaluation uses `ca_personas.evaluate.evaluate_predictions` (bands resolved from predicted scores).
- Private File A/B/C were used for generation but are **not** included in this export package.

---

## 8. Package contents

```
00_run_metadata.json
README_EXPORT.txt
REPORT_results_interpretation_and_model_success.md
REPORT_results_interpretation_and_model_success.html
tables/   (metrics, evaluation, confusion, stereotyping)
figures/  (MAE, band accuracy, exact match)
raw/      (prompts, generations, predictions)
```

---

*Generated automatically from the full-cohort Llama-3.1-8B-Instruct vLLM run on the PSYCH 755 analytic sample (N=241).*


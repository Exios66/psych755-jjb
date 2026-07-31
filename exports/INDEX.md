# vLLM export index

_Regenerated 2026-07-30 22:13:21_

Layout:

```
exports/
  INDEX.md              ← this file
  v1/                   ← prompt-v1 CA baselines (done)
  v2/                   ← enhanced 5-tier CA (done: DeepSeek / Llama-3.1 / Llama-3.2-Instruct)
  v3/                   ← 8-tier greedy ablations (done: 4 packages; DeepSeek v3 still pending)
  transit_focus/        ← TF1/TF2 transit-prediction LLM twins
  prior_v3_greedy/      ← archived greedy v3 (identical predictions to `v3/`; kept for the pre-stamp record)
  zips/
```

CA packages: `scripts/package_vllm_export.py` → `exports/<v1|v2|v3>/`.
Transit-focus packages: `scripts/package_transit_focus_export.py` → `exports/transit_focus/`.

## Enhanced CA queue matrix

| Status | Bucket | Tag | Model |
|---|---|---|---|
| ✅ done | `v2` | `llama31_8b_instruct_v2` | Llama-3.1-8B-Instruct |
| ⏳ pending | `v2` | `llama32_3b_v2` | Llama-3.2-3B (base) |
| ✅ done | `v2` | `llama32_3b_instruct_v2` | Llama-3.2-3B-Instruct |
| ✅ done | `v2` | `deepseek_r1_distill_llama8b_v2` | DeepSeek-R1-Distill-Llama-8B |
| ⏳ pending | `v2` | `llama33_70b_instruct_awq_v2` | Llama-3.3-70B-Instruct-AWQ |
| ✅ done | `v3` | `llama31_8b_instruct_v3` | Llama-3.1-8B-Instruct |
| ✅ done | `v3` | `llama32_3b_v3` | Llama-3.2-3B (base) |
| ✅ done | `v3` | `llama32_3b_instruct_v3` | Llama-3.2-3B-Instruct |
| ⏳ pending | `v3` | `deepseek_r1_distill_llama8b_v3` | DeepSeek-R1-Distill-Llama-8B |
| ✅ done | `v3` | `llama33_70b_instruct_awq_v3` | Llama-3.3-70B-Instruct-AWQ |

Note: `llama32_3b_v3` (base) parses **0%** of outputs (7.1% in v1) and is excluded from reported metrics; `llama32_3b_v2` remains pending for the same reason. `v3/` packages are greedy-decode ablations identical to `prior_v3_greedy/`.

## Transit-focus queue matrix (TF1/TF2)

Predict `regular_transit` + Q26 from demos/employment/geo/(+CA) with mobility held out.

| Status | Bucket | Tag | Model |
|---|---|---|---|
| ⏳ pending | `transit_focus` | `llama31_8b_instruct_transit_focus` | Llama-3.1-8B-Instruct |
| ⏳ pending | `transit_focus` | `llama32_3b_instruct_transit_focus` | Llama-3.2-3B-Instruct |
| ⏳ pending | `transit_focus` | `deepseek_r1_distill_llama8b_transit_focus` | DeepSeek-R1-Distill-Llama-8B |
| ⏳ pending | `transit_focus` | `llama33_70b_instruct_awq_transit_focus` | Llama-3.3-70B-Instruct-AWQ |

## `v1/` — Published prompt-v1 baselines (greedy / Jul-26).

| Package dir | Model | Tiers | Parse | Report |
|---|---|---:|---:|---|
| [`psych755_vllm_deepseek_r1_distill_llama8b_v1_full_cohort_20260726_0324`](v1/psych755_vllm_deepseek_r1_distill_llama8b_v1_full_cohort_20260726_0324/) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 5 | 99.8% | yes |
| [`psych755_vllm_llama31_8b_instruct_v1_full_cohort_20260726_0221`](v1/psych755_vllm_llama31_8b_instruct_v1_full_cohort_20260726_0221/) | `meta-llama/Llama-3.1-8B-Instruct` | 5 | 100.0% | yes |
| [`psych755_vllm_llama32_3b_base_v1_full_cohort_20260726_0246`](v1/psych755_vllm_llama32_3b_base_v1_full_cohort_20260726_0246/) | `meta-llama/Llama-3.2-3B` | 5 | 7.1% | yes |
| [`psych755_vllm_llama32_3b_instruct_v1_full_cohort_20260726_0252`](v1/psych755_vllm_llama32_3b_instruct_v1_full_cohort_20260726_0252/) | `meta-llama/Llama-3.2-3B-Instruct` | 5 | 100.0% | yes |
| [`psych755_vllm_llama33_70b_instruct_awq_v1_full_cohort_20260728_2209`](v1/psych755_vllm_llama33_70b_instruct_awq_v1_full_cohort_20260728_2209/) | `casperhansen/llama-3.3-70b-instruct-awq` | 5 | 100.0% | yes |

## `v2/` — Signal-first 5-tier packaging + `v2_enhanced` / `large_model` presets.

| Package dir | Model | Tiers | Parse | Report |
|---|---|---:|---:|---|
| [`psych755_vllm_deepseek_r1_distill_llama8b_v2_full_cohort_20260730_2213`](v2/psych755_vllm_deepseek_r1_distill_llama8b_v2_full_cohort_20260730_2213/) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 5 | 99.8% | yes |
| [`psych755_vllm_llama31_8b_instruct_v2_full_cohort_20260728_2214`](v2/psych755_vllm_llama31_8b_instruct_v2_full_cohort_20260728_2214/) | `meta-llama/Llama-3.1-8B-Instruct` | 5 | 99.9% | yes |
| [`psych755_vllm_llama32_3b_instruct_v2_full_cohort_20260728_2353`](v2/psych755_vllm_llama32_3b_instruct_v2_full_cohort_20260728_2353/) | `meta-llama/Llama-3.2-3B-Instruct` | 5 | 100.0% | yes |

## `v3/` — 8-tier greedy ablations (anti-bleed prompts; same decoding as the archived `prior_v3_greedy` runs).

> `v3_enhanced` (temp 0.3, seed 42, guided JSON) remains a **pending** decode refresh — the committed `v3/` packages are greedy-decode ablations.

| Package dir | Model | Tiers | Parse | Report |
|---|---|---:|---:|---|
| [`psych755_vllm_llama31_8b_instruct_v3_full_cohort_20260729_1039`](v3/psych755_vllm_llama31_8b_instruct_v3_full_cohort_20260729_1039/) | `meta-llama/Llama-3.1-8B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama32_3b_instruct_v3_full_cohort_20260729_1039`](v3/psych755_vllm_llama32_3b_instruct_v3_full_cohort_20260729_1039/) | `meta-llama/Llama-3.2-3B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama32_3b_v3_full_cohort_20260729_1039`](v3/psych755_vllm_llama32_3b_v3_full_cohort_20260729_1039/) | `meta-llama/Llama-3.2-3B` | 8 | 0.0% | yes |
| [`psych755_vllm_llama33_70b_instruct_awq_v3_full_cohort_20260729_1039`](v3/psych755_vllm_llama33_70b_instruct_awq_v3_full_cohort_20260729_1039/) | `meta-llama/Llama-3.3-70B-Instruct-AWQ` | 8 | 100.0% | yes |

## `transit_focus/` — TF1/TF2: predict regular transit + Q26 with mobility held out.

_Empty — new packages will appear here._

## `prior_v3_greedy/` — Archived greedy v3 runs (predictions identical to the stamped `v3/` packages; kept for the 20260728 record).

| Package dir | Model | Tiers | Parse | Report |
|---|---|---:|---:|---|
| [`psych755_vllm_llama31_8b_instruct_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama31_8b_instruct_v3_prior_greedy_full_cohort_20260728_2210/) | `meta-llama/Llama-3.1-8B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama32_3b_instruct_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama32_3b_instruct_v3_prior_greedy_full_cohort_20260728_2210/) | `meta-llama/Llama-3.2-3B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama33_70b_instruct_awq_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama33_70b_instruct_awq_v3_prior_greedy_full_cohort_20260728_2210/) | `casperhansen/llama-3.3-70b-instruct-awq` | 8 | 100.0% | yes |

## Not on disk yet (will appear as queues finish)

- `v2/psych755_vllm_llama32_3b_v2_full_cohort_<stamp>/` — Llama-3.2-3B (base) 
- `v2/psych755_vllm_llama33_70b_instruct_awq_v2_full_cohort_<stamp>/` — Llama-3.3-70B-Instruct-AWQ 
- `v3/psych755_vllm_deepseek_r1_distill_llama8b_v3_full_cohort_<stamp>/` — DeepSeek-R1-Distill-Llama-8B 
- `transit_focus/psych755_vllm_llama31_8b_instruct_transit_focus_full_cohort_<stamp>/` — Llama-3.1-8B-Instruct 
- `transit_focus/psych755_vllm_llama32_3b_instruct_transit_focus_full_cohort_<stamp>/` — Llama-3.2-3B-Instruct 
- `transit_focus/psych755_vllm_deepseek_r1_distill_llama8b_transit_focus_full_cohort_<stamp>/` — DeepSeek-R1-Distill-Llama-8B 
- `transit_focus/psych755_vllm_llama33_70b_instruct_awq_transit_focus_full_cohort_<stamp>/` — Llama-3.3-70B-Instruct-AWQ 

---

Refresh this file anytime:

```bash
python scripts/refresh_vllm_export_index.py
```

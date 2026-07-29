# vLLM export index

_Regenerated 2026-07-28 23:00:02_

Layout:

```
exports/
  INDEX.md              ← this file
  v1/                   ← prompt-v1 baselines (done)
  v2/                   ← enhanced 5-tier (filling now)
  v3/                   ← enhanced 8-tier (queued)
  prior_v3_greedy/      ← archived pre-refresh v3
  zips/
```

New packages from `scripts/package_vllm_export.py` land under
`exports/<bucket>/psych755_vllm_<tag>_full_cohort_<stamp>/`
(bucket inferred from the tag suffix `_v1` / `_v2` / `_v3` / `prior_greedy`).

## Enhanced queue matrix (what will be new)

| Status | Bucket | Tag | Model |
|---|---|---|---|
| ✅ done | `v2` | `llama31_8b_instruct_v2` | Llama-3.1-8B-Instruct |
| 🔄 running | `v2` | `llama32_3b_v2` | Llama-3.2-3B (base) |
| ⏳ pending | `v2` | `llama32_3b_instruct_v2` | Llama-3.2-3B-Instruct |
| ⏳ pending | `v2` | `deepseek_r1_distill_llama8b_v2` | DeepSeek-R1-Distill-Llama-8B |
| ⏳ pending | `v2` | `llama33_70b_instruct_awq_v2` | Llama-3.3-70B-Instruct-AWQ |
| ⏳ pending | `v3` | `llama31_8b_instruct_v3` | Llama-3.1-8B-Instruct |
| ⏳ pending | `v3` | `llama32_3b_v3` | Llama-3.2-3B (base) |
| ⏳ pending | `v3` | `llama32_3b_instruct_v3` | Llama-3.2-3B-Instruct |
| ⏳ pending | `v3` | `deepseek_r1_distill_llama8b_v3` | DeepSeek-R1-Distill-Llama-8B |
| ⏳ pending | `v3` | `llama33_70b_instruct_awq_v3` | Llama-3.3-70B-Instruct-AWQ |

**Queue currently on:** `llama32_3b_v2`

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
| [`psych755_vllm_llama31_8b_instruct_v2_full_cohort_20260728_2214`](v2/psych755_vllm_llama31_8b_instruct_v2_full_cohort_20260728_2214/) | `meta-llama/Llama-3.1-8B-Instruct` | 5 | 99.9% | yes |

## `v3/` — 8-tier ablations + `v3_enhanced` / `large_model` presets (refreshed anti-bleed).

_Empty — new packages will appear here._

## `prior_v3_greedy/` — Pre-refresh greedy v3 archives (comparison only; not the new target).

| Package dir | Model | Tiers | Parse | Report |
|---|---|---:|---:|---|
| [`psych755_vllm_llama31_8b_instruct_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama31_8b_instruct_v3_prior_greedy_full_cohort_20260728_2210/) | `meta-llama/Llama-3.1-8B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama32_3b_instruct_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama32_3b_instruct_v3_prior_greedy_full_cohort_20260728_2210/) | `meta-llama/Llama-3.2-3B-Instruct` | 8 | 100.0% | yes |
| [`psych755_vllm_llama33_70b_instruct_awq_v3_prior_greedy_full_cohort_20260728_2210`](prior_v3_greedy/psych755_vllm_llama33_70b_instruct_awq_v3_prior_greedy_full_cohort_20260728_2210/) | `casperhansen/llama-3.3-70b-instruct-awq` | 8 | 100.0% | yes |

## Not on disk yet (will appear as queue finishes)

- `v2/psych755_vllm_llama32_3b_v2_full_cohort_<stamp>/` — Llama-3.2-3B (base) ← running
- `v2/psych755_vllm_llama32_3b_instruct_v2_full_cohort_<stamp>/` — Llama-3.2-3B-Instruct 
- `v2/psych755_vllm_deepseek_r1_distill_llama8b_v2_full_cohort_<stamp>/` — DeepSeek-R1-Distill-Llama-8B 
- `v2/psych755_vllm_llama33_70b_instruct_awq_v2_full_cohort_<stamp>/` — Llama-3.3-70B-Instruct-AWQ 
- `v3/psych755_vllm_llama31_8b_instruct_v3_full_cohort_<stamp>/` — Llama-3.1-8B-Instruct 
- `v3/psych755_vllm_llama32_3b_v3_full_cohort_<stamp>/` — Llama-3.2-3B (base) 
- `v3/psych755_vllm_llama32_3b_instruct_v3_full_cohort_<stamp>/` — Llama-3.2-3B-Instruct 
- `v3/psych755_vllm_deepseek_r1_distill_llama8b_v3_full_cohort_<stamp>/` — DeepSeek-R1-Distill-Llama-8B 
- `v3/psych755_vllm_llama33_70b_instruct_awq_v3_full_cohort_<stamp>/` — Llama-3.3-70B-Instruct-AWQ 

---

Refresh this file anytime:

```bash
python scripts/refresh_vllm_export_index.py
```

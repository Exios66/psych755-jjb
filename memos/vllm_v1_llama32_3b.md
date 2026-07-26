---
title: "Memo: Does Llama-3.2-3B-Instruct recover PRCA on prompt v1?"
subtitle: "Research memorandum — full-cohort vLLM baseline (Terrarium 5-tier ladder)"
author: Jack J. Burleson
date: 2026-07-26
---

**Research question:** Does the smaller `meta-llama/Llama-3.2-3B-Instruct` model recover PRCA group and interpersonal scores on the same prompt-v1 ladder as the 8B baseline — and does it avoid the transit-induced interpersonal failure seen in Llama-3.1?

**Formal write-up:** [`docs/llm_baseline_llama32_instruct_v1.md`](../docs/llm_baseline_llama32_instruct_v1.md)  
**Cross-model memo:** [`vllm_v1_cross_model_comparison.md`](vllm_v1_cross_model_comparison.md)

---

## Answer, Response, + Summary of Results

Full-cohort vLLM run (**n = 241**; **1,205** prompts; parse **100%**; ~21.0 samples/s; export `psych755_vllm_llama32_3b_instruct_full_cohort_20260726_0252`). A parallel `Llama-3.2-3B` **base** (non-instruct) run only parsed ~7% of outputs and is excluded from inference claims.

**Short answer:** Still not a digital twin (exact match 9.1% / 7.7%), but **better overall MAE than Llama-3.1-8B** (group **5.51** vs 5.92; IP **5.35** vs 5.82) and **much stronger group band accuracy** (**52.7%** vs 28.8%). Transit still **hurts** interpersonal MAE (+0.58 vs demos) but far less than the 8B collapse (+3.51). IP band accuracy remains near chance (~30%). Best group MAE is at `full` (**5.29**), still above Ridge ML (~4.49).

### Overall vs Llama-3.1-8B (same prompts)

| Metric (all tiers) | 3B-Instruct | 8B-Instruct |
|---|---:|---:|
| MAE group | **5.51** | 5.92 |
| MAE interpersonal | **5.35** | 5.82 |
| Band acc group | **52.7%** | 28.8% |
| Band acc interpersonal | 30.0% | **40.2%** |
| Exact group / IP | 9.1% / 7.7% | 6.1% / 9.0% |

### Metrics by tier

| Tier | MAE G | MAE IP | Band G | Band IP | Δ MAE G vs demos | Δ MAE IP vs demos |
|---|---:|---:|---:|---:|---:|---:|
| demos | 5.59 | **5.07** | 53.9% | 29.5% | — | — |
| employment | 5.59 | 5.25 | 53.5% | 29.0% | 0.00 | +0.18 |
| geo | 5.63 | 5.27 | 52.7% | 30.3% | +0.04 | +0.21 |
| transit | 5.47 | 5.65 | 52.7% | 29.5% | **−0.12** | +0.58 |
| full | **5.29** | 5.53 | 50.6% | **31.5%** | **−0.29** | +0.46 |

### Band confusion (pooled) — key pattern

Group predictions mass on **low** (including many moderate/high ground-truth rows), which inflates group band accuracy relative to 8B while still missing high-CA individuals. Interpersonal predictions mass on **moderate**, yielding ~chance band accuracy.

### RQ verdicts

- **RQ1 (employment):** No meaningful help (group flat; IP slightly worse).
- **RQ2 (transit):** Weak group MAE gain; IP MAE worse (+0.58) — same *direction* as 8B, much smaller magnitude.
- **RQ3 (full):** Best group MAE in the ladder; IP still worse than demos-only.

**Conclusion.** On prompt v1, **smaller Instruct ≠ worse** for this CA recovery task: 3B beats 8B on pooled MAE and group bands, and is more robust to transit text. It does **not** unlock interpersonal band recovery (8B’s demos-tier strength). Prefer 3B when reporting v1 group-band / overall-MAE baselines; keep 8B as the documented transit-failure case for packaging redesign.

*Sources:* `data/vllm/llama32_instruct.md` · `docs/llm_baseline_llama32_instruct_v1.md` · [github.com/Exios66/psych755-jjb](https://github.com/Exios66/psych755-jjb)

---

## What questions or uncertainties remain?

Is the group-band advantage mostly a low-CA prior (confusion matrix) rather than true calibration? Would v3 public-transit vs rideshare ablations change the modest transit IP penalty? How does 3B stereotyping by Sex/Age compare to 8B at demos vs full?

## What other analyses pair with this memo?

Cross-model synthesis: [`vllm_v1_cross_model_comparison.md`](vllm_v1_cross_model_comparison.md). Llama-3.1 failure modes: [`vllm_v1_llama31_8b.md`](vllm_v1_llama31_8b.md). DeepSeek distill: [`vllm_v1_deepseek_r1_distill.md`](vllm_v1_deepseek_r1_distill.md).

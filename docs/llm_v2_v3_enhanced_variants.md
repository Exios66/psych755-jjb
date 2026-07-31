---
title: "v2/v3 enhanced variants"
subtitle: "Literature- and data-backed improvements for the next GPU digital-twin rounds"
---

**Evidence base:** Llama-3.1 transit IP collapse ([baseline](llm_baseline_llama31_v1.md)); 70B mode collapse ([baseline](llm_baseline_llama33_70b_v1.md)); Q28 dominance ([memo](../memos/q27_q28_predict_transit.md)); Argyle et al. 2023 silicon sampling; Park et al. 2023/2024 generative agents; Hu & Collier 2024 persona effect; Cheng 2023 / Santurkar 2023 stereotyping.

**Code:** [`config/vllm_presets.yaml`](../config/vllm_presets.yaml) · [`src/ca_personas/personas.py`](../src/ca_personas/personas.py) · [`src/inference/predict_vllm.py`](../src/inference/predict_vllm.py) · [`src/ca_personas/stereotyping.py`](../src/ca_personas/stereotyping.py)

---

## What failed in v1 (data)

| Failure | Observation | Implication |
|---|---|---|
| Transit → IP bleed | Llama-3.1 IP MAE 4.67 → **8.17**; signed error −2.21 → **+6.52** | Mobility language activated a high-IP prior |
| Kitchen-sink transit | Q27/Q29 even when Never; 4-dp geo | Token-heavy low-signal cues |
| Mode collapse | 70B ≈93% constant `(18,12)` | Greedy soft-JSON at scale is unsafe |
| Weak stereotyping ops | Pipeline wrote only Student/Employment MAE tables | Cannot fully answer “does error track group membership?” |

---

## Packaging enhancements (in code; GPU-evaluated — see [results](#results-july-2026-runs))

| Change | Rationale |
|---|---|
| 1-decimal geo; no country repeat | Cut low-value geo tokens (tabular lon/lat ≈ chance for transit) |
| Skip Q27/Q29 when Never | Remove Never→intensity stereotype fuel |
| Q26 → Q28 order; car last | Align with Q28 AUC **0.762** importance |
| Independent subscales + mid-scale (14–19) prior | Counter fused-ask bleed |
| **Mobility anti-bleed** system/user clauses | Explicit: transit/ride-share ≠ interpersonal CA |
| `v3_rideshare` / `v3_public_transit` / `v3_voice` | Disaggregate kitchen-sink confounders |

No demographic “don’t stereotype” rails — those would invalidate Cheng/Santurkar-style audits ([`persona_prompt_efficiency.md`](persona_prompt_efficiency.md)).

---

## Decode / runtime enhancements (presets)

| Preset | When to use | Key settings |
|---|---|---|
| `v1_baseline` | Reproduce published v1 | temp=0, top_p=1, no seed, no guided JSON |
| `v2_enhanced` | Signal-first 5-tier re-run (Llama-3.1 / 3.2) | temp=**0.3**, top_p=0.95, rep=1.05, **seed=42**, guided JSON |
| `v3_enhanced` | 8-tier ablations + DeepSeek | same + `max_output_tokens=512` |
| `large_model` | ≥70B after v1 collapse | temp=**0.5**, top_p=0.9, rep=1.1, seed=42, guided JSON |

**Literature backing**

- Mild temperature + fixed seed: reproducible silicon samples without pure greedy collapse (Argyle et al. 2023; contrast 70B v1).
- Guided JSON: reduce format failures without changing the psychometric ask (soft schema already in system prompt).
- Persona packaging sensitivity: Hu & Collier (2024) — treat packaging + decode as jointly manipulated factors.

```bash
# Priority GPU recipe (Llama-3.1 packaging test)
python -m inference.export_prompts --tiers demos employment geo transit full \
  --output-dir outputs/vllm_prompts_v2
VLLM_PRESET=v2_enhanced MODEL=meta-llama/Llama-3.1-8B-Instruct ./scripts/run_vllm.sh

# Ablation round
python -m inference.export_prompts --output-dir outputs/vllm_prompts_v3  # all 8 tiers
VLLM_PRESET=v3_enhanced MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B ./scripts/run_vllm.sh
```

## Results (July 2026 runs)

GPU exports under `exports/v2/` (v2_enhanced decode; 5 tiers) and `exports/v3/` (greedy decode; 8 tiers) are evaluated with the project evaluator. Pooled metrics per model:

| Prompt version | Model | MAE group | MAE IP | Band G | Band IP | Notes |
|---|---|---:|---:|---:|---:|---|
| v2 | DeepSeek-R1-Distill-Llama-8B | **5.02** | **5.26** | 35.7% | 34.2% | Best live result; transit IP 5.09 (no collapse) |
| v2 | Llama-3.1-8B-Instruct | 5.99 | 7.63 | 29.9% | 25.7% | Collapse persists (transit 8.23); demos IP 8.45 |
| v2 | Llama-3.2-3B-Instruct | 5.73 | 6.07 | 32.2% | 42.3% | v1 IP win (5.35) not replicated |
| v3 | Llama-3.1-8B-Instruct | 5.99 | 5.76 | 25.5% | 42.9% | Transit tier 7.77; ablations 4.85–5.92 |
| v3 | Llama-3.2-3B-Instruct | 5.72 | 6.81 | 41.8% | 40.6% | — |
| v3 | Llama-3.3-70B-Instruct-AWQ | 6.01 | 4.61† | 26.5% | 52.6%† | †Constant prior `(18, 12)` persists |

Llama-3.2-3B (base) failed JSON parsing in both v1 (7.1%) and v3 (0%) and is excluded. v3 packages are byte-identical to the archived `prior_v3_greedy` runs.

**Acceptance-criteria verdicts (vs v1 Llama baseline):**

1. **Transit IP no longer explodes?** *No for Llama-3.1* — v2 transit IP 8.23 ≈ v1 8.17, and demos IP worsens (4.67 → 8.45). DeepSeek v2 is flat (transit IP 5.09). Packaging changes error in model-specific directions.
2. **`v3_rideshare` ≥ `v3_public_transit` on group MAE?** *Yes* — Llama-3.1: 5.86 < 6.04, and `v3_rideshare` (5.86) beats the kitchen-sink `transit` tier (6.07).
3. **`v3_voice` helps without IP penalty?** *Yes* — group 5.64, IP 5.82 vs transit 6.07 / 7.77.
4. **Collapse combination-specific?** *Yes* — single cues stable (IP 4.85–5.92); only the bundled mobility dump collapses (7.77).
5. **Stereotyping gaps remain measurable?** *Pending* — run `ca-personas stereotype-eval` on each package’s `tables/02_evaluation_rowlevel.csv`.

Still open: canonical `v3_enhanced` decode refresh, v2/v3 on the base 3B, and 70B under `large_model`.

---

## Discriminatory / stereotyping evaluation (enhanced)

Primary RQs require testing whether **error correlates with group membership** and whether gaps **widen with tiers**.

```bash
ca-personas stereotype-eval --join inner --provider mock
# or after a live ingest:
ca-personas stereotype-eval --evaluation outputs/evaluation/evaluation.csv \
  --participants data/processed/participants.csv
```

Writes under `outputs/evaluation/stereotyping/`:

| Artifact | Content |
|---|---|
| `error_by_{sex,student_status,employment_status,age_bin,regular_transit,q28}.csv` | MAE + signed error by group × tier |
| `mae_gaps_by_tier.csv` | max−min MAE within tier (spread) |
| `mae_gap_deltas_vs_demos.csv` | Δ spread vs demos (widening) |
| `association_tests_by_tier.csv` | Kruskal–Wallis / Spearman Age tests |
| `stereotyping_results_card.json` | Machine-readable headlines |

**Interpretation rules**

1. Large `mae_*_gap` at `demos` ⇒ demographic stereotype surface before mobility context.
2. Positive `delta_mae_*_gap` at `transit`/`full` ⇒ richer context **widens** uneven error (context-sensitive stereotyping).
3. Mobility slices (`regular_transit`, `Q28`) test whether error tracks **exposure cues** rather than only Sex/Age — required because Q28 dominates tabular signal.
4. Signed `mean_error_*` shows over- vs under-prediction direction (Llama-3.1 transit = systematic high-IP over-prediction).
5. Significant Kruskal *p* with non-trivial ε² supports the claim that error is not exchangeable across groups.

Pipeline `ca-personas run` now emits the stereotyping battery automatically beside legacy Student/Employment CSVs.

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

## Packaging enhancements (already in code; GPU-evaluated on Llama-3.1 v2; v3 prior runs archived)

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

**Acceptance criteria** (vs v1 Llama baseline): transit IP MAE no longer explodes; `v3_rideshare` ≥ `v3_public_transit` on group MAE; `v3_voice` helps without IP penalty; stereotyping gaps remain measurable ([`persona_prompt_versions.md`](persona_prompt_versions.md)).

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

---
title: "Persona prompt packaging (v3.1)"
subtitle: "Signal-first Terrarium narratives for RQ1–RQ3 predictive efficiency"
---

**Code:** [`src/ca_personas/personas.py`](../src/ca_personas/personas.py)  
**System contract:** [`prompts/system_prompt.md`](../prompts/system_prompt.md)  
**Examples:** [`prompts/examples/`](../prompts/examples/)  
**Evidence:** [`llm_baseline_llama31_v1.md`](llm_baseline_llama31_v1.md) (prompt **v1** Llama-3.1) · [`persona_prompt_versions.md`](persona_prompt_versions.md) · [`factor_feature_importance.md`](factor_feature_importance.md)

---

## 1. Why packaging changed

A full-cohort Llama-3.1-8B-Instruct run (N = 241 × 5 tiers = 1,205 prompts) showed that **adding context did not uniformly help**:

| Tier vs `demos` | Δ Group MAE | Δ Interpersonal MAE |
|---|---:|---:|
| employment | −0.02 | +0.07 |
| geo | −0.03 | −0.04 |
| transit | −0.37 | **+3.51** |
| full | −0.21 | +2.25 |

Transit/full tiers **over-predicted interpersonal CA** (signed error ≈ +6.5 at `transit`). Tabular permutation importance ranks **Q28 > lon/lat > Q26 > employment**; **Q27/Q29/Q20/Q21 ≈ 0**. Older prompts still dumped 4-decimal coordinates, restated country in `geo`, and emitted rides-per-day even when frequency was `Never` — token-heavy cues that were easy for the LLM to misuse as anxiety signals.

## 2. Design rule (hybrid)

Keep the **same cumulative tier fields** so RQ1–RQ3 and stereotyping contrasts stay valid. Rewrite **packaging** for signal clarity and mid-scale calibration. Do **not** add “don’t stereotype by sex/employment” rails.

## 3. What v3.1 changes

| Layer | Change |
|---|---|
| `geo` | Approximate survey location at **1 decimal**; **do not** repeat country (already in `demos`); drop `UserLanguage` |
| `transit` | Order **Q26 → Q28 → license/car**; **skip Q27** when Q26 is Never; **skip Q29** when Q28 is Never |
| CA ask | Rate group vs interpersonal **independently**; note that mid-scale scores are common; no single circumstance determines CA |
| System prompt | Same inhabitance + JSON contract; add independent-subscale / non-deterministic-context sentences |

Core ladder topology is unchanged: `demos → employment → geo → transit → full`.

## 4. v3 ablation tiers (8 total)

Kitchen-sink `transit` / `full` confound the strongest CA cues. Three parallel tiers share the demos→employment→geo base, then tip with one focused signal:

| Tier | Tip | Role |
|---|---|---|
| `v3_rideshare` | Q28 (+ Q29 if used) | Isolate #1 tabular CA covariate |
| `v3_public_transit` | Q26 (+ Q27 if used) | Isolate public-transit CA association without rideshare/car |
| `v3_voice` | Q18.1 advice + Q19 mobility ideal | Isolate open-text attitude lift without transit dump |

`RESEARCH_TIERS` remains the original four for primary RQ tables; `TIERS` / default exports include all eight.

## 5. Regenerating the prompt DB

```bash
source .venv/bin/activate
ca-personas build-personas --join inner --output-dir outputs/personas
python -m inference.export_prompts --join inner --output-dir outputs/vllm_prompts
```

Expected: **1,928** prompts (241 × 8). Artifacts (gitignored):

- `outputs/personas/persona_prompts.csv` / `.db`
- `outputs/vllm_prompts/prompts.csv` + `ground_truth.csv`

## 6. Status and how to read a future re-run

**Status:** v3.1 packaging and the three v3 ablation tiers are **implemented in code** and covered by unit tests. Live GPU evaluation is committed for **v2** (signal-first packaging, `v2_enhanced` decode, on Llama-3.1-8B, Llama-3.2-3B-Instruct, and DeepSeek-R1-Distill-8B) and **v3** (greedy 8-tier ablations on Llama-3.1, Llama-3.2-3B-Instruct, and Llama-3.3-70B); the canonical `v3_enhanced` decode refresh remains pending. See [`persona_prompt_versions.md`](persona_prompt_versions.md) and the pooled results in [`memos/vllm_v2_v3_evaluation.qmd`](../memos/vllm_v2_v3_evaluation.qmd).

Compare a new LLM/vLLM export against [`llm_baseline_llama31_v1.md`](llm_baseline_llama31_v1.md):

1. Does interpersonal MAE at `transit`/`full` stop exploding under v3.1 packaging?
2. Does `v3_rideshare` beat `v3_public_transit` (and approach or beat full `transit`) on group MAE?
3. Does `v3_voice` help without the transit interpersonal penalty?
4. Do stereotyping MAE gaps by Sex / Employment / Student remain measurable?

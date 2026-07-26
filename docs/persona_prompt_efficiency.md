---
title: "Persona prompt packaging (v3.1)"
subtitle: "Signal-first Terrarium narratives for RQ1–RQ3 predictive efficiency"
---

**Code:** [`src/ca_personas/personas.py`](../src/ca_personas/personas.py)  
**System contract:** [`prompts/system_prompt.md`](../prompts/system_prompt.md)  
**Examples:** [`prompts/examples/`](../prompts/examples/)  
**Evidence:** [`data/vllm/llama3_1.md`](../data/vllm/llama3_1.md) · [`docs/factor_feature_importance.md`](factor_feature_importance.md)

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

Tier topology is unchanged: `demos → employment → geo → transit → full`.

## 4. Regenerating the prompt DB

```bash
source .venv/bin/activate
ca-personas build-personas --join inner \
  --tiers demos employment geo transit full \
  --output-dir outputs/personas
python -m inference.export_prompts --join inner \
  --tiers demos employment geo transit full \
  --output-dir outputs/vllm_prompts
```

Expected: **1,205** prompts (241 × 5). Artifacts (gitignored):

- `outputs/personas/persona_prompts.csv` / `.db`
- `outputs/vllm_prompts/prompts.csv` + `ground_truth.csv`

## 5. How to read a re-run

Compare a new LLM/vLLM export against [`data/vllm/llama3_1.md`](../data/vllm/llama3_1.md):

1. Does interpersonal MAE at `transit`/`full` stop exploding?
2. Do RQ1/RQ2 still show only modest employment/geo lifts (as the tabular floor predicts)?
3. Do stereotyping MAE gaps by Sex / Employment / Student remain measurable (packaging should not erase them)?

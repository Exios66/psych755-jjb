---
title: "v1 vLLM run specifications"
subtitle: "Models, context, prompts, sampling, and decode settings for prompt-v1 baselines"
---

**Companion:** [`llm_v2_v3_enhanced_variants.md`](llm_v2_v3_enhanced_variants.md) · [`persona_prompt_versions.qmd`](persona_prompt_versions.qmd) · [`config/vllm_presets.yaml`](../config/vllm_presets.yaml)

This page pins the **prompt-v1** digital-twin generation stack used for the four published full-cohort baselines (N = 241 × 5 = 1,205 prompts each). Primary manuscript claims remain on these runs.

---

## Models (prompt v1)

| Model | HF id | Role in v1 ranking |
|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Cautionary transit IP collapse |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Best group bands / IP MAE among non-collapsed |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Best pooled group MAE; tier-stable |
| Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | Mode-collapse cautionary case |

Export stamps (where documented): Llama-3.1 `20260726_0221`; Llama-3.2 `…_20260726_0252`; DeepSeek `…_20260726_0324`. Host for 8B/3B/DeepSeek: `rogers-gpu-1.discovery.wisc.edu` (RTX A5000; fp8 Marlin weight-only where stated).

---

## Prompt styling (v1)

| Layer | v1 specification |
|---|---|
| **Framing** | AI Terrarium / ICA 2026 digital twin: second-person narrative (“You are a …”), not checklist / “adopt this profile” |
| **System prompt** | Inhabit persona; first-person CA self-report; JSON-only contract with group + interpersonal scores (6–30) and bands (low/moderate/high). Historical v1 system text **omitted** the later independent-subscale / non-deterministic / mobility-anti-bleed clauses |
| **User prompt** | Cumulative narrative (`demos` → `employment` → `geo` → `transit` → `full`) + fused CA ask |
| **Geo** | High-precision lat/lon (≈4 decimal) |
| **Transit** | Full mobility dump including rides-per-day even when frequency was Never |
| **Instruction style** | Instruct-tuned chat template (`tokenizer.apply_chat_template`); system + user messages; DeepSeek may emit `<think>` traces before JSON (post-processed at ingest) |
| **JSON constraint** | Soft prompt only (“Respond with ONLY a JSON object”); **no** guided decoding in v1 |

Current tree code implements **v2/v3.1-enhanced packaging** (signal-first + anti-bleed). Historical v1 wording is preserved in this page and the published baseline write-ups, not as a live `--prompt-version` switch.

---

## Runtime / decode (v1)

Values below are the **`predict_vllm.py` / `run_vllm.sh` defaults** that applied when published launchers omitted sampling flags. Write-ups documented quantisation and throughput for some models but often omitted temperature; treat greedy defaults as the reproducible reconstruction.

| Parameter | v1 value | Notes |
|---|---:|---|
| `temperature` | **0.0** | Greedy decode (code default) |
| `top_p` | **1.0** | Nucleus disabled |
| `repetition_penalty` | **1.0** | Off |
| `seed` | **unset** | Not reproducible across engines |
| `max_output_tokens` | **256** | CA JSON headroom |
| `max_model_len` (context) | **8192** | Shell / CLI default |
| `quantization` | **fp8** | Documented for 3.1 / DeepSeek; assumed for siblings |
| `gpu_memory_utilization` | **0.9** | |
| `batch_size` / `save_freq` | **16** / **200** | Checkpoint-resume CSV |
| `tensor_parallel_size` | **1** (shell) / **2** (CLI default) | Published 8B runs used TP=1 on A5000 |
| Guided JSON | **false** | Prompt-only contract |

Preset name for reconstruction: `v1_baseline` in [`config/vllm_presets.yaml`](../config/vllm_presets.yaml).

```bash
VLLM_PRESET=v1_baseline ./scripts/run_vllm.sh
# or
python -m inference.predict_vllm --preset v1_baseline ...
```

---

## Why these settings matter for RQs

- **Greedy + soft JSON** recovered valid JSON at 100% for Llama instruct models, but **Llama-3.3-70B** collapsed to ≈93% constant prior `(18, 12)` — parse success ≠ twin success.
- **Fused transit dump + high-precision geo** coincided with Llama-3.1 interpersonal MAE 4.67 → **8.17** and signed error −2.2 → **+6.5** (systematic high-IP over-prediction).
- **No seed** limits exact replay of stochastic siblings; v2/v3 enhanced presets fix seed=42.

Enhanced decode + packaging for future re-runs: [`llm_v2_v3_enhanced_variants.md`](llm_v2_v3_enhanced_variants.md).

# Inference Module

Batch inference for **CA digital-twin** prompt CSVs using [vLLM](https://github.com/vllm-project/vllm)'s offline `LLM` engine.

Launcher structure adapted from [`ai_terrarium_v2`](https://github.com/Exios66/ai_terrarium_v2); schema and system prompt are this project's PRCA persona task.

## File overview

```
src/inference/
├── __init__.py              # Package exports
├── predict_vllm.py          # Core: vllm_predict() + CLI (+ Braintrust hooks)
├── utils.py                 # caseid helpers, checkpoint resume, HF token
├── ca_prompts.py            # PersonaPrompt ↔ caseid/prompt bridge
├── export_prompts.py        # CLI: build prompts.csv (+ ground_truth.csv)
├── ingest_results.py        # CLI: results.csv → CA predictions.csv
├── braintrust_tracing.py    # PRCA scorers + experiment logging helpers
├── braintrust_log_results.py# Post-hoc: results CSV → Braintrust experiment
├── braintrust_eval.py       # Offline Eval() for prompt comparison / bt eval
└── README.md

prompts/
└── braintrust_ca_system.py  # Pushable Braintrust prompt (bt functions push)

scripts/
├── run_vllm.sh           # One-command launcher (foreground or nohup)
└── run_vllm_monitor.sh   # Tail / poll / status for inference logs
```

## Input / output schema

### Input CSV

| Column | Required | Description |
|---|---|---|
| `caseid` | Yes | Unique row id (`{participant_id}__{tier}`) for checkpoint-resume |
| `prompt` | Yes | User-facing persona prompt from `ca_personas` |
| `answer` | No | Optional ground truth JSON; carried through to results |

### Output CSV

| Column | Description |
|---|---|
| `caseid` | Echoed from input |
| `answer` | Echoed from input (if present) |
| `generated_text` | Model JSON (newlines collapsed to spaces) |

## Quick start

```bash
pip install -e ".[vllm]"

# 1) Export digital-twin prompts in the vLLM schema.
# Defaults prefer ../sibling_data File A/B/C (cleaned analytic sample);
# falls back to data/excerpts/ when sibling data is absent.
python -m inference.export_prompts \
    --tiers demos employment geo transit full \
    --output-dir outputs/vllm_prompts
# Or pin paths explicitly:
# python -m inference.export_prompts \
#     --prolific ../sibling_data/PRCAProlificExport_FileA.csv \
#               ../sibling_data/PRCAProlificExport_FileB.csv \
#     --qualtrics ../sibling_data/PRCAQualtricsExport_FileC.csv \
#     --output-dir outputs/vllm_prompts

# 2) Run vLLM (needs CUDA + gated-model token for Llama)
echo "hf_YOUR_TOKEN" > hf_access_token.txt
./scripts/run_vllm.sh          # VLLM_TP_SIZE defaults to 1 in the launcher
# or:
python -m inference.predict_vllm \
    --prompt_csv outputs/vllm_prompts/prompts.csv \
    --result_csv outputs/vllm_results/results.csv \
    --ground_truth_csv outputs/vllm_prompts/ground_truth.csv \
    --gpu 0 --tensor_parallel_size 1 --quantization fp8

# 3) Ingest generations into the CA evaluation table
# Fails loudly if zero rows parse into CA JSON.
python -m inference.ingest_results \
    --result_csv outputs/vllm_results/results.csv \
    --predictions_csv outputs/predictions/vllm_predictions.csv
```

`--ground_truth_csv` coalesces / fills missing `answer` values even when the
prompt CSV already has a partial `answer` column, and validates completeness.

## Checkpoint-resume

If the result CSV already exists, rows whose `caseid` is present are skipped. Corrupt checkpoints (missing `caseid`, duplicates, empty header) raise a clear error so new rows are not appended blindly.

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model id or local path |
| `--gpu` | `0` | First GPU id |
| `--tensor_parallel_size` | `2` (CLI) / `1` (`run_vllm.sh` via `VLLM_TP_SIZE`) | Tensor-parallel GPU count |
| `--quantization` | `fp8` | `fp8`, `bitsandbytes`, `awq`, `gptq`, or `none` |
| `--max_output_tokens` | `256` | Headroom for CA JSON |
| `--batch_size` | `16` | Sub-batch size for `llm.generate` |
| `--save_freq` | `200` | Flush results every N rows |
| `--hf_access_token_file` | `hf_access_token.txt` | Token file for gated models |

See `python -m inference.predict_vllm --help` for the full list.

## Braintrust integration

Every vLLM path can log generations into a Braintrust **experiment** and load
the system prompt from the Braintrust **prompt registry** so playground edits
can be A/B'd without redeploying code.

### Install

```bash
pip install -e ".[vllm]"         # includes braintrust
# or scoring / prompt push only:
pip install -e ".[braintrust]"
```

### Enable

```bash
export BRAINTRUST_API_KEY=...          # required to upload
export BRAINTRUST_PROJECT=psych755-ca-personas
export BRAINTRUST_PROMPT_SLUG=ca-digital-twin-system
# optional pin:
# export BRAINTRUST_PROMPT_VERSION=<version>
./scripts/run_vllm.sh
```

When the API key is unset, logging is a no-op (local CSV still written). Use
`--no-braintrust` / `BRAINTRUST_ENABLED=false` to force off.

### Metrics logged per case

| Score (0–1, higher better) | Meaning |
|---|---|
| `parse_ok` | Generated text parses into valid CA JSON |
| `exact_match_{group,interpersonal}` | Predicted integer equals ground truth |
| `band_match_{group,interpersonal}` | Resolved PRCA band matches GT |
| `score_accuracy_*` | `1 − abs_error/24` |
| `band_accuracy_*` | `1 − band_distance/2` |
| `exact_match_mean` / `band_match_mean` / `score_accuracy_mean` | Aggregates |
| `inverse_mae_mean` | `1 − mae/24` for prompt ranking |

Raw MAE / signed error also land in Braintrust **metrics**.

### Prompt iteration loop

```bash
# 1) Push local SYSTEM_PROMPT into the registry (once / after local edits)
bt functions push prompts/braintrust_ca_system.py

# 2) Edit in Braintrust playground (slug: ca-digital-twin-system)

# 3) Re-run vLLM — predict_vllm loads the registry system message when keyed
BRAINTRUST_API_KEY=... ./scripts/run_vllm.sh

# 4) Or score an existing results CSV without regenerating
python -m inference.braintrust_log_results \
  --result_csv outputs/vllm_results/results.csv \
  --prompt_csv outputs/vllm_prompts/prompts.csv \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preset v2_enhanced

# 5) Offline Eval() / playground remote eval (task replays stored generations)
VLLM_RESULT_CSV=outputs/vllm_results/results.csv \
  python -m inference.braintrust_eval
```

When a playground version wins, copy the system text back into
`src/ca_personas/personas.py` and `prompts/system_prompt.md` so mock / Quarto
paths stay in sync.

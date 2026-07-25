# Inference Module

Batch inference for **CA digital-twin** prompt CSVs using [vLLM](https://github.com/vllm-project/vllm)'s offline `LLM` engine.

Launcher structure adapted from [`ai_terrarium_v2`](https://github.com/Exios66/ai_terrarium_v2); schema and system prompt are this project's PRCA persona task.

## File overview

```
src/inference/
├── __init__.py           # Package exports
├── predict_vllm.py       # Core: vllm_predict() + CLI
├── utils.py              # caseid helpers, checkpoint resume, HF token
├── ca_prompts.py         # PersonaPrompt ↔ caseid/prompt bridge
├── export_prompts.py     # CLI: build prompts.csv (+ ground_truth.csv)
├── ingest_results.py     # CLI: results.csv → CA predictions.csv
└── README.md

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

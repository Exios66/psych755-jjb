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

# 1) Export digital-twin prompts in the vLLM schema
python -m inference.export_prompts \
    --tiers demos employment geo transit full \
    --output-dir outputs/vllm_prompts

# 2) Run vLLM (needs CUDA + gated-model token for Llama)
echo "hf_YOUR_TOKEN" > hf_access_token.txt
./scripts/run_vllm.sh
# or:
python -m inference.predict_vllm \
    --prompt_csv outputs/vllm_prompts/prompts.csv \
    --result_csv outputs/vllm_results/results.csv \
    --ground_truth_csv outputs/vllm_prompts/ground_truth.csv \
    --gpu 0 --tensor_parallel_size 1 --quantization fp8

# 3) Ingest generations into the CA evaluation table
python -m inference.ingest_results \
    --result_csv outputs/vllm_results/results.csv \
    --predictions_csv outputs/predictions/vllm_predictions.csv
```

## Checkpoint-resume

If the result CSV already exists, rows whose `caseid` is present are skipped. Corrupt checkpoints (missing `caseid`, duplicates, empty header) raise a clear error so new rows are not appended blindly.

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model id or local path |
| `--gpu` | `0` | First GPU id |
| `--tensor_parallel_size` | `2` | Tensor-parallel GPU count |
| `--quantization` | `fp8` | `fp8`, `bitsandbytes`, `awq`, `gptq`, or `none` |
| `--max_output_tokens` | `256` | Headroom for CA JSON |
| `--batch_size` | `16` | Sub-batch size for `llm.generate` |
| `--save_freq` | `200` | Flush results every N rows |
| `--hf_access_token_file` | `hf_access_token.txt` | Token file for gated models |

See `python -m inference.predict_vllm --help` for the full list.

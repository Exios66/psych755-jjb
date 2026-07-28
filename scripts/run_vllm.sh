#!/usr/bin/env bash
#
# One-command vLLM launcher for CA digital-twin prompt CSVs.
#
# Usage:
#   ./scripts/run_vllm.sh
#   VLLM_PRESET=v2_enhanced ./scripts/run_vllm.sh
#   VLLM_PRESET=v3_enhanced MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B ./scripts/run_vllm.sh
#   VLLM_PRESET=large_model MODEL=meta-llama/Llama-3.3-70B-Instruct VLLM_TP_SIZE=2 ./scripts/run_vllm.sh
#   BACKGROUND=0 GPU=0 VLLM_TP_SIZE=1 ./scripts/run_vllm.sh
#
# Presets live in config/vllm_presets.yaml (v1_baseline | v2_enhanced | v3_enhanced | large_model).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROMPT_PATH="${PROMPT_PATH:-${PROJECT_ROOT}/outputs/vllm_prompts/prompts.csv}"
RESULT_PATH="${RESULT_PATH:-${PROJECT_ROOT}/outputs/vllm_results/results.csv}"
GROUND_TRUTH_CSV="${GROUND_TRUTH_CSV:-${PROJECT_ROOT}/outputs/vllm_prompts/ground_truth.csv}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU="${GPU:-0}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.9}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-200}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
VLLM_PRESET="${VLLM_PRESET:-v1_baseline}"
VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-}"
VLLM_TOP_P="${VLLM_TOP_P:-}"
VLLM_SEED="${VLLM_SEED:-}"
VLLM_GUIDED_JSON="${VLLM_GUIDED_JSON:-}"
HF_ACCESS_TOKEN_FILE="${HF_ACCESS_TOKEN_FILE:-hf_access_token.txt}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export HF_HOME

BACKGROUND="${BACKGROUND:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ ! -f "$PROMPT_PATH" ]]; then
  echo "Prompt CSV not found: $PROMPT_PATH" >&2
  echo "Export first: python -m inference.export_prompts" >&2
  exit 1
fi

mkdir -p "$(dirname "$RESULT_PATH")"
mkdir -p "${PROJECT_ROOT}/logging"

DATE_TAG="$(date +%Y%m%d)"
TIME_TAG="$(date +%H%M%S)"
LOG_PATH="${PROJECT_ROOT}/logging/${DATE_TAG}_ca_vllm_${TIME_TAG}.log"

GT_ARGS=()
if [[ -f "$GROUND_TRUTH_CSV" ]]; then
  GT_ARGS+=(--ground_truth_csv "$GROUND_TRUTH_CSV")
fi

CMD=(
  "$PYTHON_BIN" -m inference.predict_vllm
  --prompt_csv "$PROMPT_PATH"
  --result_csv "$RESULT_PATH"
  --gpu "$GPU"
  --model "$MODEL"
  --tensor_parallel_size "$VLLM_TP_SIZE"
  --max_model_len "$VLLM_MAX_MODEL_LEN"
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTIL"
  --quantization "$VLLM_QUANTIZATION"
  --batch_size "$BATCH_SIZE"
  --save_freq "$SAVE_FREQ"
  --preset "$VLLM_PRESET"
  --hf_access_token_file "$HF_ACCESS_TOKEN_FILE"
  "${GT_ARGS[@]}"
)
if [[ -n "$MAX_OUTPUT_TOKENS" ]]; then
  CMD+=(--max_output_tokens "$MAX_OUTPUT_TOKENS")
fi
if [[ -n "$VLLM_TEMPERATURE" ]]; then
  CMD+=(--temperature "$VLLM_TEMPERATURE")
fi
if [[ -n "$VLLM_TOP_P" ]]; then
  CMD+=(--top_p "$VLLM_TOP_P")
fi
if [[ -n "$VLLM_SEED" ]]; then
  CMD+=(--seed "$VLLM_SEED")
fi
if [[ "$VLLM_GUIDED_JSON" == "1" || "$VLLM_GUIDED_JSON" == "true" ]]; then
  CMD+=(--guided_json)
elif [[ "$VLLM_GUIDED_JSON" == "0" || "$VLLM_GUIDED_JSON" == "false" ]]; then
  CMD+=(--no-guided_json)
fi

{
  echo "==== CA digital-twin vLLM run ===="
  echo "date: $(date -Is)"
  echo "prompt_csv: $PROMPT_PATH"
  echo "result_csv: $RESULT_PATH"
  echo "model: $MODEL"
  echo "preset: $VLLM_PRESET"
  echo "gpu: $GPU"
  echo "tensor_parallel_size: $VLLM_TP_SIZE"
  echo "quantization: $VLLM_QUANTIZATION"
  echo "HF_HOME: $HF_HOME"
  echo "cmd: ${CMD[*]}"
  echo "================================="
} | tee "$LOG_PATH"

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >>"$LOG_PATH" 2>&1 &
  PID=$!
  echo "Started PID=$PID; log=$LOG_PATH"
  echo "$PID" > "${PROJECT_ROOT}/logging/latest_vllm.pid"
  echo "$LOG_PATH" > "${PROJECT_ROOT}/logging/latest_vllm.logpath"
else
  "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
fi

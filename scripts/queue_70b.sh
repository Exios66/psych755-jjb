#!/usr/bin/env bash
# Run 70B instruct model on v2 and transit_focus prompts
# Needs both GPUs (tensor_parallel_size=2) and large_model preset
set -euo pipefail

PROJECT="/mnt/ws/home/jburleson/755/psych755-jjb"
cd "$PROJECT"
mkdir -p logging

MODEL70B="/scratch/jburleson/hf_cache/models--casperhansen--llama-3.3-70b-instruct-awq/snapshots/64d255621f40b42adaf6d1f32a47e1d4534c0f14"

run_70b() {
    local name="$1" prompt_dir="$2" result_dir="$3" log="$4"
    local prompt_csv="$PROJECT/outputs/$prompt_dir/prompts.csv"
    local result_csv="$PROJECT/outputs/$result_dir/results.csv"
    local gt_csv="$PROJECT/outputs/$prompt_dir/ground_truth.csv"
    local sys_msg="$PROJECT/outputs/$prompt_dir/system_prompt.md"
    local logfile="$PROJECT/logging/$log"

    mkdir -p "$PROJECT/outputs/$result_dir"

    local extra=""
    if [ -f "$sys_msg" ]; then
        extra="--system_msg_file $sys_msg"
    fi

    echo "  Starting 70B on $prompt_dir (both GPUs)..."
    BRAINTRUST_ENABLED=0 WANDB_ENABLED=0 \
    nohup "$PROJECT/.venv/bin/python" -m inference.predict_vllm \
        --prompt_csv "$prompt_csv" \
        --result_csv "$result_csv" \
        --gpu 0 \
        --model "$MODEL70B" \
        --tensor_parallel_size 2 \
        --max_model_len 8192 \
        --gpu_memory_utilization 0.90 \
        --quantization awq \
        --batch_size 8 \
        --save_freq 100 \
        --preset large_model \
        --hf_access_token_file "$PROJECT/hf_access_token.txt" \
        --ground_truth_csv "$gt_csv" \
        --max_output_tokens 256 \
        --no-braintrust --no-wandb \
        $extra \
        >> "$logfile" 2>&1 &
    echo "  PID: $!"
    wait $!
    echo "  Done: $name"
}

echo "=== 70B on v2 prompts ==="
run_70b "70b_v2" "vllm_prompts_v2" "vllm_results_llama33_70b_instruct_awq_v2" "vllm_70b_v2.log"

echo "=== 70B on transit_focus prompts ==="
run_70b "70b_transit" "vllm_prompts_transit_focus" "vllm_results_llama33_70b_instruct_awq_transit_focus" "vllm_70b_transit.log"

echo "=== Pushing 70B results to GitHub ==="
git add -f outputs/vllm_results_llama33_70b_instruct_awq_v2/results.csv outputs/vllm_results_llama33_70b_instruct_awq_transit_focus/results.csv
git commit -m "vLLM 70B instruct results: v2 + transit_focus" 2>/dev/null || echo "Nothing new"
GIT_REMOTE="https://148591095:gho_fg8Kf2W3jYBq2zYOD2tQ2qAk0p8VA0@github.com/Exios66/psych755-jjb.git"
git remote set-url origin "$GIT_REMOTE"
git push origin main 2>&1
git remote set-url origin "https://github.com/Exios66/psych755-jjb.git"

echo "=== 70B jobs complete! ==="
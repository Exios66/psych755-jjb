#!/usr/bin/env bash
# Queue up remaining vLLM jobs on rogers-gpu-1
set -euo pipefail

PROJECT="/mnt/ws/home/jburleson/755/psych755-jjb"
cd "$PROJECT"
mkdir -p logging

start_job() {
    local name="$1" gpu="$2" model="$3" preset="$4" prompt_dir="$5" result_dir="$6" tokens="$7"
    local prompt_csv="$PROJECT/outputs/$prompt_dir/prompts.csv"
    local result_csv="$PROJECT/outputs/$result_dir/results.csv"
    local gt_csv="$PROJECT/outputs/$prompt_dir/ground_truth.csv"
    local sys_msg="$PROJECT/outputs/$prompt_dir/system_prompt.md"
    local log="$PROJECT/logging/vllm_${name}.log"
    local hf_token="$PROJECT/hf_access_token.txt"

    mkdir -p "$PROJECT/outputs/$result_dir"

    local extra=""
    if [ -f "$sys_msg" ]; then
        extra="--system_msg_file $sys_msg"
    fi

    echo "  Starting $name on GPU $gpu..."
    BRAINTRUST_ENABLED=0 WANDB_ENABLED=0 \
    CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$PROJECT/.venv/bin/python" -m inference.predict_vllm \
        --prompt_csv "$prompt_csv" \
        --result_csv "$result_csv" \
        --gpu 0 \
        --model "$model" \
        --tensor_parallel_size 1 \
        --max_model_len 8192 \
        --gpu_memory_utilization 0.90 \
        --quantization fp8 \
        --batch_size 16 \
        --save_freq 200 \
        --preset "$preset" \
        --hf_access_token_file "$hf_token" \
        --ground_truth_csv "$gt_csv" \
        --max_output_tokens "$tokens" \
        --no-braintrust --no-wandb \
        $extra \
        >> "$log" 2>&1 &
    echo "  PID: $!"
}

wait_all() {
    while pgrep -f 'predict_vllm' > /dev/null 2>&1; do
        echo "  Jobs still running... $(pgrep -f predict_vllm | wc -l) processes"
        sleep 60
    done
    echo "  All jobs complete!"
}

DEEPSEEK="/scratch/jburleson/hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

echo "=== Batch 1: v2 Llama-3.2-3B base (GPU 0) + v3 DeepSeek-R1 (GPU 1) ==="
start_job "llama32_3b_v2" "0" "meta-llama/Llama-3.2-3B" "v2_enhanced" "vllm_prompts_v2" "vllm_results_llama32_3b_v2" "256"
start_job "deepseek_v3" "1" "$DEEPSEEK" "v3_enhanced" "vllm_prompts_v3" "vllm_results_deepseek_r1_distill_llama8b_v3" "512"
wait_all

echo "=== Batch 2: Transit Focus - Llama-3.1-8B (GPU 0) + Llama-3.2-3B-Instruct (GPU 1) ==="
start_job "transit_llama31_8b" "0" "meta-llama/Llama-3.1-8B-Instruct" "v2_enhanced" "vllm_prompts_transit_focus" "vllm_results_llama31_8b_instruct_transit_focus" "256"
start_job "transit_llama32_3b_instruct" "1" "meta-llama/Llama-3.2-3B-Instruct" "v2_enhanced" "vllm_prompts_transit_focus" "vllm_results_llama32_3b_instruct_transit_focus" "256"
wait_all

echo "=== Batch 3: Transit Focus - Llama-3.2-3B (GPU 0) + DeepSeek-R1 (GPU 1) ==="
start_job "transit_llama32_3b_base" "0" "meta-llama/Llama-3.2-3B" "v2_enhanced" "vllm_prompts_transit_focus" "vllm_results_llama32_3b_base_transit_focus" "256"
start_job "transit_deepseek" "1" "$DEEPSEEK" "v3_enhanced" "vllm_prompts_transit_focus" "vllm_results_deepseek_r1_distill_llama8b_transit_focus" "512"
wait_all

echo "=== Pushing results to GitHub ==="
git add -f outputs/vllm_results_*/results.csv
git commit -m "vLLM inference results: batch v2+v3+transit completed" 2>/dev/null || echo "Nothing new to commit"
GIT_REMOTE="https://148591095:gho_fg8Kf2W3jYBq2zYOD2tQ2qAk0p8VA0@github.com/Exios66/psych755-jjb.git"
git remote set-url origin "$GIT_REMOTE"
git push origin main 2>&1
git remote set-url origin "https://github.com/Exios66/psych755-jjb.git"

echo "=== All done! ==="
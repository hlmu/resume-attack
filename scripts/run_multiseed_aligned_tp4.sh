#!/bin/bash
# Manuscript-aligned K=5 sweep for the 4 LOCAL-vLLM open-weight models.
#
# Root cause of the earlier K=5 vs manuscript gap was tensor-parallel size:
# manuscript (run_template.sh:68) uses TP=4; the first K=5 sweep used TP=1, which
# changes vLLM's float reduction order and shifts the sampled trajectories. With
# TP=4 + seed=42 the Qwen3-Think SM count reproduces manuscript (346 vs 348).
#
# This sweep aligns EVERYTHING except seed:
#   - TP=4 (GPU 2,3,4,5)
#   - NO --max-tokens cap          (server default; matches run_template.sh)
#   - NO --temperature             (vLLM generation_config default 0.6)
#   - NO --max-model-len           (default 40960)
#   - --add-think-parser           for Qwen3-Think / Qwen3-Nonthink / DeepSeek
#                                  (OFF for Llama, per no_think_parser_models)
#   - --enable-thinking true/false for Qwen3 Think / Nonthink
#   - build_prompts selects the manuscript top_k=5/min_score=0.5 subset (463 pairs)
#   - post-check retry ON (default)
#
# Models run SEQUENTIALLY on one TP=4 server (only GPU 2-5 are free; GPU 0,1 belong
# to another user). Each tag: 34 configs x 5 seeds x 463 pairs.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-resume-attack}"

LOGDIR=local_logs/multiseed_aligned_tp4
OUTDIR=results/revision_multi_seed_aligned
mkdir -p $LOGDIR stdout $OUTDIR

export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPUS=2,3,4,5
PORT=8012

QWEN_PATH=Qwen/Qwen3-8B
LLAMA_PATH=meta-llama/Llama-3.1-8B-Instruct
DEEPSEEK_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

start_vllm() {
  local model_path=$1
  local served=$2
  local logf=$3
  echo "[$(date +%T)] Starting TP=4 vLLM: $served"
  : > "$logf"
  CUDA_VISIBLE_DEVICES=$GPUS VLLM_WORKER_MULTIPROC_METHOD=spawn nohup vllm serve "$model_path" \
    --served-model-name "$served" \
    --dtype auto --trust-remote-code --tensor-parallel-size 4 \
    --port $PORT \
    > "$logf" 2>&1 &
  echo $! > /tmp/aligned_tp4_vpid
  for i in $(seq 1 120); do
    if grep -q "Application startup complete" "$logf" 2>/dev/null; then
      echo "[$(date +%T)] vLLM ready: $served"; sleep 3; return 0
    fi
    sleep 5
  done
  echo "[$(date +%T)] WARN: vLLM not ready after 10min: $served" >&2
  return 1
}

stop_vllm() {
  local vpid=$(cat /tmp/aligned_tp4_vpid 2>/dev/null)
  if [ -n "$vpid" ]; then
    echo "[$(date +%T)] Stopping vLLM PID=$vpid"
    kill $vpid 2>/dev/null
    # wait for GPU release
    for i in $(seq 1 30); do
      if ! kill -0 $vpid 2>/dev/null; then break; fi
      sleep 2
    done
    sleep 5
  fi
}

run_sweep() {
  local model_id=$1
  local model_tag=$2
  local extra=$3
  local logf=$4
  echo "[$(date +%T)] Sweep start: $model_tag"
  PYTHONUNBUFFERED=1 python -u eval/run_multiseed_api.py \
    --base-url http://localhost:$PORT/v1 --api-key token \
    --model-id "$model_id" --model-tag "$model_tag" \
    --seeds 0,1,2,3,4 \
    --max-prompts 463 --threads 16 \
    --out-dir $OUTDIR \
    $extra \
    > "$logf" 2>&1
  echo "[$(date +%T)] Sweep done: $model_tag"
}

# ---- Server 1: Qwen3-8B (shared by Think + Nonthink) ----
# Kill any leftover verification server on these GPUs, then start fresh.
pkill -f "vllm serve.*Qwen3-8B.*tensor-parallel-size 4" 2>/dev/null && sleep 8
start_vllm "$QWEN_PATH" "Qwen/Qwen3-8B" "stdout/aligned_tp4_vllm_qwen3_8b.log"

run_sweep "Qwen/Qwen3-8B" "Qwen_Qwen3-8B_aligned_multiseed" \
  "--enable-thinking true --add-think-parser" \
  "$LOGDIR/qwen3_8b_think.log"

run_sweep "Qwen/Qwen3-8B" "Qwen_Qwen3-8B_nonthink_aligned_multiseed" \
  "--enable-thinking false --add-think-parser" \
  "$LOGDIR/qwen3_8b_nonthink.log"

stop_vllm

# ---- Server 2: Llama-3.1-8B (no think parser) ----
start_vllm "$LLAMA_PATH" "meta-llama/Llama-3.1-8B-Instruct" "stdout/aligned_tp4_vllm_llama.log"
run_sweep "meta-llama/Llama-3.1-8B-Instruct" "meta-llama_Llama-3.1-8B-Instruct_aligned_multiseed" \
  "" \
  "$LOGDIR/llama_8b.log"
stop_vllm

# ---- Server 3: DeepSeek-R1-Distill-Llama-8B (think parser on) ----
start_vllm "$DEEPSEEK_PATH" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "stdout/aligned_tp4_vllm_deepseek.log"
run_sweep "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "deepseek-ai_DeepSeek-R1-Distill-Llama-8B_aligned_multiseed" \
  "--add-think-parser" \
  "$LOGDIR/deepseek_8b.log"
stop_vllm

echo "[$(date +%T)] ALL ALIGNED TP=4 SWEEPS DONE."

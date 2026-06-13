#!/bin/bash
# Wait for each FIDS LoRA to finish training, then evaluate it.
# Training seed K=5: each LoRA evaluated at T=0 (deterministic) on 32 cells x 463 pairs.
# Inference seed K=5: LoRA seed-0 also evaluated at T=0.7 with 5 different inference seeds.
#
# Runs on GPU 4 (does NOT compete with FIDS training on GPU 3 or 8B sweeps on GPU 0/1/2).

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-resume-attack}"
mkdir -p local_logs/fids_multiseed_eval

export CUDA_VISIBLE_DEVICES=4
PORT=8004
QWEN_PATH=Qwen/Qwen3-8B

eval_one_lora() {
  local seed=$1
  local lora_path=$2
  local model_tag=$3
  local extra_args=$4

  echo "[$(date +%T)] Starting vLLM for LoRA seed${seed}..."
  CUDA_VISIBLE_DEVICES=4 nohup vllm serve "$QWEN_PATH" \
    --served-model-name Qwen/Qwen3-8B \
    --dtype auto --trust-remote-code --tensor-parallel-size 1 \
    --port $PORT --max-model-len 8192 \
    --enable-lora \
    --lora-modules fids_seed${seed}=$lora_path \
    > stdout/multiseed_fids_seed${seed}_vllm.log 2>&1 &
  local VPID=$!

  # Wait for vLLM ready
  for i in $(seq 1 120); do
    if grep -q "Application startup complete" stdout/multiseed_fids_seed${seed}_vllm.log 2>/dev/null; then
      echo "[$(date +%T)] vLLM seed${seed} ready"
      break
    fi
    sleep 5
  done
  sleep 3

  # Run evaluation
  echo "[$(date +%T)] Running eval ${model_tag}..."
  PYTHONUNBUFFERED=1 python -u eval/run_multiseed_api.py \
    --base-url http://localhost:$PORT/v1 --api-key token \
    --model-id fids_seed${seed} \
    --model-tag $model_tag \
    --max-prompts 463 --max-tokens 256 --threads 16 \
    $extra_args \
    > local_logs/fids_multiseed_eval/${model_tag}.log 2>&1

  echo "[$(date +%T)] Eval ${model_tag} done. Stopping vLLM..."
  kill $VPID 2>/dev/null
  wait $VPID 2>/dev/null
}

# Wait for each LoRA training to finish, then evaluate
for seed in 0 1 2 3 4; do
  out=LLaMA-Factory/saves/fids_multiseed/seed${seed}

  # Wait up to 6 hours for adapter_config.json (training output)
  echo "[$(date +%T)] Waiting for LoRA seed${seed} to finish training..."
  for i in $(seq 1 360); do
    if [ -f "$out/adapter_config.json" ] && [ -f "$out/adapter_model.safetensors" ]; then
      echo "[$(date +%T)] LoRA seed${seed} found at $out"
      sleep 30  # extra grace
      break
    fi
    sleep 60
  done
  if [ ! -f "$out/adapter_config.json" ]; then
    echo "[$(date +%T)] LoRA seed${seed} not ready after 6h, aborting" >&2
    exit 1
  fi

  # 1) Single deterministic seed at T=0 for training-seed std
  eval_one_lora $seed $out "Qwen_Qwen3-8B_lora_fids_train${seed}" \
    "--seeds 42 --temperature 0.0 --enable-thinking false"

  # 2) For LoRA seed-0, also run K=5 inference seeds at T=0.7 for inference-seed std on FIDS row
  if [ $seed -eq 0 ]; then
    eval_one_lora $seed $out "Qwen_Qwen3-8B_lora_fids_infseeds" \
      "--seeds 0,1,2,3,4 --temperature 0.7 --enable-thinking false"
  fi

done
echo "[$(date +%T)] All FIDS evaluations done."

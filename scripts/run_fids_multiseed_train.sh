#!/bin/bash
# Train 5 FIDS LoRAs with different seeds, one after another on GPU 3.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-resume-attack}"
mkdir -p local_logs/fids_multiseed

export HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=3

cd LLaMA-Factory
for seed in 0 1 2 3 4; do
  out=saves/fids_multiseed/seed${seed}
  if [ -f "$out/adapter_config.json" ]; then
    echo "[$(date +%T)] seed${seed} already done, skipping"
    continue
  fi
  echo "[$(date +%T)] Training seed${seed}..."
  llamafactory-cli train examples/train_lora_multiseed/qwen3_8b_fids_seed${seed}.yaml \
    > ../local_logs/fids_multiseed/train_seed${seed}.log 2>&1
  echo "[$(date +%T)] seed${seed} done."
done
echo "[$(date +%T)] All 5 LoRA trainings finished."

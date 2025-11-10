#! /bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_logs/slurm-%j.out
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_TOKEN=""
eval "$(conda shell.bash hook)"
conda activate test
PORT=8000

# model_name="Qwen/QwQ-32B"
model_name="Qwen/Qwen3-8B"
model_name_escaped=$(echo "${model_name}" | sed 's/\//_/g')

cd LLaMA-Factory
llamafactory-cli train examples/train_lora/data_instruction_10k_injected_Qwen_Qwen3-8B_lf-lora-sft.yaml
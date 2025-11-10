# Script Usage Guide

## Overview

The `run.sh` and `run_template.sh` scripts have been modified to submit multiple separate SLURM jobs with different parameter combinations, with **optional LoRA (Low-Rank Adaptation) support**.

## LoRA Support

The scripts now support **flexible LoRA configuration**:

- **LoRA parameters are passed from `run.sh` to `run_template.sh`**
- **LoRA can be enabled or disabled** by setting the parameters in `run.sh`
- When LoRA is enabled, the vLLM server starts with `--enable-lora` and `--lora-modules` parameters
- When LoRA is disabled, the server runs with the base model only

### LoRA Configuration in run.sh

```bash
# LoRA configuration (leave empty to disable LoRA)
lora_name="qwen3-lora"  # Set to "" to disable LoRA
lora_path="LLaMA-Factory/saves/qwen3-8b/lora/sft"  # LoRA adapter path
```

### Enabling/Disabling LoRA

- **Enable LoRA**: Set both `lora_name` and `lora_path` to non-empty values
- **Disable LoRA**: Set `lora_name=""` (empty string)

## Usage

### Basic Usage
```bash
# Submit three separate SLURM jobs with configurations
./scripts/run.sh
```

This will submit three separate jobs:
1. **Normal configuration**: `--add-think-parser` with output suffix `normal`
2. **Adversarial configuration**: `--add-think-parser --add-adversarial-prompt` with output suffix `adv_1`
3. **Adversarial + Defense configuration**: `--add-think-parser --add-adversarial-prompt --add-defense-prompt` with output suffix `adv_1_defense_1`

### Test Configuration
```bash
# Test the current LoRA configuration
./scripts/test_lora.sh
```

## Behavior

- `run.sh` submits three separate SLURM jobs using `sbatch`
- Each job runs `run_template.sh` with different parameter combinations
- Each job will:
  1. Start a VLLM server for the specified model (**with or without LoRA** based on configuration)
  2. Run the `candidate_classifier_vllm.py` script with the specified parameters
  3. Stop the VLLM server when complete

## Output Files

Output files are saved in the `results/` directory with different formats based on LoRA usage:

### With LoRA Enabled
Format: `job_candidate_classification_{model_name_escaped}_lora_{suffix}.json`

Example files:
- `job_candidate_classification_Qwen_Qwen3-8B_lora_normal.json`
- `job_candidate_classification_Qwen_Qwen3-8B_lora_adv_1.json`
- `job_candidate_classification_Qwen_Qwen3-8B_lora_adv_1_defense_1.json`

### With LoRA Disabled
Format: `job_candidate_classification_{model_name_escaped}_{suffix}.json`

Example files:
- `job_candidate_classification_Qwen_Qwen3-8B_normal.json`
- `job_candidate_classification_Qwen_Qwen3-8B_adv_1.json`
- `job_candidate_classification_Qwen_Qwen3-8B_adv_1_defense_1.json`

## Customization

### Change Model
To use a different model, modify the `model_name` variable in `run.sh`:
```bash
model_name="Qwen/QwQ-32B"  # Change this line
```

### Enable LoRA
To use a LoRA adapter, set both parameters in `run.sh`:
```bash
lora_name="my-custom-lora"  # LoRA name for vLLM
lora_path="path/to/your/lora/adapter"  # Path to LoRA files
```

### Disable LoRA
To use only the base model, set `lora_name` to empty in `run.sh`:
```bash
lora_name=""  # Disables LoRA
lora_path=""  # Can be left empty when LoRA is disabled
```

### Multiple LoRA Adapters
You can create different versions of `run.sh` with different LoRA configurations:
1. Create `run_lora1.sh`, `run_lora2.sh`, etc.
2. Set different `lora_name` and `lora_path` in each
3. Run different LoRA experiments independently

## SLURM Job Monitoring

- Job logs are saved in `slurm_logs/slurm-{job_id}.out`
- VLLM server logs are saved in `stdout/{job_name}_{job_id}_{model_name_escaped}.vllm`

## LoRA Requirements (when enabled)

- Ensure your LoRA adapter files are present at the specified path
- LoRA adapters should be in PEFT format (compatible with vLLM)
- GPU compute capability >= 8.0 is required for LoRA support in vLLM
- The base model should support LoRA adapters in vLLM

## Examples

### Example 1: Run with LoRA
```bash
# In run.sh:
lora_name="qwen3-lora"
lora_path="LLaMA-Factory/saves/qwen3-8b/lora/sft"

./scripts/run.sh  # Will use LoRA
```

### Example 2: Run without LoRA
```bash
# In run.sh:
lora_name=""  # Empty string disables LoRA
lora_path=""

./scripts/run.sh  # Will use base model only
``` 
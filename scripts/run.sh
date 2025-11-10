#! /bin/bash

# model_name="Qwen/Qwen3-8B"
# model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_name="gemini-2.5-flash"
model_name="google/gemini-2.5-flash"

# LoRA configuration (leave empty to disable LoRA)
# lora_name="qwen3-lora"  # Set to "" to disable LoRA
# lora_path="LLaMA-Factory/saves/qwen3-8b/lora/sft"  # LoRA adapter path
lora_name=""  # Set to "" to disable LoRA
lora_path=""  # LoRA adapter path

dataset_name="job_matching_reverse_150"

# Define adversarial types and positions
adversarial_types=("instruction" "invisible_keywords" "invisible_experience" "job_manipulation")
adversarial_positions=("about_beginning" "about_end" "metadata" "resume_end")

# Define models that should NOT use --add-think-parser
no_think_parser_models=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct")

# Check if current model should use think parser
think_parser_arg="--add-think-parser"
for no_think_model in "${no_think_parser_models[@]}"; do
    if [[ "$model_name" == "$no_think_model" ]]; then
        think_parser_arg=""
        echo "Disabling think parser for model: $model_name"
        break
    fi
done

# Check if we should use direct execution for gemini model
if [[ "$model_name" == "google/gemini-2.5-flash" ]]; then
    echo "Using direct bash execution for gemini model with local logging..."
    
    # Create logs directory if it doesn't exist
    mkdir -p local_logs
    
    # Submit baseline jobs
    echo "Running baseline configuration (no adversarial, no defense)..."
    bash scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "" "" "baseline_normal" "$lora_name" "$lora_path" 2>&1 | tee "local_logs/baseline_normal.log"
    
    echo "Running baseline configuration (no adversarial, with defense)..."
    bash scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "" "--add-defense-prompt" "baseline_defense_only" "$lora_name" "$lora_path" 2>&1 | tee "local_logs/baseline_defense_only.log"
    
    # Submit jobs for all combinations of adversarial-type and adversarial-position
    for adv_type in "${adversarial_types[@]}"; do
        for adv_pos in "${adversarial_positions[@]}"; do
            output_suffix="adv_${adv_type}_${adv_pos}"
            echo "Running adversarial job: type=${adv_type}, position=${adv_pos}..."
            bash scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "--add-adversarial-prompt" "--adversarial-type ${adv_type} --adversarial-position ${adv_pos}" "${output_suffix}" "$lora_name" "$lora_path" 2>&1 | tee "local_logs/${output_suffix}.log"
            
            # Also submit with defense prompt
            output_suffix_defense="adv_${adv_type}_${adv_pos}_defense"
            echo "Running adversarial + defense job: type=${adv_type}, position=${adv_pos}..."
            bash scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "--add-adversarial-prompt" "--adversarial-type ${adv_type} --adversarial-position ${adv_pos} --add-defense-prompt" "${output_suffix_defense}" "$lora_name" "$lora_path" 2>&1 | tee "local_logs/${output_suffix_defense}.log"
        done
    done
else
    # Use sbatch for other models
    echo "Using sbatch for SLURM job submission..."
    
    # Submit baseline jobs
    echo "Submitting baseline configuration (no adversarial, no defense)..."
    sbatch scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "" "" "baseline_normal" "$lora_name" "$lora_path"

    echo "Submitting baseline configuration (no adversarial, with defense)..."
    sbatch scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "" "--add-defense-prompt" "baseline_defense_only" "$lora_name" "$lora_path"

    # Submit jobs for all combinations of adversarial-type and adversarial-position
    for adv_type in "${adversarial_types[@]}"; do
        for adv_pos in "${adversarial_positions[@]}"; do
            output_suffix="adv_${adv_type}_${adv_pos}"
            echo "Submitting adversarial job: type=${adv_type}, position=${adv_pos}..."
            sbatch scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "--add-adversarial-prompt" "--adversarial-type ${adv_type} --adversarial-position ${adv_pos}" "${output_suffix}" "$lora_name" "$lora_path"
            
            # Also submit with defense prompt
            output_suffix_defense="adv_${adv_type}_${adv_pos}_defense"
            echo "Submitting adversarial + defense job: type=${adv_type}, position=${adv_pos}..."
            sbatch scripts/run_template.sh "$model_name" "$dataset_name" "$think_parser_arg" "--add-adversarial-prompt" "--adversarial-type ${adv_type} --adversarial-position ${adv_pos} --add-defense-prompt" "${output_suffix_defense}" "$lora_name" "$lora_path"
        done
    done
fi

if [[ "$model_name" == "gemini-2.5-flash" ]]; then
    echo "Completed $(( 2 + ${#adversarial_types[@]} * ${#adversarial_positions[@]} * 2 )) jobs with local logs:"
    echo "- 1 baseline (no adversarial, no defense) -> local_logs/baseline_normal.log"
    echo "- 1 baseline (no adversarial, with defense) -> local_logs/baseline_defense_only.log"
    echo "- $(( ${#adversarial_types[@]} * ${#adversarial_positions[@]} )) adversarial configurations -> local_logs/adv_*.log"
    echo "- $(( ${#adversarial_types[@]} * ${#adversarial_positions[@]} )) adversarial + defense configurations -> local_logs/adv_*_defense.log"
    echo ""
    echo "All output was displayed on stdout and saved to local_logs/"
else
    echo "Submitted $(( 2 + ${#adversarial_types[@]} * ${#adversarial_positions[@]} * 2 )) jobs total:"
    echo "- 1 baseline (no adversarial, no defense)"
    echo "- 1 baseline (no adversarial, with defense)"
    echo "- $(( ${#adversarial_types[@]} * ${#adversarial_positions[@]} )) adversarial configurations"
    echo "- $(( ${#adversarial_types[@]} * ${#adversarial_positions[@]} )) adversarial + defense configurations"
fi

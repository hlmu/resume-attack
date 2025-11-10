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

# Parse command line arguments
model_name="${1:-Qwen/Qwen3-8B}"  # Default if not provided
dataset_name="${2:-job_matching_reverse_150}"
add_think_parser="$3"
add_adversarial_prompt="$4"
adversarial_params="$5"  # This contains adversarial-type, adversarial-position, and possibly defense params
output_suffix="$6"
lora_name="$7"  # LoRA name, empty string means no LoRA
lora_path="$8"  # LoRA adapter path

model_name_escaped=$(echo "${model_name}" | sed 's/\//_/g')

if [ "$model_name" = "google/gemini-2.5-flash" ]; then
    base_url="https://openrouter.ai/api/v1"
    api_key=""
    use_vllm=false
    use_lora=false
    echo "Using Gemini API configuration for model: $model_name"
else
    base_url="http://localhost:${PORT}/v1"
    api_key="token"
    use_vllm=true
    
    # Determine if LoRA should be used
    use_lora=false
    if [ -n "$lora_name" ] && [ -n "$lora_path" ]; then
        use_lora=true
        echo "LoRA enabled: $lora_name at $lora_path"
    else
        echo "LoRA disabled, using base model only"
    fi
fi

original_dir=$(pwd)
# Function to start the vllm server and monitor logs
start_vllm_server() {
    local model_name="$1"
    local log_file="$2"
    
    # Build vLLM command
    local vllm_cmd="vllm serve \"$model_name\" --dtype auto --trust_remote_code --tensor-parallel-size 4 --port ${PORT}"
    
    # Add LoRA parameters if enabled
    if [ "$use_lora" = true ]; then
        vllm_cmd="$vllm_cmd --enable-lora --lora-modules ${lora_name}=${lora_path}"
        echo "Starting vLLM server for model $model_name with LoRA ${lora_name} at ${lora_path}..."
    else
        echo "Starting vLLM server for model $model_name (base model only)..."
    fi
    
    # Start the vllm server in the background and redirect logs to a temporary file
    eval "$vllm_cmd" >>"$log_file" 2>&1 &
    SERVER_PID=$!
    echo "Started vLLM server with PID ${SERVER_PID}. Monitoring logs in ${log_file}..."
    sleep 10
    
    # Wait for the server to output a specific pattern indicating it's ready
    while ! grep -q "Application startup complete" "$log_file"; do
        echo "Waiting for the vLLM server ($model_name) to start..."
        sleep 5
    done
    echo "vLLM server is running and ready."
}
# Function to stop the vllm server
stop_vllm_server() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vllm server with PID ${SERVER_PID}..."
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null
        echo "vllm server stopped."
    else
        echo "No vllm server to stop."
    fi
}
# Only start vLLM server if not using Gemini
if [ "$use_vllm" = true ]; then
    vllm_output_file="${original_dir}/stdout/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${model_name_escaped}.vllm"
    mkdir -p "${original_dir}/stdout/"
    
    # Start the vllm server for current model
    start_vllm_server "${model_name}" "${vllm_output_file}"
fi

# Determine which model name to use for Python calls
if [ "$use_lora" = true ]; then
    python_model_name="$lora_name"
    output_file="results/results_d_${dataset_name}_m_${model_name_escaped}_lora_${output_suffix}.json"
    echo "Running with custom configuration using LoRA..."
else
    python_model_name="$model_name"
    output_file="results/results_d_${dataset_name}_m_${model_name_escaped}_${output_suffix}.json"
    echo "Running with custom configuration using base model..."
fi

# Build the command with specified parameters
cmd="python -m eval.candidate_classifier_vllm --base-url $base_url --api-key $api_key --model $python_model_name --reverse-matches results/$dataset_name.json"

# Add optional parameters
if [ -n "$add_think_parser" ]; then
    cmd="$cmd $add_think_parser"
fi

if [ -n "$add_adversarial_prompt" ]; then
    cmd="$cmd $add_adversarial_prompt"
fi

# Add additional parameters (type, position, defense, etc.)
if [ -n "$adversarial_params" ]; then
    cmd="$cmd $adversarial_params"
fi

cmd="$cmd --output \"$output_file\""

echo "Running command: $cmd"
eval "$cmd"

# Only stop vLLM server if it was started
if [ "$use_vllm" = true ]; then
    stop_vllm_server
fi

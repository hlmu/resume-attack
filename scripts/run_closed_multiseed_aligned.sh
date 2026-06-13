#!/bin/bash
# Restart of the 6 closed-source K=5 sweeps AFTER the build_prompts fix
# (manuscript top_k=5/min_score=0.5 selection -> the correct 463-pair subset).
#
# Why a restart: the first closed sweep used the OLD build_prompts, which covered
# only ~70 of 150 jobs (a different 463 subset), so its numbers were not comparable
# to the manuscript single-seed cells. The fix is in eval/run_multiseed_api.py.
#
# NOTE: TP does NOT apply here (all closed models are external HTTP APIs).
# Routing decisions (probed 2026-06-04):
#   - LLM-Center serves gemini / gpt-4o-mini / gpt-5-mini / gpt-5  (50 RPM per key)
#   - LLM-Center does NOT serve claude ("模型不存在")  -> claude must use openrouter+proxy
#   - claude via openrouter+proxy works (single call ~1.7s); the earlier 67% timeout
#     rate was concurrency (6 threads) overrunning openrouter limits, not an outage.
#     -> drop claude to 2 threads.
#   - gpt-oss-120b is handled SEPARATELY (subagent, LLM-Center, raised max_tokens).
#
# Output dir is SEPARATE so the old (wrong-subset) files are not overwritten and
# can be compared / discarded explicitly.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Provide an OpenAI-compatible endpoint that serves the closed models below.
# In the paper these were routed through an internal gateway; any provider that
# exposes the same model ids (OpenRouter, OpenAI, etc.) works.
LLM_CENTER_BASE_URL="${CLOSED_BASE_URL:-https://openrouter.ai/api/v1}"
LLM_CENTER_API_KEY="${CLOSED_API_KEY:-}"
OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"

LOG=local_logs/closed_multiseed_aligned
OUT=results/revision_multi_seed_aligned_closed
mkdir -p "$LOG" "$OUT"

CONDA="conda run -n ${CONDA_ENV:-resume-attack} --no-capture-output"
PY="$CONDA python eval/run_multiseed_api.py"

# 1. Claude 3.5 Haiku via openrouter (proxy). Throttle to 2 threads to stay under limits.
nohup $PY \
  --base-url "https://openrouter.ai/api/v1" --api-key "$OPENROUTER_KEY" \
  --proxy http://127.0.0.1:18890 \
  --model-id "anthropic/claude-3.5-haiku" --model-tag "claude-3.5-haiku_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 4096 --threads 2 \
  --out-dir "$OUT" > "$LOG/claude_haiku.log" 2>&1 &
echo "claude-3.5-haiku PID=$!"

# 2. Gemini 2.5 Flash
nohup $PY \
  --base-url "$LLM_CENTER_BASE_URL" --api-key "$LLM_CENTER_API_KEY" \
  --model-id "gemini-2.5-flash" --model-tag "gemini-2.5-flash_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 4096 --threads 8 \
  --out-dir "$OUT" > "$LOG/gemini_flash.log" 2>&1 &
echo "gemini-2.5-flash PID=$!"

# 3. GPT-4o-mini
nohup $PY \
  --base-url "$LLM_CENTER_BASE_URL" --api-key "$LLM_CENTER_API_KEY" \
  --model-id "gpt-4o-mini-2024-07-18" --model-tag "gpt-4o-mini_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 4096 --threads 8 \
  --out-dir "$OUT" > "$LOG/gpt4o_mini.log" 2>&1 &
echo "gpt-4o-mini PID=$!"

# 4. GPT-5 mini, reasoning effort=high
nohup $PY \
  --base-url "$LLM_CENTER_BASE_URL" --api-key "$LLM_CENTER_API_KEY" \
  --model-id "gpt-5-mini" --model-tag "gpt-5-mini-high_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 8192 --threads 6 \
  --reasoning-effort high \
  --out-dir "$OUT" > "$LOG/gpt5_mini_high.log" 2>&1 &
echo "gpt-5-mini-high PID=$!"

# 5. GPT-5 mini, reasoning effort=minimal
nohup $PY \
  --base-url "$LLM_CENTER_BASE_URL" --api-key "$LLM_CENTER_API_KEY" \
  --model-id "gpt-5-mini" --model-tag "gpt-5-mini-minimal_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 4096 --threads 8 \
  --reasoning-effort minimal \
  --out-dir "$OUT" > "$LOG/gpt5_mini_minimal.log" 2>&1 &
echo "gpt-5-mini-minimal PID=$!"

# 6. GPT-5, reasoning effort=minimal
nohup $PY \
  --base-url "$LLM_CENTER_BASE_URL" --api-key "$LLM_CENTER_API_KEY" \
  --model-id "gpt-5" --model-tag "gpt-5-minimal_aligned_multiseed" \
  --seeds 0,1,2,3,4 --temperature 0.7 --max-tokens 4096 --threads 6 \
  --reasoning-effort minimal \
  --out-dir "$OUT" > "$LOG/gpt5_minimal.log" 2>&1 &
echo "gpt-5-minimal PID=$!"

echo "All six closed-model aligned sweeps launched."
wait
echo "All six closed-model aligned sweeps completed."

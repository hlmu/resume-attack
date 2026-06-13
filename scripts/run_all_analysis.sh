#!/bin/bash
# Regenerate every paper table/figure from the shipped raw results (Stage 3).
# Run `git lfs pull` first so results/ is populated. This does NOT run inference;
# see REPRODUCE.md for the data-generation stages.
#
# Best-effort: each step is independent and logged; a failure in one does not
# abort the rest.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Unpack the compressed results bundle if not already extracted.
bash scripts/extract_results.sh

run() {  # label  command...
  local label=$1; shift
  echo "=============================================================="
  echo "[$(date +%T)] $label"
  echo "    $*"
  if "$@"; then echo "[ok] $label"; else echo "[FAILED] $label (rc=$?)"; fi
}

# 0) Aggregate the single-seed pipeline -> full_results.csv, latex_tables/, figures.
run "aggregate (full_results.csv, latex tables, figures)" \
    python eval/analyze_results.py --results-dir results/

# 1) Overall ASR tables (need results/latex_tables/ from step 0).
run "overall ASR by injection position (tab:overall_method_averaged_asr)" \
    python eval/analyze_method_averaged_data.py
run "overall ASR by attack type (tab:position_averaged_overall_asr)" \
    python eval/analyze_position_averaged_data.py

# 2) Native-trace covert analysis (tab:representative_failure_modes).
run "covert-trace analysis" python eval/why_fail_native.py

# 3) Defense tables (need results/analysis/full_results.csv from step 0).
run "Qwen3-8B defense tables, non-thinking" python eval/qwen3_8b_defense_tables.py --latex
run "Qwen3-8B defense tables, thinking"     python eval/qwen3_8b_think_defense_tables.py --latex

# 4) Utility / FRR (tab:utility_preservation).
run "utility preservation"        python eval/evaluate_utility_preservation.py
run "FRR Wilson + bootstrap CI"   python eval/frr_wilson_ci.py
run "FRR FIDS train multi-seed"   python eval/frr_fids_train_multiseed.py

# 5) Per-model attack ASR with Wilson CIs.
run "per-model Wilson ASR table"  python eval/per_model_wilson_table.py

# 6) Inter-model agreement (tab:inter_model_agreement).
run "inter-model agreement"       python eval/model_consistency_evaluation.py

# 7) Human annotation + consensus subset (tab:human-agreement, tab:consensus_not_match).
run "human consensus analysis"        python eval/extended_human_consensus_analysis.py
run "consensus-subset aggregate"      python eval/consensus_subset_aggregate_table.py
run "all-models consensus table"      python eval/all_models_consensus_table.py

# 8) Domain distribution (tab:domain_distribution).
run "domain distribution sampling"    python eval/domain_distribution_sampling.py

# 9) Parser end-to-end table (tab:parser_e2e).
run "parser end-to-end table"         python eval/parser_e2e_table.py --latex

echo "=============================================================="
echo "[$(date +%T)] analysis pipeline finished. Outputs under results/."

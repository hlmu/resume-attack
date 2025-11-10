#!/usr/bin/env python
"""
Evaluate the impact of defense mechanisms on legitimate (non-cheating) candidates.
This measures utility preservation - whether defenses cause false rejections.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_results(file_path: str) -> dict:
    """Load JSON results file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_baseline_metrics(results: dict) -> dict:
    """Extract metrics for legitimate candidates (no adversarial content)"""
    summary = results.get('summary', {})
    match_dist = summary.get('match_distribution', {})
    total = summary.get('total_classifications', 1)
    
    return {
        'total': total,
        'strong_match': match_dist.get('STRONG_MATCH', 0),
        'potential_match': match_dist.get('POTENTIAL_MATCH', 0),
        'not_match': match_dist.get('NOT_MATCH', 0),
        'acceptance_rate': (match_dist.get('STRONG_MATCH', 0) + match_dist.get('POTENTIAL_MATCH', 0)) / total if total > 0 else 0,
        'rejection_rate': match_dist.get('NOT_MATCH', 0) / total if total > 0 else 0,
    }

def calculate_utility_metrics(baseline_file: str, defense_file: str) -> dict:
    """Calculate utility preservation metrics between baseline and defense configurations"""
    
    # Load results
    baseline_results = load_results(baseline_file)
    defense_results = load_results(defense_file)
    
    # Extract metrics
    baseline_metrics = extract_baseline_metrics(baseline_results)
    defense_metrics = extract_baseline_metrics(defense_results)
    
    # Calculate utility preservation metrics
    utility_metrics = {
        'false_rejection_increase': defense_metrics['rejection_rate'] - baseline_metrics['rejection_rate'],
        'acceptance_rate_change': defense_metrics['acceptance_rate'] - baseline_metrics['acceptance_rate'],
        'utility_preservation_score': 1.0 - abs(defense_metrics['acceptance_rate'] - baseline_metrics['acceptance_rate']),
    }
    
    # Calculate classification shift matrix
    shift_matrix = {
        'strong_to_potential': 0,
        'strong_to_not': 0,
        'potential_to_strong': 0,
        'potential_to_not': 0,
        'not_to_strong': 0,
        'not_to_potential': 0,
    }
    
    # Compare individual classifications if available
    baseline_classifications = baseline_results.get('classifications', {})
    defense_classifications = defense_results.get('classifications', {})
    
    common_jobs = set(baseline_classifications.keys()) & set(defense_classifications.keys())
    
    for job_id in common_jobs:
        baseline_applicants = {a['profile_id']: a['classification'] 
                               for a in baseline_classifications[job_id].get('applicants', [])}
        defense_applicants = {a['profile_id']: a['classification'] 
                             for a in defense_classifications[job_id].get('applicants', [])}
        
        for profile_id in set(baseline_applicants.keys()) & set(defense_applicants.keys()):
            baseline_class = baseline_applicants[profile_id]
            defense_class = defense_applicants[profile_id]
            
            if baseline_class == 'STRONG_MATCH' and defense_class == 'POTENTIAL_MATCH':
                shift_matrix['strong_to_potential'] += 1
            elif baseline_class == 'STRONG_MATCH' and defense_class == 'NOT_MATCH':
                shift_matrix['strong_to_not'] += 1
            elif baseline_class == 'POTENTIAL_MATCH' and defense_class == 'STRONG_MATCH':
                shift_matrix['potential_to_strong'] += 1
            elif baseline_class == 'POTENTIAL_MATCH' and defense_class == 'NOT_MATCH':
                shift_matrix['potential_to_not'] += 1
            elif baseline_class == 'NOT_MATCH' and defense_class == 'STRONG_MATCH':
                shift_matrix['not_to_strong'] += 1
            elif baseline_class == 'NOT_MATCH' and defense_class == 'POTENTIAL_MATCH':
                shift_matrix['not_to_potential'] += 1
    
    return {
        'baseline_metrics': baseline_metrics,
        'defense_metrics': defense_metrics,
        'utility_metrics': utility_metrics,
        'classification_shifts': shift_matrix
    }

def evaluate_all_defenses(results_dir: str = 'results/') -> pd.DataFrame:
    """Evaluate utility preservation for all defense configurations"""
    
    configurations = [
        ('Prompt Defense', 
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json',
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_defense_only.json'),
        
        ('SFT Defense (LoRA)', 
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json',
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_normal.json'),
        
        ('Combined Defense', 
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json',
         f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_defense_only.json'),
    ]
    
    results = []
    
    for defense_name, baseline_file, defense_file in configurations:
        if os.path.exists(baseline_file) and os.path.exists(defense_file):
            metrics = calculate_utility_metrics(baseline_file, defense_file)
            
            results.append({
                'Defense Method': defense_name,
                'Baseline Acceptance Rate': f"{metrics['baseline_metrics']['acceptance_rate']*100:.1f}%",
                'Defense Acceptance Rate': f"{metrics['defense_metrics']['acceptance_rate']*100:.1f}%",
                'False Rejection Increase': f"{metrics['utility_metrics']['false_rejection_increase']*100:.1f}%",
                'Utility Preservation': f"{metrics['utility_metrics']['utility_preservation_score']*100:.1f}%",
                'Downgrades (Strong→Not)': metrics['classification_shifts']['strong_to_not'],
                'Downgrades (Potential→Not)': metrics['classification_shifts']['potential_to_not'],
            })
    
    return pd.DataFrame(results)

def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for utility preservation results"""
    
    latex = """\\begin{table}[h]
\\centering
\\caption{Impact of Defense Mechanisms on Legitimate Candidates}
\\label{tab:utility_preservation}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Defense} & \\textbf{Baseline} & \\textbf{With Defense} & \\textbf{FRR Increase} & \\textbf{Utility} & \\textbf{Downgrades} \\\\
\\textbf{Method} & \\textbf{Accept (\\%)} & \\textbf{Accept (\\%)} & \\textbf{(\\%)} & \\textbf{Score (\\%)} & \\textbf{to NOT} \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        downgrades = row['Downgrades (Strong→Not)'] + row['Downgrades (Potential→Not)']
        latex += f"{row['Defense Method']} & {row['Baseline Acceptance Rate']} & "
        latex += f"{row['Defense Acceptance Rate']} & {row['False Rejection Increase']} & "
        latex += f"{row['Utility Preservation']} & {downgrades} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\vspace{0.5em}
\\begin{flushleft}
\\footnotesize
\\textit{Note:} FRR = False Rejection Rate increase. Utility Score measures preservation of legitimate candidate classifications (100\\% = perfect preservation). Downgrades show candidates moved from MATCH categories to NOT\\_MATCH.
\\end{flushleft}
\\end{table}"""
    
    return latex

def analyze_hiring_impact(results_dir: str = 'results/') -> dict:
    """Analyze the practical impact on hiring decisions"""
    
    # Load baseline results
    baseline_file = f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json'
    lora_file = f'{results_dir}/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_normal.json'
    
    if not os.path.exists(baseline_file) or not os.path.exists(lora_file):
        return {}
    
    baseline = load_results(baseline_file)
    lora = load_results(lora_file)
    
    # Calculate hiring pool changes
    baseline_classifications = baseline.get('classifications', {})
    lora_classifications = lora.get('classifications', {})
    
    hiring_impact = {
        'jobs_analyzed': len(baseline_classifications),
        'baseline_avg_candidates_per_job': 0,
        'lora_avg_candidates_per_job': 0,
        'pool_size_change': 0,
        'jobs_with_reduced_pool': 0,
        'jobs_with_empty_pool': 0,
    }
    
    baseline_pools = []
    lora_pools = []
    
    for job_id in baseline_classifications.keys():
        if job_id in lora_classifications:
            baseline_matches = sum(1 for a in baseline_classifications[job_id]['applicants'] 
                                 if a['classification'] in ['STRONG_MATCH', 'POTENTIAL_MATCH'])
            lora_matches = sum(1 for a in lora_classifications[job_id]['applicants'] 
                             if a['classification'] in ['STRONG_MATCH', 'POTENTIAL_MATCH'])
            
            baseline_pools.append(baseline_matches)
            lora_pools.append(lora_matches)
            
            if lora_matches < baseline_matches:
                hiring_impact['jobs_with_reduced_pool'] += 1
            if lora_matches == 0 and baseline_matches > 0:
                hiring_impact['jobs_with_empty_pool'] += 1
    
    hiring_impact['baseline_avg_candidates_per_job'] = np.mean(baseline_pools) if baseline_pools else 0
    hiring_impact['lora_avg_candidates_per_job'] = np.mean(lora_pools) if lora_pools else 0
    hiring_impact['pool_size_change'] = hiring_impact['lora_avg_candidates_per_job'] - hiring_impact['baseline_avg_candidates_per_job']
    
    return hiring_impact

def main():
    parser = argparse.ArgumentParser(description='Evaluate utility preservation of defense mechanisms')
    parser.add_argument('--results-dir', default='results/', help='Directory containing result files')
    parser.add_argument('--output-dir', default='results/utility_analysis/', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Utility Preservation Analysis ===\n")
    
    # Evaluate all defense configurations
    df = evaluate_all_defenses(args.results_dir)
    
    if df.empty:
        print("No baseline comparison data found. Please ensure baseline evaluation has been completed.")
        return
    
    print("Defense Impact on Legitimate Candidates:")
    print(df.to_string(index=False))
    print()
    
    # Save results
    df.to_csv(os.path.join(args.output_dir, 'utility_preservation.csv'), index=False)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open(os.path.join(args.output_dir, 'utility_preservation.tex'), 'w') as f:
        f.write(latex_table)
    
    # Analyze hiring impact
    hiring_impact = analyze_hiring_impact(args.results_dir)
    
    if hiring_impact:
        print("\n=== Hiring Pool Impact Analysis ===")
        print(f"Jobs analyzed: {hiring_impact['jobs_analyzed']}")
        print(f"Average qualified candidates per job (Baseline): {hiring_impact['baseline_avg_candidates_per_job']:.1f}")
        print(f"Average qualified candidates per job (LoRA): {hiring_impact['lora_avg_candidates_per_job']:.1f}")
        print(f"Change in pool size: {hiring_impact['pool_size_change']:.1f} candidates")
        print(f"Jobs with reduced candidate pool: {hiring_impact['jobs_with_reduced_pool']} ({hiring_impact['jobs_with_reduced_pool']/hiring_impact['jobs_analyzed']*100:.1f}%)")
        print(f"Jobs with no remaining candidates: {hiring_impact['jobs_with_empty_pool']}")
        
        # Save hiring impact
        with open(os.path.join(args.output_dir, 'hiring_impact.json'), 'w') as f:
            json.dump(hiring_impact, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Files generated:")
    print("  - utility_preservation.csv: Detailed metrics")
    print("  - utility_preservation.tex: LaTeX table for paper")
    print("  - hiring_impact.json: Practical hiring pool analysis")

if __name__ == "__main__":
    main()
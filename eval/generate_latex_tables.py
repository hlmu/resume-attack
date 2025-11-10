#!/usr/bin/env python
"""
Generate LaTeX tables for the journal paper from evaluation results.

This script creates publication-ready tables comparing SFT-based and prompt-based defenses.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from pathlib import Path
import argparse

def load_results(file_path: str) -> dict:
    """Load JSON results file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(results: dict) -> dict:
    """Extract classification metrics from results"""
    summary = results.get('summary', {})
    match_dist = summary.get('match_distribution', {})
    total = summary.get('total_classifications', 1)
    
    return {
        'strong_match': match_dist.get('STRONG_MATCH', 0),
        'potential_match': match_dist.get('POTENTIAL_MATCH', 0),
        'not_match': match_dist.get('NOT_MATCH', 0),
        'total': total,
        'strong_match_rate': (match_dist.get('STRONG_MATCH', 0) / total * 100) if total > 0 else 0,
        'not_match_rate': (match_dist.get('NOT_MATCH', 0) / total * 100) if total > 0 else 0,
    }

def generate_main_comparison_table(results_dir: str = 'results/') -> str:
    """Generate main comparison table for defense methods"""
    
    # Define configurations to compare
    configs = [
        ('Base-None', 'results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json'),
        ('Base-Prompt', 'results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_defense_only.json'),
        ('LoRA-None', 'results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_normal.json'),
        ('LoRA-Prompt', 'results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_defense_only.json'),
    ]
    
    # Collect baseline metrics
    baseline_data = []
    for label, filename in configs:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            results = load_results(filepath)
            metrics = extract_metrics(results)
            baseline_data.append({
                'Configuration': label,
                'Strong Match (\\%)': f"{metrics['strong_match_rate']:.1f}",
                'Not Match (\\%)': f"{metrics['not_match_rate']:.1f}",
            })
    
    # Generate LaTeX table
    latex = """\\begin{table}[h]
\\centering
\\caption{Baseline Performance of Different Defense Configurations}
\\label{tab:baseline_comparison}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Configuration} & \\textbf{Strong Match (\\%)} & \\textbf{Not Match (\\%)} \\\\
\\midrule
"""
    
    for row in baseline_data:
        strong_match = row['Strong Match (\\%)']
        not_match = row['Not Match (\\%)']
        config = row['Configuration']
        latex += f"{config} & {strong_match} & {not_match} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

def generate_attack_effectiveness_table(results_dir: str = 'results/') -> str:
    """Generate table showing attack effectiveness across defense methods"""
    
    attack_types = ['instruction', 'invisible_keywords', 'invisible_experience', 'job_manipulation']
    positions = ['about_beginning', 'about_end', 'metadata', 'resume_end']
    
    # Collect attack success rates
    attack_data = {}
    
    for attack in attack_types:
        attack_data[attack] = {}
        for position in positions:
            # Base model without defense
            base_file = f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_adv_{attack}_{position}.json"
            base_path = os.path.join(results_dir, base_file)
            
            # LoRA model without defense
            lora_file = f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_adv_{attack}_{position}.json"
            lora_path = os.path.join(results_dir, lora_file)
            
            # Base model with defense
            base_def_file = f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_adv_{attack}_{position}_defense.json"
            base_def_path = os.path.join(results_dir, base_def_file)
            
            # LoRA model with defense
            lora_def_file = f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_adv_{attack}_{position}_defense.json"
            lora_def_path = os.path.join(results_dir, lora_def_file)
            
            rates = []
            for path in [base_path, base_def_path, lora_path, lora_def_path]:
                if os.path.exists(path):
                    results = load_results(path)
                    metrics = extract_metrics(results)
                    rates.append(metrics['strong_match_rate'])
                else:
                    rates.append(None)
            
            attack_data[attack][position] = rates
    
    # Generate LaTeX table
    latex = """\\begin{table*}[t]
\\centering
\\caption{Attack Success Rates (Strong Match \\%) Across Different Defense Configurations}
\\label{tab:attack_effectiveness}
\\small
\\begin{tabular}{llcccc}
\\toprule
\\multirow{2}{*}{\\textbf{Attack Type}} & \\multirow{2}{*}{\\textbf{Position}} & \\multicolumn{2}{c}{\\textbf{Base Model}} & \\multicolumn{2}{c}{\\textbf{LoRA Model}} \\\\
\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}
 & & No Defense & Prompt & No Defense & Prompt \\\\
\\midrule
"""
    
    for attack in attack_types:
        attack_label = attack.replace('_', ' ').title()
        first_row = True
        for position in positions:
            pos_label = position.replace('_', ' ').title()
            rates = attack_data[attack][position]
            
            if first_row:
                latex += f"\\multirow{{4}}{{*}}{{{attack_label}}} & {pos_label}"
                first_row = False
            else:
                latex += f" & {pos_label}"
            
            for rate in rates:
                if rate is not None:
                    latex += f" & {rate:.1f}"
                else:
                    latex += " & --"
            latex += " \\\\\n"
        
        if attack != attack_types[-1]:
            latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table*}"""
    
    return latex

def generate_improvement_summary_table(results_dir: str = 'results/') -> str:
    """Generate summary table of defense improvements"""
    
    # Calculate average improvements
    improvements = []
    
    # Collect all attack results
    attack_configs = []
    for attack in ['instruction', 'invisible_keywords', 'invisible_experience', 'job_manipulation']:
        for position in ['about_beginning', 'about_end', 'metadata', 'resume_end']:
            attack_configs.append(f"adv_{attack}_{position}")
    
    # Calculate averages for each defense configuration
    base_none_rates = []
    base_prompt_rates = []
    lora_none_rates = []
    lora_prompt_rates = []
    
    for config in attack_configs:
        # Load files
        files = [
            (f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_{config}.json", base_none_rates),
            (f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_{config}_defense.json", base_prompt_rates),
            (f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_{config}.json", lora_none_rates),
            (f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_{config}_defense.json", lora_prompt_rates),
        ]
        
        for filename, rate_list in files:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                results = load_results(filepath)
                metrics = extract_metrics(results)
                rate_list.append(metrics['strong_match_rate'])
    
    # Calculate averages and improvements
    base_none_avg = np.mean(base_none_rates) if base_none_rates else 0
    base_prompt_avg = np.mean(base_prompt_rates) if base_prompt_rates else 0
    lora_none_avg = np.mean(lora_none_rates) if lora_none_rates else 0
    lora_prompt_avg = np.mean(lora_prompt_rates) if lora_prompt_rates else 0
    
    # Generate LaTeX table
    latex = """\\begin{table}[h]
\\centering
\\caption{Defense Method Effectiveness Summary}
\\label{tab:defense_summary}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Defense Method} & \\textbf{Avg. Attack Success (\\%)} & \\textbf{Reduction vs. Baseline (\\%)} \\\\
\\midrule
"""
    
    # Add rows
    if base_none_avg > 0:
        latex += f"Base Model (No Defense) & {base_none_avg*100:.1f} & -- \\\\\n"
        
        prompt_reduction = ((base_none_avg - base_prompt_avg) / base_none_avg) * 100
        latex += f"Base Model + Prompt Defense & {base_prompt_avg*100:.1f} & {prompt_reduction:.1f} \\\\\n"
        
        sft_reduction = ((base_none_avg - lora_none_avg) / base_none_avg) * 100
        latex += f"LoRA Model (SFT Defense) & {lora_none_avg*100:.1f} & {sft_reduction:.1f} \\\\\n"
        
        combined_reduction = ((base_none_avg - lora_prompt_avg) / base_none_avg) * 100
        latex += f"LoRA Model + Prompt Defense & {lora_prompt_avg*100:.1f} & {combined_reduction:.1f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\vspace{0.5em}
\\begin{flushleft}
\\footnotesize
\\textit{Note:} Attack success is measured as the percentage of adversarial candidates classified as STRONG\\_MATCH. Lower values indicate better defense.
\\end{flushleft}
\\end{table}"""
    
    return latex

def generate_critical_attacks_table(results_dir: str = 'results/') -> str:
    """Generate table focusing on the most critical attacks (job_manipulation)"""
    
    positions = ['about_beginning', 'about_end', 'metadata', 'resume_end']
    
    # Collect job_manipulation results
    job_manip_data = []
    
    for position in positions:
        row_data = {'Position': position.replace('_', ' ').title()}
        
        # Load results for each configuration
        configs = [
            ('Base', f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_adv_job_manipulation_{position}.json"),
            ('Base+Prompt', f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_adv_job_manipulation_{position}_defense.json"),
            ('LoRA', f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_adv_job_manipulation_{position}.json"),
            ('LoRA+Prompt', f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_adv_job_manipulation_{position}_defense.json"),
        ]
        
        for label, filename in configs:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                results = load_results(filepath)
                metrics = extract_metrics(results)
                row_data[label] = metrics['strong_match_rate']
            else:
                row_data[label] = None
        
        job_manip_data.append(row_data)
    
    # Generate LaTeX table
    latex = """\\begin{table}[h]
\\centering
\\caption{Defense Effectiveness Against Job Manipulation Attacks}
\\label{tab:job_manipulation}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Attack Position} & \\textbf{Base} & \\textbf{Base+Prompt} & \\textbf{LoRA} & \\textbf{LoRA+Prompt} \\\\
\\midrule
"""
    
    for row in job_manip_data:
        latex += f"{row['Position']}"
        for config in ['Base', 'Base+Prompt', 'LoRA', 'LoRA+Prompt']:
            if row[config] is not None:
                latex += f" & {row[config]:.1f}"
            else:
                latex += " & --"
        latex += " \\\\\n"
    
    # Add average row
    latex += "\\midrule\n"
    latex += "\\textbf{Average}"
    for config in ['Base', 'Base+Prompt', 'LoRA', 'LoRA+Prompt']:
        values = [row[config] for row in job_manip_data if row[config] is not None]
        if values:
            avg = np.mean(values)
            latex += f" & \\textbf{{{avg:.1f}}}"
        else:
            latex += " & --"
    latex += " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\vspace{0.5em}
\\begin{flushleft}
\\footnotesize
\\textit{Note:} Job manipulation attacks have shown the highest success rates (81\\%+) in preliminary studies. Values show attack success rate (\\% classified as STRONG\\_MATCH).
\\end{flushleft}
\\end{table}"""
    
    return latex

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for journal paper')
    parser.add_argument('--results-dir', default='results/', help='Directory containing result files')
    parser.add_argument('--output-dir', default='results/latex_tables/', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Generating LaTeX Tables for Journal Paper ===\n")
    
    # Generate tables
    tables = []
    
    print("Generating main comparison table...")
    main_table = generate_main_comparison_table(args.results_dir)
    tables.append(('main_comparison.tex', main_table))
    
    print("Generating attack effectiveness table...")
    attack_table = generate_attack_effectiveness_table(args.results_dir)
    tables.append(('attack_effectiveness.tex', attack_table))
    
    print("Generating improvement summary table...")
    summary_table = generate_improvement_summary_table(args.results_dir)
    tables.append(('defense_summary.tex', summary_table))
    
    print("Generating critical attacks table...")
    critical_table = generate_critical_attacks_table(args.results_dir)
    tables.append(('job_manipulation.tex', critical_table))
    
    # Save tables
    for filename, content in tables:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Saved: {filename}")
    
    # Create combined file
    combined_path = os.path.join(args.output_dir, 'all_tables.tex')
    with open(combined_path, 'w') as f:
        f.write("% LaTeX Tables for SFT-based Defense Evaluation\n")
        f.write("% Generated automatically from evaluation results\n\n")
        
        for filename, content in tables:
            f.write(f"% ===== {filename} =====\n")
            f.write(content)
            f.write("\n\n")
    
    print(f"\nAll tables saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - main_comparison.tex: Baseline performance comparison")
    print("  - attack_effectiveness.tex: Full attack results matrix")
    print("  - defense_summary.tex: Defense method effectiveness summary")
    print("  - job_manipulation.tex: Focus on critical attack type")
    print("  - all_tables.tex: All tables in one file")
    print("\nYou can copy these tables directly into your LaTeX document!")

if __name__ == "__main__":
    main()
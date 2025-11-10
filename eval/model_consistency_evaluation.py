#!/usr/bin/env python3
"""
Model Classification Consistency Evaluation Script

Evaluates inter-model agreement for job-candidate classification task
across different LLM models: DeepSeek, Gemini, and Qwen.
Based on the annotation consistency analysis methodology.
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import argparse
import os

def load_model_results(file_path: str) -> Dict[str, Any]:
    """Load model results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_common_classifications(model1_data: Dict, model2_data: Dict, model3_data: Dict) -> Dict[str, Dict[str, List[str]]]:
    """Extract classifications for profile_ids that appear in all three models' data"""
    common_classifications = defaultdict(lambda: defaultdict(list))
    
    # Get all job_ids and profile_ids from each model
    jobs1 = set(model1_data['classifications'].keys())
    jobs2 = set(model2_data['classifications'].keys())
    jobs3 = set(model3_data['classifications'].keys())
    
    common_jobs = jobs1.intersection(jobs2).intersection(jobs3)
    
    for job_id in common_jobs:
        profiles1 = {app['profile_id']: app['classification'] for app in model1_data['classifications'][job_id]['applicants']}
        profiles2 = {app['profile_id']: app['classification'] for app in model2_data['classifications'][job_id]['applicants']}
        profiles3 = {app['profile_id']: app['classification'] for app in model3_data['classifications'][job_id]['applicants']}
        
        common_profiles = set(profiles1.keys()).intersection(set(profiles2.keys())).intersection(set(profiles3.keys()))
        
        for profile_id in common_profiles:
            common_classifications[job_id][profile_id] = [
                profiles1[profile_id],
                profiles2[profile_id], 
                profiles3[profile_id]
            ]
    
    return common_classifications

def calculate_pairwise_agreement(classifications1: List[str], classifications2: List[str]) -> Tuple[float, float]:
    """Calculate pairwise agreement and Cohen's kappa between two models"""
    if len(classifications1) != len(classifications2):
        return 0.0, 0.0
    
    if len(classifications1) == 0:
        return 0.0, 0.0
    
    # Simple agreement
    agreement = sum(1 for c1, c2 in zip(classifications1, classifications2) if c1 == c2) / len(classifications1)
    
    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(classifications1, classifications2)
    except:
        kappa = 0.0
    
    return agreement, kappa

def calculate_fleiss_kappa(classifications_matrix: List[List[str]]) -> float:
    """Calculate Fleiss' kappa for multiple models"""
    if not classifications_matrix or len(classifications_matrix[0]) < 2:
        return 0.0
    
    n_items = len(classifications_matrix)
    n_raters = len(classifications_matrix[0])
    
    # Get all unique categories
    all_categories = set()
    for item_classifications in classifications_matrix:
        all_categories.update(item_classifications)
    categories = sorted(list(all_categories))
    
    if len(categories) <= 1:
        return 1.0 if len(categories) == 1 else 0.0
    
    # Create rating matrix
    rating_matrix = np.zeros((n_items, len(categories)))
    for i, item_classifications in enumerate(classifications_matrix):
        for classification in item_classifications:
            j = categories.index(classification)
            rating_matrix[i, j] += 1
    
    # Calculate Fleiss' kappa
    # P_e: expected agreement by chance
    p_j = np.sum(rating_matrix, axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j ** 2)
    
    # P_o: observed agreement
    P_i = np.sum(rating_matrix * (rating_matrix - 1), axis=1) / (n_raters * (n_raters - 1))
    P_o = np.mean(P_i)
    
    if P_e == 1.0:
        return 1.0
    
    kappa = (P_o - P_e) / (1 - P_e)
    return kappa

def analyze_binary_classification(model1_data: Dict, model2_data: Dict, model3_data: Dict, model_names: List[str]):
    """Analyze agreement with binary classification (MATCH vs NOT_MATCH)"""
    
    def convert_to_binary(classification: str) -> str:
        """Convert 3-class to binary"""
        return "MATCH" if classification in ["STRONG_MATCH", "POTENTIAL_MATCH"] else "NOT_MATCH"
    
    # Extract common classifications and convert to binary
    common_classifications = extract_common_classifications(model1_data, model2_data, model3_data)
    
    all_common_classifications = []
    classification_pairs = {model_names[0]: [], model_names[1]: [], model_names[2]: []}
    
    for job_id, profiles in common_classifications.items():
        for profile_id, classifications in profiles.items():
            if len(classifications) == 3:  # All three models
                binary_classifications = [convert_to_binary(cls) for cls in classifications]
                all_common_classifications.append(binary_classifications)
                classification_pairs[model_names[0]].append(binary_classifications[0])
                classification_pairs[model_names[1]].append(binary_classifications[1])
                classification_pairs[model_names[2]].append(binary_classifications[2])
    
    n_common = len(all_common_classifications)
    print(f"Common Binary Classifications: {n_common} profile-job pairs")
    
    if n_common == 0:
        print("No common classifications found.")
        return
    
    # Binary distribution for each model
    print("\nBinary Match Distribution:")
    for model, classifications in classification_pairs.items():
        match_count = classifications.count("MATCH")
        not_match_count = classifications.count("NOT_MATCH")
        total = len(classifications)
        print(f"{model}: MATCH={match_count} ({match_count/total*100:.1f}%), "
              f"NOT_MATCH={not_match_count} ({not_match_count/total*100:.1f}%)")
    
    # Pairwise agreement analysis
    print("\nPairwise Agreement (Binary):")
    pairs = [
        (f"{model_names[0]} vs {model_names[1]}", classification_pairs[model_names[0]], classification_pairs[model_names[1]]),
        (f"{model_names[0]} vs {model_names[2]}", classification_pairs[model_names[0]], classification_pairs[model_names[2]]),
        (f"{model_names[1]} vs {model_names[2]}", classification_pairs[model_names[1]], classification_pairs[model_names[2]])
    ]
    
    for pair_name, cls1, cls2 in pairs:
        agreement, kappa = calculate_pairwise_agreement(cls1, cls2)
        print(f"{pair_name}: Agreement={agreement:.3f}, Cohen's κ={kappa:.3f}")
    
    # Overall agreement (Fleiss' kappa)
    fleiss_kappa = calculate_fleiss_kappa(all_common_classifications)
    print(f"\nOverall Agreement (Binary Fleiss' κ): {fleiss_kappa:.3f}")
    
    # Kappa interpretation
    def interpret_kappa(kappa):
        if kappa < 0: return "Poor"
        elif kappa < 0.20: return "Slight"
        elif kappa < 0.40: return "Fair"
        elif kappa < 0.60: return "Moderate"
        elif kappa < 0.80: return "Substantial"
        else: return "Almost Perfect"
    
    print(f"Binary Interpretation: {interpret_kappa(fleiss_kappa)}")
    
    # Agreement breakdown
    exact_matches = sum(1 for cls in all_common_classifications if len(set(cls)) == 1)
    disagreements = sum(1 for cls in all_common_classifications if len(set(cls)) > 1)
    
    print(f"\nBinary Agreement Breakdown:")
    print(f"Exact agreement (all 3 agree): {exact_matches}/{n_common} ({exact_matches/n_common*100:.1f}%)")
    print(f"Any disagreement: {disagreements}/{n_common} ({disagreements/n_common*100:.1f}%)")
    
    return fleiss_kappa

def analyze_model_consistency(model1_file: str, model2_file: str, model3_file: str, model_names: List[str]):
    """Main analysis function"""
    # Load data from all three models
    model1_data = load_model_results(model1_file)
    model2_data = load_model_results(model2_file)
    model3_data = load_model_results(model3_file)
    
    print("=== Model Classification Consistency Analysis ===\n")
    
    # Also analyze with binary classification (MATCH vs NOT_MATCH)
    print("=== BINARY CLASSIFICATION ANALYSIS (MATCH vs NOT_MATCH) ===\n")
    analyze_binary_classification(model1_data, model2_data, model3_data, model_names)
    print("\n" + "="*60 + "\n")
    
    print("=== THREE-CLASS ANALYSIS (STRONG/POTENTIAL/NOT_MATCH) ===\n")
    
    # Basic statistics
    print("Basic Statistics:")
    for i, (model_name, data) in enumerate(zip(model_names, [model1_data, model2_data, model3_data])):
        print(f"{model_name}: {data['summary']['total_classifications']} classifications across {data['summary']['total_jobs']} jobs")
    
    print("\nMatch Distribution:")
    for model_name, data in zip(model_names, [model1_data, model2_data, model3_data]):
        dist = data['summary']['match_distribution']
        total = sum(dist.values())
        print(f"{model_name}: STRONG_MATCH={dist['STRONG_MATCH']} ({dist['STRONG_MATCH']/total*100:.1f}%), "
              f"POTENTIAL_MATCH={dist['POTENTIAL_MATCH']} ({dist['POTENTIAL_MATCH']/total*100:.1f}%), "
              f"NOT_MATCH={dist['NOT_MATCH']} ({dist['NOT_MATCH']/total*100:.1f}%)")
    
    # Extract common classifications
    common_classifications = extract_common_classifications(model1_data, model2_data, model3_data)
    
    # Flatten classifications for analysis
    all_common_classifications = []
    classification_pairs = {model_names[0]: [], model_names[1]: [], model_names[2]: []}
    
    for job_id, profiles in common_classifications.items():
        for profile_id, classifications in profiles.items():
            if len(classifications) == 3:  # All three models
                all_common_classifications.append(classifications)
                classification_pairs[model_names[0]].append(classifications[0])
                classification_pairs[model_names[1]].append(classifications[1])
                classification_pairs[model_names[2]].append(classifications[2])
    
    n_common = len(all_common_classifications)
    print(f"\nCommon Classifications: {n_common} profile-job pairs classified by all three models")
    
    if n_common == 0:
        print("No common classifications found between all three models.")
        return
    
    # Pairwise agreement analysis
    print("\n=== Pairwise Agreement Analysis ===")
    
    pairs = [
        (f"{model_names[0]} vs {model_names[1]}", classification_pairs[model_names[0]], classification_pairs[model_names[1]]),
        (f"{model_names[0]} vs {model_names[2]}", classification_pairs[model_names[0]], classification_pairs[model_names[2]]),
        (f"{model_names[1]} vs {model_names[2]}", classification_pairs[model_names[1]], classification_pairs[model_names[2]])
    ]
    
    for pair_name, cls1, cls2 in pairs:
        agreement, kappa = calculate_pairwise_agreement(cls1, cls2)
        print(f"{pair_name}: Agreement={agreement:.3f}, Cohen's κ={kappa:.3f}")
    
    # Overall agreement (Fleiss' kappa)
    fleiss_kappa = calculate_fleiss_kappa(all_common_classifications)
    print(f"\nOverall Agreement (Fleiss' κ): {fleiss_kappa:.3f}")
    
    # Kappa interpretation
    def interpret_kappa(kappa):
        if kappa < 0: return "Poor"
        elif kappa < 0.20: return "Slight"
        elif kappa < 0.40: return "Fair"
        elif kappa < 0.60: return "Moderate"
        elif kappa < 0.80: return "Substantial"
        else: return "Almost Perfect"
    
    print(f"Interpretation: {interpret_kappa(fleiss_kappa)}")
    
    # Confusion analysis
    print("\n=== Disagreement Analysis ===")
    
    # Count exact matches
    exact_matches = sum(1 for cls in all_common_classifications if len(set(cls)) == 1)
    partial_matches = sum(1 for cls in all_common_classifications if len(set(cls)) == 2)
    complete_disagreements = sum(1 for cls in all_common_classifications if len(set(cls)) == 3)
    
    print(f"Exact agreement (all 3 agree): {exact_matches}/{n_common} ({exact_matches/n_common*100:.1f}%)")
    print(f"Partial agreement (2 agree): {partial_matches}/{n_common} ({partial_matches/n_common*100:.1f}%)")
    print(f"Complete disagreement: {complete_disagreements}/{n_common} ({complete_disagreements/n_common*100:.1f}%)")
    
    # Category-specific analysis
    print("\n=== Category-specific Disagreements ===")
    
    category_disagreements = defaultdict(int)
    for cls in all_common_classifications:
        if len(set(cls)) > 1:  # There's disagreement
            sorted_cls = sorted(cls)
            disagreement_type = " vs ".join(sorted_cls)
            category_disagreements[disagreement_type] += 1
    
    for disagreement_type, count in category_disagreements.items():
        print(f"{disagreement_type}: {count} cases")
    
    # Detailed examples of disagreements
    print("\n=== Sample Disagreements ===")
    disagreement_examples = []
    
    for job_id, profiles in common_classifications.items():
        for profile_id, classifications in profiles.items():
            if len(set(classifications)) > 1:  # Disagreement exists
                job_title = model1_data['classifications'][job_id]['job_info']['job_title']
                disagreement_examples.append({
                    'job_id': job_id,
                    'job_title': job_title,
                    'profile_id': profile_id,
                    model_names[0]: classifications[0],
                    model_names[1]: classifications[1],
                    model_names[2]: classifications[2]
                })
    
    # Show first 5 disagreement examples
    for i, example in enumerate(disagreement_examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"Job: {example['job_title'][:50]}...")
        print(f"Profile ID: {example['profile_id']}")
        print(f"{model_names[0]}: {example[model_names[0]]}, {model_names[1]}: {example[model_names[1]]}, {model_names[2]}: {example[model_names[2]]}")
    
    return {
        'n_common_classifications': n_common,
        'fleiss_kappa': fleiss_kappa,
        'exact_agreement_rate': exact_matches / n_common if n_common > 0 else 0,
        'category_disagreements': dict(category_disagreements)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate classification consistency between LLM models')
    parser.add_argument('--deepseek-file', default='results/results_d_job_matching_reverse_150_m_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_baseline_normal.json',
                        help='Path to DeepSeek results file')
    parser.add_argument('--gemini-file', default='results/results_d_job_matching_reverse_150_m_google_gemini-2.5-flash_baseline_normal.json',
                        help='Path to Gemini results file')
    parser.add_argument('--qwen-file', default='results/results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json',
                        help='Path to Qwen results file')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.deepseek_file, args.gemini_file, args.qwen_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    model_names = ['DeepSeek', 'Gemini', 'Qwen']
    
    results = analyze_model_consistency(
        args.deepseek_file, 
        args.gemini_file, 
        args.qwen_file, 
        model_names
    )
    
    print(f"\n=== Summary ===")
    print(f"Models compared: {', '.join(model_names)}")
    print(f"Common classifications: {results['n_common_classifications']}")
    print(f"Overall agreement (Fleiss' κ): {results['fleiss_kappa']:.3f}")
    print(f"Exact agreement rate: {results['exact_agreement_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
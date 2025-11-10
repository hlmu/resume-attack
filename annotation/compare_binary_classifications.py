#!/usr/bin/env python3
"""
Compare binary classification consistency (NOT_MATCH vs MATCH) between model results and human annotations.
"""

import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_to_binary(classification: str) -> str:
    """Convert multi-class classification to binary (NOT_MATCH vs MATCH)."""
    if classification == "NOT_MATCH":
        return "NOT_MATCH"
    elif classification in ["STRONG_MATCH", "POTENTIAL_MATCH"]:
        return "MATCH"
    else:  # NOT_DECIDED or other
        return "NOT_DECIDED"

def extract_binary_classifications(data: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """Extract (job_id, profile_id) -> binary classification mapping from data."""
    classifications = {}
    
    for job_id, job_data in data["classifications"].items():
        for applicant in job_data["applicants"]:
            profile_id = applicant["profile_id"]
            original_classification = applicant["classification"]
            binary_classification = convert_to_binary(original_classification)
            classifications[(job_id, profile_id)] = binary_classification
    
    return classifications

def calculate_binary_metrics(model_classifications: Dict[Tuple[str, str], str], 
                           human_classifications: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    """Calculate binary classification metrics between model and human classifications."""
    
    # Find common (job_id, profile_id) pairs
    common_pairs = set(model_classifications.keys()) & set(human_classifications.keys())
    
    if not common_pairs:
        return {"error": "No common job-profile pairs found"}
    
    # Filter out NOT_DECIDED cases for binary analysis
    valid_pairs = []
    for pair in common_pairs:
        model_class = model_classifications[pair]
        human_class = human_classifications[pair]
        if model_class != "NOT_DECIDED" and human_class != "NOT_DECIDED":
            valid_pairs.append(pair)
    
    if not valid_pairs:
        return {"error": "No valid pairs for binary comparison (all are NOT_DECIDED)"}
    
    # Calculate binary confusion matrix
    true_positives = 0   # Human: MATCH, Model: MATCH
    true_negatives = 0   # Human: NOT_MATCH, Model: NOT_MATCH
    false_positives = 0  # Human: NOT_MATCH, Model: MATCH
    false_negatives = 0  # Human: MATCH, Model: NOT_MATCH
    
    detailed_comparisons = []
    
    for pair in valid_pairs:
        model_class = model_classifications[pair]
        human_class = human_classifications[pair]
        
        if human_class == "MATCH" and model_class == "MATCH":
            true_positives += 1
        elif human_class == "NOT_MATCH" and model_class == "NOT_MATCH":
            true_negatives += 1
        elif human_class == "NOT_MATCH" and model_class == "MATCH":
            false_positives += 1
        elif human_class == "MATCH" and model_class == "NOT_MATCH":
            false_negatives += 1
        
        detailed_comparisons.append({
            "job_id": pair[0],
            "profile_id": pair[1],
            "human_binary": human_class,
            "model_binary": model_class,
            "agreement": model_class == human_class
        })
    
    total_valid = len(valid_pairs)
    accuracy = (true_positives + true_negatives) / total_valid if total_valid > 0 else 0
    
    # Calculate precision, recall, and F1-score for MATCH class
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate specificity for NOT_MATCH class
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return {
        "total_comparisons": len(common_pairs),
        "valid_comparisons": total_valid,
        "not_decided_cases": len(common_pairs) - total_valid,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "detailed_comparisons": detailed_comparisons
    }

def print_binary_report(metrics: Dict[str, Any], model_file: str, human_file: str):
    """Print a detailed binary comparison report."""
    print("=" * 80)
    print("BINARY CLASSIFICATION CONSISTENCY COMPARISON REPORT")
    print("(NOT_MATCH vs MATCH)")
    print("=" * 80)
    print(f"Model results file: {model_file}")
    print(f"Human annotations file: {human_file}")
    print()
    
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return
    
    # Overall metrics
    print("OVERALL METRICS:")
    print(f"Total comparisons: {metrics['total_comparisons']}")
    print(f"Valid comparisons (excluding NOT_DECIDED): {metrics['valid_comparisons']}")
    print(f"NOT_DECIDED cases: {metrics['not_decided_cases']}")
    print(f"Binary accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print()
    
    # Confusion matrix
    print("BINARY CONFUSION MATRIX:")
    print("Rows: Human annotations, Columns: Model predictions")
    header_label = "Human\\Model"
    print(f"{header_label:<12} {'NOT_MATCH':<12} {'MATCH':<12} {'Total':<12}")
    print("-" * 50)
    
    human_not_match_total = metrics['true_negatives'] + metrics['false_positives']
    human_match_total = metrics['true_positives'] + metrics['false_negatives']
    
    print(f"{'NOT_MATCH':<12} {metrics['true_negatives']:<12} {metrics['false_positives']:<12} {human_not_match_total:<12}")
    print(f"{'MATCH':<12} {metrics['false_negatives']:<12} {metrics['true_positives']:<12} {human_match_total:<12}")
    
    model_not_match_total = metrics['true_negatives'] + metrics['false_negatives']
    model_match_total = metrics['true_positives'] + metrics['false_positives']
    print(f"{'Total':<12} {model_not_match_total:<12} {model_match_total:<12} {metrics['valid_comparisons']:<12}")
    print()
    
    # Performance metrics
    print("PERFORMANCE METRICS:")
    print(f"Precision (MATCH): {metrics['precision']:.3f}")
    print(f"Recall (MATCH): {metrics['recall']:.3f}")
    print(f"F1-Score (MATCH): {metrics['f1_score']:.3f}")
    print(f"Specificity (NOT_MATCH): {metrics['specificity']:.3f}")
    print()
    
    # Error analysis
    agreements = sum(1 for comp in metrics['detailed_comparisons'] if comp['agreement'])
    disagreements = len(metrics['detailed_comparisons']) - agreements
    
    print("ERROR ANALYSIS:")
    print(f"Agreements: {agreements}")
    print(f"Disagreements: {disagreements}")
    
    if disagreements > 0:
        false_positive_cases = [comp for comp in metrics['detailed_comparisons'] 
                               if comp['human_binary'] == 'NOT_MATCH' and comp['model_binary'] == 'MATCH']
        false_negative_cases = [comp for comp in metrics['detailed_comparisons'] 
                               if comp['human_binary'] == 'MATCH' and comp['model_binary'] == 'NOT_MATCH']
        
        print(f"False Positives (Model says MATCH, Human says NOT_MATCH): {len(false_positive_cases)}")
        print(f"False Negatives (Model says NOT_MATCH, Human says MATCH): {len(false_negative_cases)}")
        
        if len(false_positive_cases) > len(false_negative_cases):
            print("-> Model tends to be more optimistic (over-predicts matches)")
        elif len(false_negative_cases) > len(false_positive_cases):
            print("-> Model tends to be more conservative (under-predicts matches)")
        else:
            print("-> Model shows balanced error patterns")
    print()

def main():
    model_file = "../results/job_candidate_classification_Qwen_Qwen3-8B_normal.json"
    human_file = "merged_annotations.json"
    
    print("Loading classification files...")
    model_data = load_json_file(model_file)
    human_data = load_json_file(human_file)
    
    print("Converting to binary classifications...")
    model_binary = extract_binary_classifications(model_data)
    human_binary = extract_binary_classifications(human_data)
    
    print(f"Model binary classifications: {len(model_binary)} entries")
    print(f"Human binary classifications: {len(human_binary)} entries")
    
    print("Calculating binary metrics...")
    metrics = calculate_binary_metrics(model_binary, human_binary)
    
    print_binary_report(metrics, model_file, human_file)
    
    # Save detailed results
    output_file = "binary_classification_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed binary results saved to: {output_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compare classification consistency between model results and human annotations.
"""

import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_classifications(data: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """Extract (job_id, profile_id) -> classification mapping from data."""
    classifications = {}
    
    for job_id, job_data in data["classifications"].items():
        for applicant in job_data["applicants"]:
            profile_id = applicant["profile_id"]
            classification = applicant["classification"]
            classifications[(job_id, profile_id)] = classification
    
    return classifications

def calculate_agreement_metrics(model_classifications: Dict[Tuple[str, str], str], 
                              human_classifications: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    """Calculate various agreement metrics between model and human classifications."""
    
    # Find common (job_id, profile_id) pairs
    common_pairs = set(model_classifications.keys()) & set(human_classifications.keys())
    
    if not common_pairs:
        return {"error": "No common job-profile pairs found"}
    
    # Count agreements and disagreements
    agreements = 0
    disagreements = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    detailed_comparisons = []
    
    for pair in common_pairs:
        model_class = model_classifications[pair]
        human_class = human_classifications[pair]
        
        if model_class == human_class:
            agreements += 1
        else:
            disagreements += 1
        
        confusion_matrix[human_class][model_class] += 1
        detailed_comparisons.append({
            "job_id": pair[0],
            "profile_id": pair[1],
            "human_classification": human_class,
            "model_classification": model_class,
            "agreement": model_class == human_class
        })
    
    total_comparisons = len(common_pairs)
    overall_accuracy = agreements / total_comparisons if total_comparisons > 0 else 0
    
    # Calculate per-class metrics
    class_metrics = {}
    all_classes = set(model_classifications.values()) | set(human_classifications.values())
    
    for cls in all_classes:
        true_positives = confusion_matrix[cls][cls]
        false_positives = sum(confusion_matrix[other_cls][cls] for other_cls in all_classes if other_cls != cls)
        false_negatives = sum(confusion_matrix[cls][other_cls] for other_cls in all_classes if other_cls != cls)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": sum(confusion_matrix[cls].values())
        }
    
    return {
        "total_comparisons": total_comparisons,
        "agreements": agreements,
        "disagreements": disagreements,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": dict(confusion_matrix),
        "class_metrics": class_metrics,
        "detailed_comparisons": detailed_comparisons
    }

def print_comparison_report(metrics: Dict[str, Any], model_file: str, human_file: str):
    """Print a detailed comparison report."""
    print("=" * 80)
    print("CLASSIFICATION CONSISTENCY COMPARISON REPORT")
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
    print(f"Agreements: {metrics['agreements']}")
    print(f"Disagreements: {metrics['disagreements']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)")
    print()
    
    # Confusion matrix
    print("CONFUSION MATRIX (Human vs Model):")
    print("Rows: Human annotations, Columns: Model predictions")
    confusion = metrics['confusion_matrix']
    all_classes = sorted(set().union(*[conf.keys() for conf in confusion.values()]) | set(confusion.keys()))
    
    # Header
    header_label = "Human\\Model"
    print(f"{header_label:<15}", end="")
    for cls in all_classes:
        print(f"{cls:<15}", end="")
    print("Total")
    
    # Matrix rows
    for human_cls in all_classes:
        print(f"{human_cls:<15}", end="")
        row_total = 0
        for model_cls in all_classes:
            count = confusion.get(human_cls, {}).get(model_cls, 0)
            print(f"{count:<15}", end="")
            row_total += count
        print(f"{row_total}")
    print()
    
    # Per-class metrics
    print("PER-CLASS METRICS:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    
    for cls, metrics_data in metrics['class_metrics'].items():
        print(f"{cls:<15} {metrics_data['precision']:<10.3f} {metrics_data['recall']:<10.3f} "
              f"{metrics_data['f1_score']:<10.3f} {metrics_data['support']:<10}")
    print()
    
    # Disagreement analysis
    disagreements = [comp for comp in metrics['detailed_comparisons'] if not comp['agreement']]
    if disagreements:
        print(f"DISAGREEMENT ANALYSIS ({len(disagreements)} cases):")
        disagreement_patterns = Counter((comp['human_classification'], comp['model_classification']) 
                                       for comp in disagreements)
        
        print("Most common disagreement patterns:")
        for (human_cls, model_cls), count in disagreement_patterns.most_common(10):
            print(f"  Human: {human_cls} -> Model: {model_cls} ({count} cases)")
        print()

def main():
    model_file = "../results/job_candidate_classification_Qwen_Qwen3-8B_normal.json"
    human_file = "merged_annotations.json"
    
    print("Loading classification files...")
    model_data = load_json_file(model_file)
    human_data = load_json_file(human_file)
    
    print("Extracting classifications...")
    model_classifications = extract_classifications(model_data)
    human_classifications = extract_classifications(human_data)
    
    print(f"Model classifications: {len(model_classifications)} entries")
    print(f"Human classifications: {len(human_classifications)} entries")
    
    print("Calculating agreement metrics...")
    metrics = calculate_agreement_metrics(model_classifications, human_classifications)
    
    print_comparison_report(metrics, model_file, human_file)
    
    # Save detailed results
    output_file = "classification_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
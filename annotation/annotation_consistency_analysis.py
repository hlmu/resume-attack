#!/usr/bin/env python3
"""
Annotation Consistency Evaluation Script

Evaluates inter-annotator agreement for job-candidate classification task
across three annotators: jinghao, rui, and kaiyang.
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def load_annotation_data(file_path: str) -> Dict[str, Any]:
    """Load annotation data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_common_annotations(data1: Dict, data2: Dict, data3: Dict) -> Dict[str, Dict[str, List[str]]]:
    """Extract annotations for profile_ids that appear in all three annotators' data"""
    common_annotations = defaultdict(lambda: defaultdict(list))
    
    # Get all job_ids and profile_ids from each annotator
    jobs1 = set(data1['classifications'].keys())
    jobs2 = set(data2['classifications'].keys())
    jobs3 = set(data3['classifications'].keys())
    
    common_jobs = jobs1.intersection(jobs2).intersection(jobs3)
    
    for job_id in common_jobs:
        profiles1 = {app['profile_id']: app['classification'] for app in data1['classifications'][job_id]['applicants']}
        profiles2 = {app['profile_id']: app['classification'] for app in data2['classifications'][job_id]['applicants']}
        profiles3 = {app['profile_id']: app['classification'] for app in data3['classifications'][job_id]['applicants']}
        
        common_profiles = set(profiles1.keys()).intersection(set(profiles2.keys())).intersection(set(profiles3.keys()))
        
        for profile_id in common_profiles:
            common_annotations[job_id][profile_id] = [
                profiles1[profile_id],
                profiles2[profile_id], 
                profiles3[profile_id]
            ]
    
    return common_annotations

def calculate_pairwise_agreement(annotations1: List[str], annotations2: List[str]) -> Tuple[float, float]:
    """Calculate pairwise agreement and Cohen's kappa between two annotators"""
    if len(annotations1) != len(annotations2):
        return 0.0, 0.0
    
    if len(annotations1) == 0:
        return 0.0, 0.0
    
    # Simple agreement
    agreement = sum(1 for a1, a2 in zip(annotations1, annotations2) if a1 == a2) / len(annotations1)
    
    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(annotations1, annotations2)
    except:
        kappa = 0.0
    
    return agreement, kappa

def calculate_fleiss_kappa(annotations_matrix: List[List[str]]) -> float:
    """Calculate Fleiss' kappa for multiple annotators"""
    if not annotations_matrix or len(annotations_matrix[0]) < 2:
        return 0.0
    
    n_items = len(annotations_matrix)
    n_raters = len(annotations_matrix[0])
    
    # Get all unique categories
    all_categories = set()
    for item_annotations in annotations_matrix:
        all_categories.update(item_annotations)
    categories = sorted(list(all_categories))
    
    if len(categories) <= 1:
        return 1.0 if len(categories) == 1 else 0.0
    
    # Create rating matrix
    rating_matrix = np.zeros((n_items, len(categories)))
    for i, item_annotations in enumerate(annotations_matrix):
        for annotation in item_annotations:
            j = categories.index(annotation)
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

def analyze_binary_classification(jinghao_data: Dict, rui_data: Dict, kaiyang_data: Dict):
    """Analyze agreement with binary classification (MATCH vs NOT_MATCH)"""
    
    def convert_to_binary(classification: str) -> str:
        """Convert 3-class to binary"""
        return "MATCH" if classification in ["STRONG_MATCH", "POTENTIAL_MATCH"] else "NOT_MATCH"
    
    # Extract common annotations and convert to binary
    common_annotations = extract_common_annotations(jinghao_data, rui_data, kaiyang_data)
    
    all_common_annotations = []
    annotation_pairs = {'jinghao': [], 'rui': [], 'kaiyang': []}
    
    for job_id, profiles in common_annotations.items():
        for profile_id, annotations in profiles.items():
            if len(annotations) == 3:  # All three annotators
                binary_annotations = [convert_to_binary(ann) for ann in annotations]
                all_common_annotations.append(binary_annotations)
                annotation_pairs['jinghao'].append(binary_annotations[0])
                annotation_pairs['rui'].append(binary_annotations[1])
                annotation_pairs['kaiyang'].append(binary_annotations[2])
    
    n_common = len(all_common_annotations)
    print(f"Common Binary Annotations: {n_common} profile-job pairs")
    
    if n_common == 0:
        print("No common annotations found.")
        return
    
    # Binary distribution for each annotator
    print("\nBinary Match Distribution:")
    for annotator, annotations in annotation_pairs.items():
        match_count = annotations.count("MATCH")
        not_match_count = annotations.count("NOT_MATCH")
        total = len(annotations)
        print(f"{annotator.capitalize()}: MATCH={match_count} ({match_count/total*100:.1f}%), "
              f"NOT_MATCH={not_match_count} ({not_match_count/total*100:.1f}%)")
    
    # Pairwise agreement analysis
    print("\nPairwise Agreement (Binary):")
    pairs = [
        ("Jinghao vs Rui", annotation_pairs['jinghao'], annotation_pairs['rui']),
        ("Jinghao vs Kaiyang", annotation_pairs['jinghao'], annotation_pairs['kaiyang']),
        ("Rui vs Kaiyang", annotation_pairs['rui'], annotation_pairs['kaiyang'])
    ]
    
    for pair_name, ann1, ann2 in pairs:
        agreement, kappa = calculate_pairwise_agreement(ann1, ann2)
        print(f"{pair_name}: Agreement={agreement:.3f}, Cohen's κ={kappa:.3f}")
    
    # Overall agreement (Fleiss' kappa)
    fleiss_kappa = calculate_fleiss_kappa(all_common_annotations)
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
    exact_matches = sum(1 for ann in all_common_annotations if len(set(ann)) == 1)
    disagreements = sum(1 for ann in all_common_annotations if len(set(ann)) > 1)
    
    print(f"\nBinary Agreement Breakdown:")
    print(f"Exact agreement (all 3 agree): {exact_matches}/{n_common} ({exact_matches/n_common*100:.1f}%)")
    print(f"Any disagreement: {disagreements}/{n_common} ({disagreements/n_common*100:.1f}%)")
    
    return fleiss_kappa

def analyze_annotation_consistency():
    """Main analysis function"""
    # Load data from all three annotators
    jinghao_data = load_annotation_data('/home/engie/src/align_fun/annotation/job_candidate_classification_human_20250719T181332_jinghao.json')
    rui_data = load_annotation_data('/home/engie/src/align_fun/annotation/job_candidate_classification_human_20250721T132418_rui.json')
    kaiyang_data = load_annotation_data('/home/engie/src/align_fun/annotation/job_candidate_classification_human_20250724T095134_kaiyang.json')
    
    print("=== Annotation Consistency Analysis ===\n")
    
    # Also analyze with binary classification (MATCH vs NOT_MATCH)
    print("=== BINARY CLASSIFICATION ANALYSIS (MATCH vs NOT_MATCH) ===\n")
    analyze_binary_classification(jinghao_data, rui_data, kaiyang_data)
    print("\n" + "="*60 + "\n")
    
    print("=== THREE-CLASS ANALYSIS (STRONG/POTENTIAL/NOT_MATCH) ===\n")
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"Jinghao: {jinghao_data['summary']['total_classifications']} classifications across {jinghao_data['summary']['total_jobs']} jobs")
    print(f"Rui: {rui_data['summary']['total_classifications']} classifications across {rui_data['summary']['total_jobs']} jobs")
    print(f"Kaiyang: {kaiyang_data['summary']['total_classifications']} classifications across {kaiyang_data['summary']['total_jobs']} jobs")
    
    print("\nMatch Distribution:")
    for annotator, data in [("Jinghao", jinghao_data), ("Rui", rui_data), ("Kaiyang", kaiyang_data)]:
        dist = data['summary']['match_distribution']
        total = sum(dist.values())
        print(f"{annotator}: STRONG_MATCH={dist['STRONG_MATCH']} ({dist['STRONG_MATCH']/total*100:.1f}%), "
              f"POTENTIAL_MATCH={dist['POTENTIAL_MATCH']} ({dist['POTENTIAL_MATCH']/total*100:.1f}%), "
              f"NOT_MATCH={dist['NOT_MATCH']} ({dist['NOT_MATCH']/total*100:.1f}%)")
    
    # Extract common annotations
    common_annotations = extract_common_annotations(jinghao_data, rui_data, kaiyang_data)
    
    # Flatten annotations for analysis
    all_common_annotations = []
    annotation_pairs = {'jinghao': [], 'rui': [], 'kaiyang': []}
    
    for job_id, profiles in common_annotations.items():
        for profile_id, annotations in profiles.items():
            if len(annotations) == 3:  # All three annotators
                all_common_annotations.append(annotations)
                annotation_pairs['jinghao'].append(annotations[0])
                annotation_pairs['rui'].append(annotations[1])
                annotation_pairs['kaiyang'].append(annotations[2])
    
    n_common = len(all_common_annotations)
    print(f"\nCommon Annotations: {n_common} profile-job pairs annotated by all three annotators")
    
    if n_common == 0:
        print("No common annotations found between all three annotators.")
        return
    
    # Pairwise agreement analysis
    print("\n=== Pairwise Agreement Analysis ===")
    
    pairs = [
        ("Jinghao vs Rui", annotation_pairs['jinghao'], annotation_pairs['rui']),
        ("Jinghao vs Kaiyang", annotation_pairs['jinghao'], annotation_pairs['kaiyang']),
        ("Rui vs Kaiyang", annotation_pairs['rui'], annotation_pairs['kaiyang'])
    ]
    
    for pair_name, ann1, ann2 in pairs:
        agreement, kappa = calculate_pairwise_agreement(ann1, ann2)
        print(f"{pair_name}: Agreement={agreement:.3f}, Cohen's κ={kappa:.3f}")
    
    # Overall agreement (Fleiss' kappa)
    fleiss_kappa = calculate_fleiss_kappa(all_common_annotations)
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
    exact_matches = sum(1 for ann in all_common_annotations if len(set(ann)) == 1)
    partial_matches = sum(1 for ann in all_common_annotations if len(set(ann)) == 2)
    complete_disagreements = sum(1 for ann in all_common_annotations if len(set(ann)) == 3)
    
    print(f"Exact agreement (all 3 agree): {exact_matches}/{n_common} ({exact_matches/n_common*100:.1f}%)")
    print(f"Partial agreement (2 agree): {partial_matches}/{n_common} ({partial_matches/n_common*100:.1f}%)")
    print(f"Complete disagreement: {complete_disagreements}/{n_common} ({complete_disagreements/n_common*100:.1f}%)")
    
    # Category-specific analysis
    print("\n=== Category-specific Disagreements ===")
    
    category_disagreements = defaultdict(int)
    for ann in all_common_annotations:
        if len(set(ann)) > 1:  # There's disagreement
            sorted_ann = sorted(ann)
            disagreement_type = " vs ".join(sorted_ann)
            category_disagreements[disagreement_type] += 1
    
    for disagreement_type, count in category_disagreements.items():
        print(f"{disagreement_type}: {count} cases")
    
    # Detailed examples of disagreements
    print("\n=== Sample Disagreements ===")
    disagreement_examples = []
    
    for job_id, profiles in common_annotations.items():
        for profile_id, annotations in profiles.items():
            if len(set(annotations)) > 1:  # Disagreement exists
                job_title = jinghao_data['classifications'][job_id]['job_info']['job_title']
                disagreement_examples.append({
                    'job_id': job_id,
                    'job_title': job_title,
                    'profile_id': profile_id,
                    'jinghao': annotations[0],
                    'rui': annotations[1],
                    'kaiyang': annotations[2]
                })
    
    # Show first 5 disagreement examples
    for i, example in enumerate(disagreement_examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"Job: {example['job_title'][:50]}...")
        print(f"Profile ID: {example['profile_id']}")
        print(f"Jinghao: {example['jinghao']}, Rui: {example['rui']}, Kaiyang: {example['kaiyang']}")
    
    return {
        'n_common_annotations': n_common,
        'fleiss_kappa': fleiss_kappa,
        'exact_agreement_rate': exact_matches / n_common if n_common > 0 else 0,
        'category_disagreements': dict(category_disagreements)
    }

if __name__ == "__main__":
    results = analyze_annotation_consistency()
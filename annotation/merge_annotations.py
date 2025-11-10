#!/usr/bin/env python3
"""
Merge annotation results using majority voting.
For non-majority agreements, set classification to "NOT_DECIDED".
"""

import json
import glob
from collections import Counter
from typing import Dict, List, Any

def load_annotation_files(annotation_dir: str) -> List[Dict[str, Any]]:
    """Load all annotation JSON files from the directory."""
    files = glob.glob(f"{annotation_dir}/*.json")
    annotations = []
    
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            annotations.append(data)
    
    return annotations

def merge_annotations(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge annotations using majority voting."""
    if not annotations:
        raise ValueError("No annotations provided")
    
    # Use the first annotation as template structure
    merged = json.loads(json.dumps(annotations[0]))  # Deep copy
    
    # Reset summary stats - we'll recalculate
    merged["summary"]["match_distribution"] = {"STRONG_MATCH": 0, "POTENTIAL_MATCH": 0, "NOT_MATCH": 0, "NOT_DECIDED": 0}
    
    # Process each job
    for job_id in merged["classifications"]:
        job_data = merged["classifications"][job_id]
        
        # Process each applicant
        for i, applicant in enumerate(job_data["applicants"]):
            profile_id = applicant["profile_id"]
            
            # Collect classifications for this applicant from all annotators
            classifications = []
            for annotation in annotations:
                if job_id in annotation["classifications"]:
                    job_applicants = annotation["classifications"][job_id]["applicants"]
                    # Find matching applicant by profile_id
                    for app in job_applicants:
                        if app["profile_id"] == profile_id:
                            classifications.append(app["classification"])
                            break
            
            # Determine majority vote
            if not classifications:
                final_classification = "NOT_DECIDED"
            else:
                vote_counts = Counter(classifications)
                most_common = vote_counts.most_common()
                
                # Check if there's a clear majority
                if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                    final_classification = most_common[0][0]
                else:
                    final_classification = "NOT_DECIDED"
            
            # Update the merged result
            merged["classifications"][job_id]["applicants"][i]["classification"] = final_classification
            merged["classifications"][job_id]["applicants"][i]["response_content"] = f"Merged result: {final_classification}"
            
            # Update summary statistics
            merged["summary"]["match_distribution"][final_classification] += 1
    
    # Update total classifications count
    total_classifications = sum(merged["summary"]["match_distribution"].values())
    merged["summary"]["total_classifications"] = total_classifications
    
    return merged

def main():
    annotation_dir = "."
    output_file = "merged_annotations.json"
    
    print(f"Loading annotation files from {annotation_dir}...")
    annotations = load_annotation_files(annotation_dir)
    print(f"Found {len(annotations)} annotation files")
    
    print("Merging annotations using majority voting...")
    merged_result = merge_annotations(annotations)
    
    print(f"Writing merged results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_result, f, indent=2)
    
    # Print summary
    print("\nMerged annotation summary:")
    print(f"Total jobs: {merged_result['summary']['total_jobs']}")
    print(f"Total classifications: {merged_result['summary']['total_classifications']}")
    print("Match distribution:")
    for match_type, count in merged_result['summary']['match_distribution'].items():
        print(f"  {match_type}: {count}")

if __name__ == "__main__":
    main()
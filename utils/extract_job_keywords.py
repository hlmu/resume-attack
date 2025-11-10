#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
import re

def load_job_data(file_path):
    """Load job data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading job data: {e}")
        return None

def extract_keywords_from_jobs(jobs_data):
    """Extract all keywords from job discovery_input fields"""
    keywords = []
    
    for job in jobs_data:
        if isinstance(job, dict):
            discovery_input = job.get('discovery_input', {})
            if isinstance(discovery_input, dict):
                keyword = discovery_input.get('keyword')
                if keyword and isinstance(keyword, str):
                    # Clean and normalize the keyword
                    if " OR " in keyword:
                        for k in keyword.split(" OR "):
                            k = k.strip().lower()
                            if k and len(k) > 1:  # Filter out single characters
                                keywords.append(k)
                    else:
                        keyword = keyword.strip().lower()
                        if keyword and len(keyword) > 1:  # Filter out single characters
                            keywords.append(keyword)
    
    return keywords

def clean_and_process_keywords(keywords):
    """Clean and process the extracted keywords"""
    # Count frequency of keywords
    keyword_counts = Counter(keywords)
    
    # Filter out very common but not useful keywords
    stop_words = {'job', 'work', 'position', 'role', 'career', 'opportunity', 
                  'full-time', 'part-time', 'remote', 'onsite', 'hybrid',
                  'entry', 'senior', 'junior', 'manager', 'director', 'lead',
                  'the', 'and', 'or', 'for', 'with', 'in', 'at', 'to', 'of'}
    
    # Clean keywords
    cleaned_keywords = []
    for keyword, count in keyword_counts.items():
        # Skip stop words
        if keyword.lower() in stop_words:
            continue
            
        # Skip very short keywords (less than 2 characters)
        if len(keyword) < 2:
            continue
            
        # Skip purely numeric keywords
        if keyword.isdigit():
            continue
            
        # Clean the keyword
        cleaned_keyword = re.sub(r'[^\w\s\.\+\-]', '', keyword).strip()
        if cleaned_keyword and len(cleaned_keyword) >= 2:
            cleaned_keywords.append((cleaned_keyword, count))
    
    # Sort by frequency (descending)
    cleaned_keywords.sort(key=lambda x: x[1], reverse=True)
    
    return cleaned_keywords

def extract_skills_from_descriptions(jobs_data, max_skills=200):
    """Extract technical skills from job descriptions"""
    # Common technical patterns
    tech_patterns = [
        r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|swift|kotlin|scala)\b',
        r'\b(react|angular|vue\.js|node\.js|django|flask|spring|laravel|rails)\b',
        r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra)\b',
        r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab)\b',
        r'\b(machine learning|ai|data science|deep learning|nlp|computer vision)\b',
        r'\b(agile|scrum|devops|ci/cd|microservices|api|rest|graphql)\b',
        r'\b(html|css|sass|scss|webpack|babel|npm|yarn)\b',
        r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib)\b',
        r'\b(linux|windows|macos|unix|bash|powershell)\b',
        r'\b(tableau|power bi|excel|r|matlab|jupyter|anaconda)\b'
    ]
    
    skills_found = Counter()
    
    for job in jobs_data:
        if isinstance(job, dict):
            # Check job description
            description = job.get('job_description_formatted', '') or job.get('job_summary', '')
            if description:
                description = description.lower()
                
                for pattern in tech_patterns:
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    for match in matches:
                        skills_found[match] += 1
            
            # Check job title
            title = job.get('job_title', '')
            if title:
                title = title.lower()
                for pattern in tech_patterns:
                    matches = re.findall(pattern, title, re.IGNORECASE)
                    for match in matches:
                        skills_found[match] += 1
    
    # Return top skills
    return skills_found.most_common(max_skills)

def save_keywords_to_file(keywords, skills, output_file):
    """Save extracted keywords and skills to a Python file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env python\n')
        f.write('# -*- coding: utf-8 -*-\n\n')
        f.write('# Auto-generated keywords extracted from LinkedIn job data\n\n')
        
        # Write keywords from discovery_input
        f.write('# Keywords extracted from job search terms\n')
        f.write('EXTRACTED_KEYWORDS = [\n')
        # for keyword, count in keywords[:100]:  # Top 100 keywords
        for keyword, count in keywords:
            f.write(f'    "{keyword}",  # appeared {count} times\n')
        f.write(']\n\n')
        
        # Write skills from job descriptions
        f.write('# Technical skills extracted from job descriptions\n')
        f.write('EXTRACTED_SKILLS = [\n')
        for skill, count in skills:
            f.write(f'    "{skill}",  # appeared {count} times\n')
        f.write(']\n\n')
        
        # Write combined list
        combined_list = []
        
        # Add top keywords (excluding very generic ones)
        generic_terms = {'software', 'engineer', 'developer', 'analyst', 'manager', 
                        'specialist', 'coordinator', 'assistant', 'intern', 'consultant'}
        
        # for keyword, _ in keywords[:50]:
        for keyword, _ in keywords:
            if keyword not in generic_terms and len(keyword) > 2:
                combined_list.append(keyword)
        
        # Add all extracted skills
        for skill, _ in skills:
            if skill not in combined_list:
                combined_list.append(skill)
        
        f.write('# Combined list for use in adversarial prompt generation\n')
        f.write('COMMON_SKILLS = [\n')
        for skill in combined_list:
            f.write(f'    "{skill}",\n')
        f.write(']\n')

def main():
    parser = argparse.ArgumentParser(description='Extract keywords from LinkedIn job data')
    parser.add_argument('--input', default='datasets/Linkedin job listings information.json',
                        help='Input JSON file with job data')
    parser.add_argument('--output', default='utils/extracted_keywords.py',
                        help='Output Python file with extracted keywords')
    parser.add_argument('--max-keywords', type=int, default=100,
                        help='Maximum number of keywords to extract')
    parser.add_argument('--max-skills', type=int, default=200,
                        help='Maximum number of skills to extract from descriptions')
    
    args = parser.parse_args()
    
    print(f"Loading job data from {args.input}...")
    jobs_data = load_job_data(args.input)
    
    if not jobs_data:
        print("Failed to load job data")
        return
    
    print(f"Loaded {len(jobs_data)} job entries")
    
    # Extract keywords from discovery_input
    print("Extracting keywords from job search terms...")
    raw_keywords = extract_keywords_from_jobs(jobs_data)
    print(f"Found {len(raw_keywords)} raw keywords")
    
    # Clean and process keywords
    print("Cleaning and processing keywords...")
    cleaned_keywords = clean_and_process_keywords(raw_keywords)
    print(f"Processed to {len(cleaned_keywords)} unique keywords")
    
    # Extract skills from job descriptions
    print("Extracting technical skills from job descriptions...")
    extracted_skills = extract_skills_from_descriptions(jobs_data, args.max_skills)
    print(f"Extracted {len(extracted_skills)} technical skills")
    
    # Save to file
    print(f"Saving results to {args.output}...")
    save_keywords_to_file(cleaned_keywords[:args.max_keywords], extracted_skills, args.output)
    
    # Print summary
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Top 20 keywords by frequency:")
    for i, (keyword, count) in enumerate(cleaned_keywords[:20], 1):
        print(f"  {i:2d}. {keyword:<20} ({count} times)")
    
    print(f"\nTop 20 technical skills:")
    for i, (skill, count) in enumerate(extracted_skills[:20], 1):
        print(f"  {i:2d}. {skill:<20} ({count} times)")
    
    print(f"\nResults saved to: {args.output}")
    print("You can now import COMMON_SKILLS from this file to replace the hardcoded list.")

if __name__ == "__main__":
    main() 
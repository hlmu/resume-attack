import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize job matching results')
    parser.add_argument('--results', default='job_matching_results.json', 
                        help='Path to job matching results JSON file')
    parser.add_argument('--output-dir', default='visualizations', 
                        help='Directory to save visualizations')
    parser.add_argument('--top-n', type=int, default=10, 
                        help='Number of top items to show in bar charts')
    return parser.parse_args()

def load_results(file_path):
    """Load job matching results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_results(results):
    """Analyze job matching results and extract insights"""
    matches = results.get('matches', {})
    summary = results.get('summary', {})
    
    # Prepare data structures for analysis
    all_matches = []
    companies = []
    job_titles = []
    locations = []
    seniority_levels = []
    job_functions = []
    similarity_scores = []
    
    # Extract data from matches
    for profile_name, profile_matches in tqdm(matches.items(), desc="Analyzing matches"):
        for match in profile_matches:
            all_matches.append({
                'profile_name': profile_name,
                'job_title': match.get('job_title'),
                'company_name': match.get('company_name'),
                'job_location': match.get('job_location'),
                'similarity_score': match.get('similarity_score'),
                'job_seniority_level': match.get('job_seniority_level'),
                'job_function': match.get('job_function')
            })
            
            companies.append(match.get('company_name'))
            job_titles.append(match.get('job_title'))
            locations.append(match.get('job_location'))
            seniority_levels.append(match.get('job_seniority_level'))
            job_functions.append(match.get('job_function'))
            similarity_scores.append(match.get('similarity_score'))
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_matches)
    
    # Count frequencies
    company_counts = Counter(companies)
    job_title_counts = Counter(job_titles)
    location_counts = Counter(locations)
    seniority_counts = Counter(seniority_levels)
    function_counts = Counter(job_functions)
    
    # Calculate statistics for similarity scores
    score_stats = {
        'mean': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        'median': sorted(similarity_scores)[len(similarity_scores)//2] if similarity_scores else 0,
        'min': min(similarity_scores) if similarity_scores else 0,
        'max': max(similarity_scores) if similarity_scores else 0,
        'count': len(similarity_scores)
    }
    
    # Prepare results
    analysis = {
        'summary': summary,
        'company_counts': company_counts,
        'job_title_counts': job_title_counts,
        'location_counts': location_counts,
        'seniority_counts': seniority_counts,
        'function_counts': function_counts,
        'score_stats': score_stats,
        'dataframe': df
    }
    
    return analysis

def plot_top_companies(analysis, output_dir, top_n):
    """Plot top companies by frequency"""
    company_counts = analysis['company_counts']
    top_companies = dict(sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_companies.keys(), top_companies.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Companies in Matched Jobs')
    plt.xlabel('Company')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_companies.png')
    plt.close()

def plot_top_job_titles(analysis, output_dir, top_n):
    """Plot top job titles by frequency"""
    job_title_counts = analysis['job_title_counts']
    top_titles = dict(sorted(job_title_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_titles.keys(), top_titles.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Job Titles in Matched Jobs')
    plt.xlabel('Job Title')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_job_titles.png')
    plt.close()

def plot_top_locations(analysis, output_dir, top_n):
    """Plot top locations by frequency"""
    location_counts = analysis['location_counts']
    top_locations = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_locations.keys(), top_locations.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Locations in Matched Jobs')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_locations.png')
    plt.close()

def plot_seniority_distribution(analysis, output_dir):
    """Plot seniority level distribution"""
    seniority_counts = analysis['seniority_counts']
    
    plt.figure(figsize=(10, 6))
    plt.bar(seniority_counts.keys(), seniority_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Seniority Levels in Matched Jobs')
    plt.xlabel('Seniority Level')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/seniority_distribution.png')
    plt.close()

def plot_function_distribution(analysis, output_dir, top_n):
    """Plot job function distribution"""
    function_counts = analysis['function_counts']
    top_functions = dict(sorted(function_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_functions.keys(), top_functions.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Job Functions in Matched Jobs')
    plt.xlabel('Job Function')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_functions.png')
    plt.close()

def plot_similarity_distribution(analysis, output_dir):
    """Plot distribution of similarity scores"""
    df = analysis['dataframe']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['similarity_score'], kde=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_distribution.png')
    plt.close()

def plot_similarity_by_seniority(analysis, output_dir):
    """Plot similarity scores by seniority level"""
    df = analysis['dataframe']
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='job_seniority_level', y='similarity_score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Similarity Scores by Seniority Level')
    plt.xlabel('Seniority Level')
    plt.ylabel('Similarity Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_by_seniority.png')
    plt.close()

def generate_statistics_report(analysis, output_dir):
    """Generate a text report with statistics"""
    summary = analysis['summary']
    score_stats = analysis['score_stats']
    
    report = [
        "# Job Matching Results - Statistics Report",
        "",
        f"## Summary",
        f"- Total profiles processed: {summary.get('total_profiles_processed', 0)}",
        f"- Total jobs processed: {summary.get('total_jobs_processed', 0)}",
        f"- Profiles with matches: {summary.get('profiles_with_matches', 0)}",
        f"- Average matches per profile: {summary.get('average_matches_per_profile', 0):.2f}",
        f"- Top N requested: {summary.get('top_n_requested', 0)}",
        "",
        f"## Similarity Score Statistics",
        f"- Mean similarity score: {score_stats['mean']:.4f}",
        f"- Median similarity score: {score_stats['median']:.4f}",
        f"- Minimum similarity score: {score_stats['min']:.4f}",
        f"- Maximum similarity score: {score_stats['max']:.4f}",
        f"- Total matches: {score_stats['count']}",
        "",
        f"## Top 10 Companies",
        *[f"- {company}: {count}" for company, count in 
          sorted(analysis['company_counts'].items(), key=lambda x: x[1], reverse=True)[:10]],
        "",
        f"## Top 10 Job Titles",
        *[f"- {title}: {count}" for title, count in 
          sorted(analysis['job_title_counts'].items(), key=lambda x: x[1], reverse=True)[:10]],
        "",
        f"## Top 10 Locations",
        *[f"- {location}: {count}" for location, count in 
          sorted(analysis['location_counts'].items(), key=lambda x: x[1], reverse=True)[:10]],
        "",
        f"## Seniority Levels",
        *[f"- {level}: {count}" for level, count in 
          sorted(analysis['seniority_counts'].items(), key=lambda x: x[1], reverse=True)],
        "",
        f"## Top 10 Job Functions",
        *[f"- {function}: {count}" for function, count in 
          sorted(analysis['function_counts'].items(), key=lambda x: x[1], reverse=True)[:10]]
    ]
    
    with open(f'{output_dir}/statistics_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    args = parse_args()
    
    # Ensure output directory exists
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    print("Analyzing results...")
    analysis = analyze_results(results)
    
    print("Generating visualizations...")
    plot_top_companies(analysis, args.output_dir, args.top_n)
    plot_top_job_titles(analysis, args.output_dir, args.top_n)
    plot_top_locations(analysis, args.output_dir, args.top_n)
    plot_seniority_distribution(analysis, args.output_dir)
    plot_function_distribution(analysis, args.output_dir, args.top_n)
    plot_similarity_distribution(analysis, args.output_dir)
    plot_similarity_by_seniority(analysis, args.output_dir)
    
    print("Generating statistics report...")
    generate_statistics_report(analysis, args.output_dir)
    
    print(f"Visualization completed! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 
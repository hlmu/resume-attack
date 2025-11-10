import json
import os
import argparse
from collections import defaultdict, Counter
from utils.utils import load_data
import random
random.seed(42)
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate reverse job matching (job to profiles)')
    parser.add_argument('--input', default='results/job_matching_results.json',
                        help='Path to the original job matching results file')
    parser.add_argument('--profiles', default="datasets/LinkedIn people profiles_verified_company.json", 
                        help='Path to LinkedIn profiles JSON file')
    parser.add_argument('--jobs', default="datasets/Linkedin job listings information.json", 
                        help='Path to job listings JSON file')
    parser.add_argument('--output', default='results/job_matching_reverse.json',
                        help='Output file for reverse matching results')
    parser.add_argument('--analysis', default='results/job_matching_analysis.json',
                        help='Output file for job matching analysis')
    parser.add_argument('--min-score', type=float, default=0.5,
                        help='Minimum similarity score to include in results')
    parser.add_argument('--max-applicants', type=int, default=None,
                        help='Maximum number of applicants to include per job (sorted by score)')
    parser.add_argument('--max-matches', type=int, default=None,
                        help='Maximum number of matches to include per job (sorted by score)')
    return parser.parse_args()

def get_profile_info(profile_id, profiles_data):
    """Get profile information for a given profile ID"""
    for profile in profiles_data:
        if str(profile.get('linkedin_num_id', '')) == str(profile_id):
            return {
                'name': profile.get('name', ''),
                'position': profile.get('position', ''),
                # 'location': profile.get('location', ''),
                # 'experience_years': get_experience_years(profile),
                'skills': get_skills(profile),
                'education': get_education_summary(profile),
                'linkedin_num_id': profile_id
            }
    return {'linkedin_num_id': profile_id, 'name': 'Unknown', 'position': 'Unknown'}

def get_job_info(job_id, jobs_data):
    """Get job information for a given job ID"""
    for job in jobs_data:
        if str(job.get('job_posting_id', '')) == str(job_id):
            return {
                'job_title': job.get('job_title', ''),
                'company_name': job.get('company_name', ''),
                'job_location': job.get('job_location', ''),
                'job_seniority_level': job.get('job_seniority_level', ''),
                'job_function': job.get('job_function', ''),
                'job_industries': job.get('job_industries', ''),
                'job_id': job_id
            }
    return {'job_id': job_id, 'job_title': 'Unknown', 'company_name': 'Unknown'}

def get_experience_years(profile):
    """Estimate total years of experience from profile"""
    if 'experience' not in profile:
        return 0
    
    total_years = 0
    for exp in profile.get('experience', []) or []:
        # Try to extract years from duration or description
        duration = exp.get('subtitle', '')
        if 'years' in duration.lower():
            try:
                # Extract numeric value before "years"
                years = duration.lower().split('years')[0].strip()
                years = ''.join(filter(lambda x: x.isdigit() or x == '.', years))
                if years:
                    total_years += float(years)
            except:
                pass
    
    return total_years

def get_skills(profile):
    """Extract skills from profile"""
    skills = []
    
    # Get skills from certifications as a proxy
    for cert in profile.get('certifications', []) or []:
        skills.append(cert.get('title', ''))
    
    # Get skills from experience descriptions
    for exp in profile.get('experience', []) or []:
        desc = exp.get('description', '').lower()
        # Look for common tech skills (simplified approach)
        tech_keywords = ['python', 'javascript', 'java', 'c++', 'react', 'angular', 'vue', 
                         'node.js', 'sql', 'nosql', 'mongodb', 'aws', 'azure', 'gcp',
                         'machine learning', 'data science', 'ai', 'docker', 'kubernetes',
                         'devops', 'product management', 'agile', 'scrum']
        
        for keyword in tech_keywords:
            if keyword in desc and keyword not in skills:
                skills.append(keyword)
    
    return skills

def get_education_summary(profile):
    """Get summary of education"""
    if 'education' not in profile:
        return "Not specified"
    
    highest_degree = ""
    for edu in profile.get('education', []) or []:
        degree = edu.get('degree', '')
        if 'phd' in degree.lower() or 'doctorate' in degree.lower():
            highest_degree = "PhD"
            break
        elif 'master' in degree.lower() and highest_degree != "PhD":
            highest_degree = "Master's"
        elif ('bachelor' in degree.lower() or 'bs' in degree.lower() or 'ba' in degree.lower()) and highest_degree not in ["PhD", "Master's"]:
            highest_degree = "Bachelor's"
    
    return highest_degree if highest_degree else "Not specified"

def generate_reverse_matching(match_data, profiles_data, jobs_data, min_score=0.5, max_applicants=None, max_matches=None):
    """Generate reverse matching (job_id to profiles) from profile to jobs matching"""
    if not match_data or 'matches' not in match_data:
        print("Error: Invalid data format")
        return None
    
    # Initialize reverse matching structure
    reverse_matches = defaultdict(list)
    
    # Process each profile and its matching jobs
    for profile_id, matching_jobs in match_data['matches'].items():
        for job in matching_jobs:
            # Check if similarity score meets minimum threshold
            if job['similarity_score'] >= min_score:
                job_id = job['job_posting_id']
                
                # Get profile information
                profile_info = get_profile_info(profile_id, profiles_data)
                
                # Add profile to the job's applicants list with additional info
                reverse_matches[job_id].append({
                    'profile': profile_info,
                    'similarity_score': job['similarity_score']
                })
    
    if max_matches:
        tmp = [(job_id, applicants) for job_id, applicants in reverse_matches.items()]
        tmp = random.sample(tmp, max_matches)
        reverse_matches = {job_id: applicants for job_id, applicants in tmp}
    
    # Sort applicants for each job by similarity score (descending)
    # and limit number of applicants if specified
    result_matches = {}
    for job_id, applicants in reverse_matches.items():
        # Sort by similarity score
        applicants.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit number of applicants if specified
        if max_applicants:
            applicants = applicants[:max_applicants]
        
        # Get job information
        job_info = get_job_info(job_id, jobs_data)
        
        # Add job and its applicants to result
        result_matches[job_id] = {
            'job_info': job_info,
            'applicants': applicants,
            'total_applicants': len(applicants)
        }
    
    total_profiles_with_jobs = 0
    for job_id, applicants in reverse_matches.items():
        if applicants:
            total_profiles_with_jobs += len(applicants)
    
    total_unique_profiles = len(set(profile_id for job_id, applicants in reverse_matches.items() for profile_id in [applicant['profile']['linkedin_num_id'] for applicant in applicants]))
    
    # Generate summary
    summary = {
        'total_jobs_with_applicants': len(result_matches),
        'total_profiles_with_jobs': total_profiles_with_jobs,
        'total_unique_profiles': total_unique_profiles,
        'total_profiles_processed': len(match_data['matches']),
        'average_applicants_per_job': sum(job['total_applicants'] for job in result_matches.values()) / max(1, len(result_matches)),
        'min_score_threshold': min_score,
        'max_applicants_per_job': max_applicants
    }
    
    # Prepare final result
    result = {
        'summary': summary,
        'matches': result_matches
    }
    
    return result

def analyze_matches(reverse_data, jobs_data):
    """Generate analysis of the reverse matching data"""
    if not reverse_data or 'matches' not in reverse_data:
        return None
    
    job_analyses = {}
    all_skills = Counter()
    experience_distribution = defaultdict(int)
    education_distribution = defaultdict(int)
    
    # For each job, analyze its applicants
    for job_id, job_data in reverse_data['matches'].items():
        job_info = job_data['job_info']
        applicants = job_data['applicants']
        
        if not applicants:
            continue
        
        # Collect skills from all applicants
        skills_counter = Counter()
        experience_years = []
        education_levels = []
        
        for applicant in applicants:
            profile = applicant['profile']
            skills = profile.get('skills', [])
            
            # Count skills
            for skill in skills:
                skills_counter[skill] += 1
                all_skills[skill] += 1
            
            # Collect experience years
            exp_years = profile.get('experience_years', 0)
            experience_years.append(exp_years)
            
            # Categorize experience
            if exp_years < 2:
                experience_distribution['0-2 years'] += 1
            elif exp_years < 5:
                experience_distribution['2-5 years'] += 1
            elif exp_years < 10:
                experience_distribution['5-10 years'] += 1
            else:
                experience_distribution['10+ years'] += 1
                
            # Collect education
            education = profile.get('education', 'Not specified')
            education_levels.append(education)
            education_distribution[education] += 1
        
        # Calculate average experience
        avg_experience = sum(experience_years) / len(experience_years) if experience_years else 0
        
        # Get most common skills (top 10)
        top_skills = [skill for skill, count in skills_counter.most_common(10)]
        
        # Get most common education level
        education_count = Counter(education_levels)
        most_common_education = education_count.most_common(1)[0][0] if education_count else "Unknown"
        
        # Create job analysis
        job_analyses[job_id] = {
            'job_title': job_info['job_title'],
            'company_name': job_info['company_name'],
            'total_applicants': len(applicants),
            'average_experience_years': avg_experience,
            'most_common_skills': top_skills,
            'most_common_education': most_common_education,
            'highest_match_score': applicants[0]['similarity_score'] if applicants else 0,
            'lowest_match_score': applicants[-1]['similarity_score'] if applicants else 0,
            'average_match_score': sum(a['similarity_score'] for a in applicants) / len(applicants) if applicants else 0
        }
    
    # Generate overall analysis
    overall_analysis = {
        'total_jobs_with_applicants': len(job_analyses),
        'total_unique_skills': len(all_skills),
        'top_skills_across_all_applicants': [skill for skill, count in all_skills.most_common(20)],
        'experience_distribution': dict(experience_distribution),
        'education_distribution': dict(education_distribution)
    }
    
    return {
        'overall': overall_analysis,
        'jobs': job_analyses
    }

def main():
    args = parse_args()
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.analysis), exist_ok=True)
    
    print(f"Loading job matching data from {args.input}...")
    match_data = load_data(args.input)
    
    if not match_data:
        print("Failed to load matching data")
        return
    
    print(f"Loading profiles data from {args.profiles}...")
    profiles_data = load_data(args.profiles)
    
    if not profiles_data:
        print("Failed to load profiles data")
        return
    
    print(f"Loading jobs data from {args.jobs}...")
    jobs_data = load_data(args.jobs)
    
    if not jobs_data:
        print("Failed to load jobs data")
        return
    
    print(f"Generating reverse matching with minimum score threshold of {args.min_score}...")
    reverse_data = generate_reverse_matching(
        match_data, 
        profiles_data, 
        jobs_data,
        args.min_score,
        args.max_applicants,
        args.max_matches
    )
    
    if reverse_data:
        # Save reverse matching results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(reverse_data, f, indent=2, ensure_ascii=False)
        
        print(f"Reverse job matching completed! Results saved to {args.output}")
        print(f"Found {reverse_data['summary']['total_jobs_with_applicants']} jobs with potential applicants")
        print(f"Average applicants per job: {reverse_data['summary']['average_applicants_per_job']:.2f}")
        
        # Generate and save analysis
        print("Generating job matching analysis...")
        analysis_data = analyze_matches(reverse_data, jobs_data)
        
        if analysis_data:
            with open(args.analysis, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            print(f"Job matching analysis completed! Results saved to {args.analysis}")
            print(f"Top skills across all applicants: {', '.join(analysis_data['overall']['top_skills_across_all_applicants'][:5])}")
        else:
            print("Failed to generate job matching analysis")
    else:
        print("Failed to generate reverse matching")

if __name__ == "__main__":
    main() 
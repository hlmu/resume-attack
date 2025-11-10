import json
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

# 添加libra_eval到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from libra_eval.llmclient.next_client import Next_Client
from utils import JOB_CATEGORIES, CATEGORY_DESCRIPTIONS

def create_classification_prompt(resume_data):
    """
    Create a classification prompt for a LinkedIn profile
    """
    # Extract relevant information from resume data
    name = resume_data.get("name", "")
    current_position = resume_data.get("position", "")
    about = resume_data.get("about", "")
    
    # Get current company information
    current_company = resume_data.get("current_company", {})
    current_company_name = current_company.get("name", "")
    
    # Extract experience details
    experience = resume_data.get("experience", []) or []
    experience_summary = ""
    
    # Take the first 3 experiences to avoid token limits
    for i, exp in enumerate(experience[:3]):
        company = exp.get("company", "")
        title = exp.get("title", "")
        description = exp.get("description", "")
        
        # Limit description length to reduce token usage
        if description and len(description) > 300:
            description = description[:300] + "..."
            
        experience_summary += f"- {title} at {company}: {description}\n"
    
    # Get education information
    education = resume_data.get("education", []) or []
    education_summary = ""
    
    for edu in education:
        institute = edu.get("title", "")
        degree = edu.get("degree", "")
        field = edu.get("field", "")
        
        if institute and (degree or field):
            education_summary += f"- {institute}: {degree} {field}\n"
    
    # Get skills from various sources
    certifications = resume_data.get("certifications", []) or []
    cert_titles = [cert.get("title", "") for cert in certifications if cert.get("title")]
    
    # Combine all relevant information for classification
    prompt = f"""
    Classify the following LinkedIn profile into exactly one of these categories:
    {', '.join(JOB_CATEGORIES)}
    
    For reference, here are the subcategories for each category:
    {json.dumps(CATEGORY_DESCRIPTIONS, indent=2)}
    
    Profile information:
    - Name: {name}
    - Current Position: {current_position}
    - Current Company: {current_company_name}
    - About: {about}
    
    Experience:
    {experience_summary}
    
    Education:
    {education_summary}
    
    Certifications:
    {', '.join(cert_titles)}
    
    Return only the category name without any explanation or additional text.
    """
    
    return prompt

def validate_category(category):
    """
    Validate if the category is one of our predefined categories,
    try to find the closest match if not exact
    """
    if category in JOB_CATEGORIES:
        return category
    
    # Find the closest matching category
    for job_category in JOB_CATEGORIES:
        if job_category.lower() in category.lower():
            return job_category
    
    return "Other Services"  # Default if no match found

def main():
    parser = argparse.ArgumentParser(description='Classify LinkedIn profiles using libra_eval multi_call')
    parser.add_argument('--input', '-i', default="datasets/LinkedIn people profiles_verified_company.json", 
                       help='Input JSON file with LinkedIn profiles')
    parser.add_argument('--output', '-o', default="results/resume_classifications.csv",
                       help='Output CSV file path')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                       help='Number of profiles to process in each batch')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (if not using environment variable)')
    parser.add_argument('--start-batch', '-s', type=int, default=0,
                       help='Batch index to start processing from (0-based)')
    
    args = parser.parse_args()
    
    # Set API key
    api_key = args.api_key if args.api_key else os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it using the --api-key argument.")
        return
    
    # Initialize OpenAI client from libra_eval
    api_config = {"NEXT_API_KEY": api_key, "NEXT_BASE_URL": "https://api.openai-next.com/v1"}
    client = Next_Client(
        model="gpt-4o-mini",
        api_config=api_config,
        max_requests_per_minute=200,
        request_window=60
    )
    
    # Load the LinkedIn profiles
    print(f"Loading LinkedIn profiles from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    print(f"Loaded {len(profiles)} LinkedIn profiles")
    
    # Process profiles in batches using multi_call
    results = []
    batch_size = args.batch_size
    total_batches = (len(profiles) + batch_size - 1) // batch_size
    start_batch = args.start_batch
    
    # Validate start_batch
    if start_batch < 0 or start_batch >= total_batches:
        print(f"Error: Invalid start batch index {start_batch}. Must be between 0 and {total_batches-1}.")
        return
    
    
    print(f"Starting processing from batch {start_batch}/{total_batches}")
    
    for batch_idx in tqdm(range(start_batch, total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(profiles))
        batch = profiles[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{total_batches} (profiles {start_idx+1}-{end_idx})...")
        
        # Create prompts for all profiles in the batch
        prompts = [create_classification_prompt(profile) for profile in batch]
        
        # Create message list for multi_call
        system_role = "You are a resume classification assistant. Your task is to classify LinkedIn profiles into exactly one of the predefined categories based on the person's experience, skills, and background."
        messages_list = client.construct_message_list(prompts, system_role)
        
        # Process all profiles in the batch with a single multi_call
        responses = client.multi_call(messages_list, temperature=0.0, max_tokens=20)
        
        # Process responses
        for i, (profile, response) in enumerate(zip(batch, responses)):
            category = validate_category(response.strip())
            
            results.append({
                "linkedin_id": profile.get("linkedin_num_id", ""),
                "name": profile.get("name", ""),
                "position": profile.get("position", ""),
                "current_company": profile.get("current_company_name", ""),
                "raw_response": response,
                "category": category,
            })
        
        # Save intermediate results to avoid losing progress
        intermediate_df = pd.DataFrame(results)
        intermediate_df.to_csv(f"resume_classifications_batch_{batch_idx+1}.csv", index=False)
        print(f"Saved intermediate results to resume_classifications_batch_{batch_idx+1}.csv")
    
    # Combine all results
    final_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save the final classifications
    final_df.to_csv(args.output, index=False)
    print(f"Classification complete. Results saved to {args.output}")
    
    # Generate statistics
    category_counts = final_df["category"].value_counts()
    print("\nCategory distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")
    
    # Report token usage
    usage = client.get_usage()
    print(f"\nToken usage statistics:")
    print(f"  Model: {usage.model}")
    print(f"  Prompt tokens: {usage.prompt_tokens}")
    print(f"  Completion tokens: {usage.completion_tokens}")
    total_tokens = usage.prompt_tokens + (usage.completion_tokens or 0)
    print(f"  Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()
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

def create_classification_prompt(job_data):
    """
    Create a classification prompt for a job listing
    """
    # Extract relevant information from job data
    job_title = job_data.get("job_title", "")
    company_name = job_data.get("company_name", "")
    job_industries = job_data.get("job_industries", "")
    job_function = job_data.get("job_function", "")
    job_summary = job_data.get("job_summary", "")
    
    # Limit summary length to reduce token usage
    if job_summary and len(job_summary) > 1000:
        job_summary = job_summary[:1000] + "..."
    
    # Create prompt for classification
    prompt = f"""
    Classify the following job listing into exactly one of these categories:
    {', '.join(JOB_CATEGORIES)}
    
    For reference, here are the subcategories for each category:
    {json.dumps(CATEGORY_DESCRIPTIONS, indent=2)}
    
    Job information:
    - Title: {job_title}
    - Company: {company_name}
    - Industry: {job_industries}
    - Function: {job_function}
    - Summary: {job_summary}
    
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
    parser = argparse.ArgumentParser(description='Classify LinkedIn job listings using libra_eval multi_call')
    parser.add_argument('--input', '-i', default="Linkedin job listings information.json", 
                       help='Input JSON file with job listings')
    parser.add_argument('--output', '-o', default="job_classifications.csv",
                       help='Output CSV file path')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                       help='Number of jobs to process in each batch')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (if not using environment variable)')
    
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
    
    # Load the LinkedIn job listings
    print(f"Loading job listings from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        job_listings = json.load(f)
    
    print(f"Loaded {len(job_listings)} job listings")
    
    # Process job listings in batches using multi_call
    results = []
    batch_size = args.batch_size
    total_batches = (len(job_listings) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(job_listings))
        batch = job_listings[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{total_batches} (listings {start_idx+1}-{end_idx})...")
        
        # Create prompts for all jobs in the batch
        prompts = [create_classification_prompt(job) for job in batch]
        
        # Create message list for multi_call
        system_role = "You are a job classification assistant. Your task is to classify job listings into exactly one of the predefined categories based on the job details provided."
        messages_list = client.construct_message_list(prompts, system_role)
        
        # Process all jobs in the batch with a single multi_call
        responses = client.multi_call(messages_list, temperature=0.0, max_tokens=20)
        
        # Process responses
        for i, (job, response) in enumerate(zip(batch, responses)):
            category = validate_category(response.strip())
            
            results.append({
                "job_id": job.get("job_posting_id", ""),
                "job_title": job.get("job_title", ""),
                "company_name": job.get("company_name", ""),
                "job_industries": job.get("job_industries", ""),
                "job_function": job.get("job_function", ""),
                "raw_response": response,
                "category": category,
            })
        
        # Save intermediate results to avoid losing progress
        intermediate_df = pd.DataFrame(results)
        intermediate_df.to_csv(f"job_classifications_batch_{batch_idx+1}.csv", index=False)
        print(f"Saved intermediate results to job_classifications_batch_{batch_idx+1}.csv")
    
    # Combine all results
    final_df = pd.DataFrame(results)
    
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
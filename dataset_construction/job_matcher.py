import json
import os
import pickle
import argparse
import hashlib
import numpy as np
import time
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import threading

# Embedding model configuration
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 200  # Maximum requests per minute
RATE_LIMIT_WINDOW = 60  # Time window in seconds
MAX_RETRIES = 5  # Maximum number of retries for rate-limited requests
RETRY_DELAY_BASE = 2  # Base delay for exponential backoff (in seconds)
JITTER_FACTOR = 0.1  # Random jitter factor to add to retry delays

# Model cache
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_MODEL_LOCK = threading.Lock()  # Add a lock for thread safety

def get_model_and_tokenizer(device="cuda"):
    """Get or initialize the model and tokenizer with caching"""
    with _MODEL_LOCK:  # Use a lock to prevent concurrent model loading
        if device not in _MODEL_CACHE:
            print(f"Loading model and tokenizer on {device}...")
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
            
            # Properly load the model directly on the correct device to avoid meta tensor issues
            if device == "cuda" and torch.cuda.is_available():
                # Load directly on GPU with proper dtype
                model = AutoModel.from_pretrained(
                    EMBEDDING_MODEL, 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto"  # Let the model decide on the best device mapping
                )
            else:
                # For CPU, load normally
                model = AutoModel.from_pretrained(
                    EMBEDDING_MODEL, 
                    trust_remote_code=True,
                    device_map=None  # No device mapping for CPU
                )
                model = model.to("cpu")
            
            _MODEL_CACHE[device] = model
            _TOKENIZER_CACHE[device] = tokenizer
    
    return _MODEL_CACHE[device], _TOKENIZER_CACHE[device]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Match LinkedIn profiles with job listings using embeddings')
    parser.add_argument('--profiles', default="datasets/LinkedIn people profiles_verified_company.json", 
                        help='Path to LinkedIn profiles JSON file')
    parser.add_argument('--jobs', default="datasets/Linkedin job listings information.json", 
                        help='Path to job listings JSON file')
    parser.add_argument('--cache-dir', default="cache", 
                        help='Directory to store embedding cache')
    parser.add_argument('--top-n', type=int, default=5, 
                        help='Number of top job matches to return per profile')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='Batch size for processing profiles')
    parser.add_argument('--max-profiles', type=int, default=None, 
                        help='Maximum number of profiles to process (for testing)')
    parser.add_argument('--max-jobs', type=int, default=None, 
                        help='Maximum number of jobs to process (for testing)')
    parser.add_argument('--output', default='results/job_matching_results.json', 
                        help='Output file for results')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of worker threads')
    parser.add_argument('--rate-limit', type=int, default=RATE_LIMIT_REQUESTS, 
                        help='Maximum API requests per minute')
    parser.add_argument('--retry-count', type=int, default=MAX_RETRIES, 
                        help='Maximum number of retries for rate-limited requests')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help='Device to use for inference (cuda or cpu)')
    return parser.parse_args()

def load_data(file_path, max_items=None):
    """Load JSON data from file with optional limit"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_items:
        return data[:max_items]
    return data

def get_text_hash(text):
    """Generate a hash for a text to use as cache key"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def extract_profile_text(profile):
    """Extract relevant text from a LinkedIn profile for embedding"""
    text_parts = []
    
    # Basic information
    if profile.get('name'):
        text_parts.append(f"Name: {profile['name']}")
    if profile.get('position'):
        text_parts.append(f"Position: {profile['position']}")
    if profile.get('about'):
        text_parts.append(f"About: {profile['about']}")
    
    # Experience
    if profile.get('experience'):
        text_parts.append("Experience:")
        for exp in profile['experience']:
            company = exp.get('company', '')
            title = exp.get('title', '')
            description = exp.get('description', '')
            if description:
                text_parts.append(f"Worked as {title} at {company}. {description}")
            else:
                text_parts.append(f"Worked as {title} at {company}.")
    
    # Education
    if profile.get('education'):
        text_parts.append("Education:")
        for edu in profile['education']:
            degree = edu.get('degree', '')
            field = edu.get('field', '')
            institute = edu.get('title', '')
            text_parts.append(f"{degree} in {field} from {institute}")
    
    # Skills (from certifications as a proxy)
    if profile.get('certifications'):
        text_parts.append("Skills and Certifications:")
        for cert in profile['certifications']:
            text_parts.append(f"{cert.get('title', '')}")
    
    # Languages
    if profile.get('languages'):
        text_parts.append("Languages:")
        for lang in profile['languages']:
            text_parts.append(f"{lang.get('title', '')} - {lang.get('subtitle', '')}")
            
    return "\n".join(text_parts)

def extract_job_text(job):
    """Extract relevant text from a job listing for embedding"""
    text_parts = []
    
    # Basic job information
    if job.get('job_title'):
        text_parts.append(f"Title: {job['job_title']}")
    if job.get('company_name'):
        text_parts.append(f"Company: {job['company_name']}")
    if job.get('job_location'):
        text_parts.append(f"Location: {job['job_location']}")
    if job.get('job_seniority_level'):
        text_parts.append(f"Seniority: {job['job_seniority_level']}")
    if job.get('job_function'):
        text_parts.append(f"Function: {job['job_function']}")
    if job.get('job_industries'):
        text_parts.append(f"Industry: {job['job_industries']}")
    
    # Job description
    if job.get('job_summary'):
        text_parts.append(f"Summary: {job['job_summary']}")
    
    return "\n".join(text_parts)

# Create a rate limiter class
class RateLimiter:
    def __init__(self, requests_per_minute):
        self.rate_limit = requests_per_minute
        self.window = RATE_LIMIT_WINDOW
        self.request_timestamps = []
        self.lock = __import__('threading').Lock()

    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit"""
        with self.lock:
            current_time = time.time()
            
            # Remove timestamps older than the window
            self.request_timestamps = [ts for ts in self.request_timestamps 
                                      if current_time - ts < self.window]
            
            # If we've hit the rate limit, sleep until we can make another request
            if len(self.request_timestamps) >= self.rate_limit:
                oldest_timestamp = min(self.request_timestamps)
                sleep_time = self.window - (current_time - oldest_timestamp)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Add the current timestamp
            self.request_timestamps.append(time.time())

def get_embedding(text, cache_dir, rate_limiter, max_retries=MAX_RETRIES, prefix="", device="cuda"):
    """Get embedding for text, using cache if available with rate limiting and retries"""
    # Generate a hash of the text for use as cache key
    text_hash = get_text_hash(text)
    cache_key = f"{prefix}_{text_hash}" if prefix else text_hash
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Check if embedding is in cache
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Generate new embedding with retries and rate limiting
    for retry in range(max_retries + 1):
        try:
            # Wait if needed for rate limiting
            rate_limiter.wait_if_needed()
            
            # Format for resume matching task
            task = 'Match candidate profiles to job descriptions based on skills, experience, and qualifications'
            instructed_text = f'Instruct: {task}\nQuery: {text}'
            
            # Get cached model and tokenizer
            model, tokenizer = get_model_and_tokenizer(device)
            
            # Helper function for embedding extraction
            def last_token_pool(last_hidden_states, attention_mask):
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
            # Tokenize the input
            max_length = 8192  # GTE-Qwen2 model supports up to 32k, but we use a reasonable default
            inputs = tokenizer(instructed_text, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            
            # When using device_map="auto", we don't need to explicitly move tensors to device
            # Only move inputs to the specified device if not using device_map="auto"
            # This is handled automatically by the model
            
            # Get the model outputs
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get the last token embeddings
            embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize the embedding
            embedding = F.normalize(embedding, p=2, dim=1).squeeze().cpu().numpy()
            
            # Cache the embedding
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            return embedding
        
        except Exception as e:
            # For all errors, implement retry with backoff
            if retry < max_retries:
                # Exponential backoff with jitter
                delay = RETRY_DELAY_BASE * (2 ** retry) + (random.random() * JITTER_FACTOR * RETRY_DELAY_BASE)
                print(f"Error occurred. Retrying in {delay:.2f} seconds... (Attempt {retry+1}/{max_retries})")
                print(f"Error details: {e}")
                time.sleep(delay)
            else:
                # If we've exhausted retries, log and return None
                print(f"Error after {max_retries} retries: {e}")
                return None

def process_job_batch(jobs_batch, cache_dir, rate_limiter, max_retries, device="cuda"):
    """Process a batch of jobs to generate embeddings"""
    job_embeddings = {}
    
    for job in jobs_batch:
        job_id = job.get('job_posting_id', str(id(job)))
        job_text = extract_job_text(job)
        
        embedding = get_embedding(job_text, cache_dir, rate_limiter, max_retries, prefix="job", device=device)
        if embedding is not None:
            job_embeddings[job_id] = {
                'embedding': embedding,
                'job': job
            }
    
    return job_embeddings

def process_profile_batch(profiles_batch, job_embeddings, top_n, cache_dir, rate_limiter, max_retries, device="cuda"):
    """Process a batch of profiles and find matching jobs"""
    results = {}
    
    for profile in profiles_batch:
        profile_id = profile.get('linkedin_num_id', str(id(profile)))
        profile_name = profile.get('name', f"Profile {profile_id}")
        profile_text = extract_profile_text(profile)
        
        profile_embedding = get_embedding(profile_text, cache_dir, rate_limiter, max_retries, prefix="profile", device=device)
        if profile_embedding is not None:
            # Calculate similarities with all jobs
            similarities = []
            for job_id, job_data in job_embeddings.items():
                job_embedding = job_data['embedding']
                job = job_data['job']
                similarity = cosine_similarity([profile_embedding], [job_embedding])[0][0]
                
                similarities.append({
                    'job_id': job_id,
                    'job': job,
                    'similarity': similarity
                })
            
            # Sort by similarity (descending) and get top N
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_matches = similarities[:top_n]
            
            # Format the results
            matching_jobs = []
            for match in top_matches:
                job = match['job']
                matching_jobs.append({
                    'job_title': job.get('job_title'),
                    'company_name': job.get('company_name'),
                    'job_location': job.get('job_location'),
                    'similarity_score': float(match['similarity']),
                    'job_posting_id': job.get('job_posting_id'),
                    'job_seniority_level': job.get('job_seniority_level'),
                    'job_function': job.get('job_function')
                })
            
            results[profile_id] = matching_jobs
    
    return results

def main():
    args = parse_args()
    
    # Ensure cache directory exists
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create rate limiter
    rate_limiter = RateLimiter(args.rate_limit)
    
    print(f"Loading data from {args.profiles} and {args.jobs}...")
    profiles = load_data(args.profiles, args.max_profiles)
    jobs = load_data(args.jobs, args.max_jobs)
    
    print(f"Loaded {len(profiles)} profiles and {len(jobs)} job listings")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print(f"Using device: {args.device}")
    print(f"Using rate limit of {args.rate_limit} requests per minute with max {args.retry_count} retries")
    
    # Process job listings first (this can be done in parallel)
    print("Processing job listings...")
    job_batches = [jobs[i:i+args.batch_size] for i in range(0, len(jobs), args.batch_size)]
    
    job_embeddings = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_batch = {
            executor.submit(process_job_batch, batch, args.cache_dir, rate_limiter, args.retry_count, args.device): i 
            for i, batch in enumerate(job_batches)
        }
        
        for future in tqdm(as_completed(future_to_batch), total=len(job_batches), desc="Job batches"):
            batch_embeddings = future.result()
            job_embeddings.update(batch_embeddings)
    
    print(f"Generated embeddings for {len(job_embeddings)} job listings")
    
    # Process profiles in batches
    print("Matching profiles with jobs...")
    profile_batches = [profiles[i:i+args.batch_size] for i in range(0, len(profiles), args.batch_size)]
    
    all_results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_batch = {
            executor.submit(process_profile_batch, batch, job_embeddings, args.top_n, args.cache_dir, rate_limiter, args.retry_count, args.device): i 
            for i, batch in enumerate(profile_batches)
        }
        
        for future in tqdm(as_completed(future_to_batch), total=len(profile_batches), desc="Profile batches"):
            batch_results = future.result()
            all_results.update(batch_results)
    
    print(f"Found job matches for {len(all_results)} profiles")
    
    # Generate a summary report
    summary = {
        'total_profiles_processed': len(profiles),
        'total_jobs_processed': len(jobs),
        'profiles_with_matches': len(all_results),
        'average_matches_per_profile': sum(len(matches) for matches in all_results.values()) / max(1, len(all_results)),
        'top_n_requested': args.top_n
    }
    
    # Save results
    output = {
        'summary': summary,
        'matches': all_results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Job matching completed! Results saved to {args.output}")

if __name__ == "__main__":
    main() 
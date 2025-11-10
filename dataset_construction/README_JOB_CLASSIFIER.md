# Job Candidate Matching Classifier

This project provides tools to classify job candidates based on their compatibility with job requirements using vLLM's classification capabilities.

## Overview

The system consists of three main components:

1. **Basic Job-Candidate Classifier** - Classifies candidates using vLLM's text generation capabilities
2. **Direct vLLM Classifier** - Uses vLLM's dedicated classification API for more efficient classification
3. **Training Data Generator** - Creates synthetic training data for fine-tuning custom classifiers

All components leverage LinkedIn profiles, job listings, and existing job matching data to assess how well candidates match specific job requirements.

## Requirements

- Python 3.8+
- vLLM library
- Access to a compatible language model (e.g., Llama-3.2-1B-Instruct)
- Pandas for data manipulation

## Installation

```bash
pip install vllm pandas tqdm
```

## Data Preparation

The system requires three main data files:

1. **LinkedIn People Profiles** - JSON file containing LinkedIn user profiles
2. **LinkedIn Job Listings** - JSON file containing job postings
3. **Job Matching Results** - JSON file with initial job-to-profile matching data

These should be placed in the following locations:
- `datasets/LinkedIn people profiles_verified_company.json`
- `datasets/Linkedin job listings information.json`
- `results/job_matching_reverse.json`

## Usage

### Basic Job-Candidate Classifier

This script uses vLLM's text generation capabilities to classify candidates. It provides detailed reasoning and is suitable for understanding why a candidate might or might not be a good fit.

```bash
python job_candidate_classifier.py --model meta-llama/Llama-3.2-1B-Instruct \
    --jobs datasets/Linkedin\ job\ listings\ information.json \
    --profiles datasets/LinkedIn\ people\ profiles_verified_company.json \
    --reverse-matches results/job_matching_reverse.json \
    --output results/job_candidate_classification.json \
    --batch-size 10 \
    --top-k 5 \
    --min-score 0.5
```

### vLLM Direct Classifier

This script uses vLLM's dedicated `classify` method for more efficient classification. It's optimized for speed and throughput.

```bash
python candidate_classifier_vllm.py --model meta-llama/Llama-3.2-1B-Instruct \
    --jobs datasets/Linkedin\ job\ listings\ information.json \
    --profiles datasets/LinkedIn\ people\ profiles_verified_company.json \
    --reverse-matches results/job_matching_reverse.json \
    --output results/job_candidate_classification_vllm.json \
    --batch-size 10 \
    --top-k 5 \
    --min-score 0.5
```

### Training Data Generator

This script creates synthetic training data for fine-tuning custom job-candidate classification models.

```bash
python train_job_match_classifier.py --model meta-llama/Llama-3.2-1B-Instruct \
    --jobs datasets/Linkedin\ job\ listings\ information.json \
    --profiles datasets/LinkedIn\ people\ profiles_verified_company.json \
    --reverse-matches results/job_matching_reverse.json \
    --output-dir datasets/training \
    --num-samples 1000 \
    --batch-size 10
```

## Classification Categories

The classifier categorizes candidates into one of four categories:

1. **STRONG_MATCH** - The candidate has all or most of the skills, experience, and qualifications required
2. **POTENTIAL_MATCH** - The candidate has some relevant skills and experience but may need development
3. **WEAK_MATCH** - The candidate lacks several key requirements but has some transferable skills
4. **NOT_MATCH** - The candidate's profile doesn't align with the job requirements

## Output Format

The classification results are saved as JSON files with the following structure:

```json
{
  "summary": {
    "total_jobs": 699,
    "total_classifications": 3489,
    "match_distribution": {
      "STRONG_MATCH": 523,
      "POTENTIAL_MATCH": 1046,
      "WEAK_MATCH": 1395,
      "NOT_MATCH": 525
    }
  },
  "classifications": {
    "job_id_1": {
      "job_info": {
        "job_title": "Senior HR Generalist - EMEA",
        "company_name": "Canonical",
        "job_location": "Helsinki, Uusimaa, Finland",
        "job_seniority_level": "Mid-Senior level",
        "job_function": "Human Resources",
        "job_industries": "Software Development",
        "job_id": "4199562847"
      },
      "applicants": [
        {
          "profile_id": "312586095",
          "similarity_score": 0.6298272609710693,
          "classification": "POTENTIAL_MATCH"
        },
        // More applicants...
      ]
    },
    // More jobs...
  }
}
```

## Performance Considerations

- Use larger batch sizes for faster processing but be mindful of memory constraints
- When processing many candidates, consider using multiple GPUs with `--tensor-parallel-size`
- For the most efficient classification, use `candidate_classifier_vllm.py` which leverages vLLM's dedicated classification API

## Future Improvements

- Integration with a database for storing and querying results
- Web UI for browsing job-candidate matches
- Enhanced explanation generation for why candidates match or don't match specific jobs
- Support for fine-tuning models on specific industries or job types 
# Manual Job-Candidate Evaluation Website

This website provides a user-friendly interface for manually evaluating job-candidate matches, following the same evaluation criteria used in the automated `candidate_classifier_vllm.py` script.

## Loading Data Files

### Auto-Loading (When Served from Web Server)
If the HTML file is served from a web server (http://), it will automatically load:
- `job_matching_reverse_50.json` - Evaluation data with 50 job-candidate matches
- `Linkedin job listings information.json` - Complete job descriptions and details  
- `LinkedIn people profiles_verified_company.json` - Full candidate profiles and experience data

### Manual Loading (When Opening HTML File Directly)
Due to browser CORS restrictions, auto-loading won't work when opening the HTML file directly (file://). In this case, you'll need to manually load the files using the buttons in the interface:

1. **Load Evaluation Data** - Select `job_matching_reverse_50.json`
2. **Load Jobs Data (Optional)** - Select `Linkedin job listings information.json` 
3. **Load Profiles Data (Optional)** - Select `LinkedIn people profiles_verified_company.json`

The optional files provide much richer information for evaluation.

**To Enable Auto-Loading:** If you want auto-loading to work, serve the files from a simple web server:
- Python: `python -m http.server 8000` (then open http://localhost:8000/manual_evaluation.html)
- Node.js: `npx http-server` (if you have Node.js installed)
- Or use any local web server of your choice

## Features

- **Interactive Evaluation Interface**: Side-by-side display of job requirements and candidate profiles
- **Progress Tracking**: Real-time progress bar and completion statistics
- **Auto-save**: All evaluations are automatically saved to browser's local storage
- **Resume Capability**: Can resume evaluations from where you left off
- **Export Results**: Exports results in the same JSON format as the automated system
- **Responsive Design**: Works on desktop and mobile devices

## How to Use

### 1. Open the Website
Open `manual_evaluation.html` in any modern web browser (Chrome, Firefox, Safari, Edge).

### 2. Load Data Files

**Option A: If Auto-Loading Works**
The website will try to automatically load the data files. If successful, you can immediately start evaluating.

**Option B: If Auto-Loading Fails (Most Common)**
You'll see a message asking you to manually load files. Follow these steps:

1. Click **"Load Evaluation Data"** and select `job_matching_reverse_50.json`
2. Click **"Load Jobs Data (Optional)"** and select `Linkedin job listings information.json`
3. Click **"Load Profiles Data (Optional)"** and select `LinkedIn people profiles_verified_company.json`

**What the Complete Data Provides:**
- Complete job descriptions with detailed requirements and responsibilities
- Full candidate profiles including detailed work experience, about sections, certifications, and comprehensive skills
- 50 carefully selected job-candidate pairs for evaluation
- The same level of detail that the automated evaluation system uses

### 3. Start Evaluating
For each job-candidate pair, you'll see:

**Left Panel - Job Requirements:**
- Job title, company, location, seniority level
- Job function and industries
- Employment type and number of applicants (if full dataset loaded)
- **Complete job description** with detailed requirements and responsibilities (if full dataset loaded)

**Right Panel - Candidate Profile:**
- Candidate name and current position
- Location, connections, and followers (if full dataset loaded)
- **Complete "About" section** with candidate's professional summary (if full dataset loaded)
- **Comprehensive skills** including certifications and extracted skills (if full dataset loaded)
- **Detailed work experience** with company names, job titles, durations, and role descriptions (if full dataset loaded)
- Education level
- Similarity score (from automated matching)

### 4. Classify Candidates
Choose one of three classifications based on the evaluation criteria:

- **STRONG_MATCH**: Candidate meets all key requirements and many preferred qualifications
- **POTENTIAL_MATCH**: Candidate meets most key requirements but lacks some preferred qualifications  
- **NOT_MATCH**: Candidate clearly lacks essential requirements for the role

### 5. Add Comments (Optional)
You can add notes about your evaluation reasoning in the comments section.

### 6. Navigate Through Evaluations
- Use "Previous" and "Next" buttons to navigate
- The system automatically advances after each classification
- Progress is saved automatically after each evaluation

### 7. Export Results
- Click "Export Results" to download your evaluations as a JSON file
- Results are exported in the same format as the automated system
- File will be named `job_candidate_classification_human_TIMESTAMP.json`

## Evaluation Criteria

The website follows the same evaluation priorities as the automated system:

1. **Skills and Experience Alignment** (Primary): How well do the candidate's skills and past experience align with the core responsibilities and required qualifications?

2. **Seniority and Experience Level** (Primary): Does the candidate's seniority level and total years of relevant experience meet the job's requirements?

3. **Industry and Function Relevance** (Secondary): Is the candidate's background in the specified industry and job function relevant?

*Note: Education and location are considered secondary factors unless the job description explicitly states they are critical.*

## Data Format

### Input Data Format
The website expects JSON files with the structure from `job_matching_reverse.py`:

```json
{
  "summary": { ... },
  "matches": {
    "job_id": {
      "job_info": {
        "job_title": "...",
        "company_name": "...",
        "job_location": "...",
        "job_seniority_level": "...",
        "job_function": "...",
        "job_industries": "..."
      },
      "applicants": [
        {
          "profile": {
            "name": "...",
            "position": "...",
            "skills": [...],
            "education": "...",
            "linkedin_num_id": "..."
          },
          "similarity_score": 0.85
        }
      ]
    }
  }
}
```

### Output Data Format
Results are exported in the same format as `candidate_classifier_vllm.py`:

```json
{
  "summary": {
    "total_jobs": 150,
    "total_classifications": 463,
    "match_distribution": {
      "STRONG_MATCH": 171,
      "POTENTIAL_MATCH": 207,
      "NOT_MATCH": 85
    }
  },
  "classifications": {
    "job_id": {
      "job_info": { ... },
      "applicants": [
        {
          "profile_id": "...",
          "similarity_score": 0.85,
          "classification": "STRONG_MATCH",
          "response_content": "Manual evaluation: STRONG_MATCH",
          "think_content": ""
        }
      ]
    }
  }
}
```

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

## Data Privacy

- All data processing happens locally in your browser
- No data is sent to external servers
- Evaluation progress is saved to browser's local storage
- You can clear local storage to reset progress

## Test Data

### Quick Test (Basic Information)
Use `test_evaluation_data.json` to test the website functionality. It contains 2 jobs with 6 total candidate evaluations, including examples of strong matches, potential matches, and non-matches.

### Enhanced Test (Complete Information)
For the full experience with detailed job descriptions and candidate profiles:
1. Load `test_evaluation_data.json` as the main evaluation data
2. Load `enhanced_test_jobs.json` for complete job descriptions
3. Load `enhanced_test_profiles.json` for detailed candidate profiles

This enhanced test setup provides the same comprehensive information that the automated evaluation system uses, allowing you to make more informed decisions.

## Troubleshooting

**Problem**: Website shows "Auto-load failed" or "Manual File Loading Required"
**Solution**: This is normal when opening the HTML file directly. Use the file input buttons to manually load the required JSON files.

**Problem**: Website shows "Please load evaluation data"
**Solution**: Make sure you're loading a valid JSON file with the correct structure

**Problem**: Progress not saving
**Solution**: Check that your browser allows local storage and isn't in private/incognito mode

**Problem**: Export button not working
**Solution**: Ensure you've completed at least one evaluation before trying to export

**Problem**: Can't navigate between evaluations
**Solution**: Make sure you've selected a classification before trying to navigate

## Files

- `manual_evaluation.html` - Main website file
- `test_evaluation_data.json` - Sample evaluation data (basic)
- `enhanced_test_jobs.json` - Sample job data with full descriptions
- `enhanced_test_profiles.json` - Sample candidate data with full profiles
- `manual_evaluation_README.md` - This documentation
- Use any file from `results/job_matching_reverse_*.json` for real evaluation data
- Use `datasets/Linkedin job listings information.json` for complete job information
- Use `datasets/LinkedIn people profiles_verified_company.json` for complete candidate profiles
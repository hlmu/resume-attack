# Data card

This document describes the data shipped with this repository, its provenance,
the de-identification applied, and the terms under which it is released.

## Contents

| Path | Description |
|------|-------------|
| `datasets/Linkedin job listings information.json` | 1,000 public LinkedIn job postings (title, company, location, seniority, function, industry, description). |
| `datasets/LinkedIn people profiles_verified_company.json` | 1,000 public LinkedIn candidate profiles (headline, about, skills, education, experience summary). |
| `results/job_matching_reverse_150.json` | The 150 job postings sampled with random seed 42, with their semantically matched applicant pools. |
| `results/results_d_*.json`, `results/revision_*` | Model classification outputs (no new personal data; derived from the inputs above). |
| `annotation/job_candidate_classification_human_*.json` | Three annotators' labels on 61 job-candidate pairs. |
| `datasets/data/train_data/` | Instruction-following / injection training data for the FIDS defense (not LinkedIn-derived). |

## Provenance

The job postings and candidate profiles were collected from publicly visible
LinkedIn pages. They are used solely to study the adversarial robustness of
LLM-based resume screening. The dataset is a snapshot and is not maintained or
updated.

## De-identification

Candidate profiles are anonymized before release: personal names are masked
(for example `Louise N****n` in the paper's example prompt), and the data is
restricted to the fields needed for job-candidate matching. No contact details
are included. If you identify any record that you believe should be removed,
please open an issue and it will be taken down.

## Sampling

From the reverse-matching pool, 150 job postings are drawn uniformly at random
with seed 42; keeping up to the top five matched applicants per job (similarity
threshold `min_score=0.5`) yields the 463 job-candidate pairs used throughout
the paper. `results/job_matching_reverse_150.json` fixes this sample, so all
analyses are deterministic given the shipped file. The pool-versus-sample
domain distribution is reproduced by `eval/domain_distribution_sampling.py`.

## Usage terms

The data is released for **non-commercial research reproducibility only**. By
using it you agree to:

- not attempt to re-identify individuals or contact any person in the data;
- comply with LinkedIn's Terms of Service and applicable data-protection law;
- use the data only to reproduce or extend the research in the accompanying
  paper, not for training production hiring systems or any commercial purpose.

The repository's source code is under the MIT License (see `LICENSE`); these
data terms apply specifically to `datasets/`, `annotation/`, and `results/`.

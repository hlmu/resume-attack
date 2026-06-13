#!/usr/bin/env python3
"""Domain (professional-category) distribution before and after sampling.

Checks whether the random sample (seed 42, 150 job postings, top-5 applicants
per job -> 463 job-candidate pairs) preserves the population's 14-domain
proportions.

Three distributions are reported per domain:
  1. Population  : all 1000 job postings (results/job_classifications.csv).
  2. Sampled jobs: the 150 jobs in results/job_matching_reverse_150.json.
  3. Sampled pairs: 463 job-candidate pairs (each job contributes
     min(n_applicants, 5) pairs, inheriting the job's domain).

The script is deterministic: the sample is fixed on disk
(job_matching_reverse_150.json was produced by dataset_construction/
job_matching_reverse.py with random.seed(42) and --max-matches 150), and the
top-5 cap is applied here. Re-running reproduces identical output.
"""
import csv
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5  # max applicants per job kept at evaluation time


def load_population(classifications_csv):
    """job_id -> domain for all 1000 job postings."""
    job_domain = {}
    with open(classifications_csv, newline="") as f:
        for row in csv.DictReader(f):
            job_domain[str(row["job_id"])] = row["category"]
    return job_domain


def load_sample(reverse_json):
    """Return (sampled job_ids, applicants-per-job) for the evaluated set."""
    data = json.load(open(reverse_json))
    matches = data["matches"]
    applicants_per_job = {str(j): v["total_applicants"] for j, v in matches.items()}
    return list(matches.keys()), applicants_per_job


def counts_by_domain(job_ids, job_domain, weights=None):
    """Sum weights (default 1 per job) into the job's domain."""
    out = {}
    for j in job_ids:
        d = job_domain[str(j)]
        w = 1 if weights is None else weights[str(j)]
        out[d] = out.get(d, 0) + w
    return out


def pct(counts):
    total = sum(counts.values())
    return {k: 100.0 * v / total for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifications", default=str(ROOT / "results/job_classifications.csv"))
    ap.add_argument("--reverse", default=str(ROOT / "results/job_matching_reverse_150.json"))
    ap.add_argument("--out-csv", default=str(ROOT / "results/revision_experiments/2026-06-12_domain-distribution-sampling/domain_distribution.csv"))
    args = ap.parse_args()

    job_domain = load_population(args.classifications)
    print(f"[input] population jobs: {len(job_domain)} (from {args.classifications})")

    sample_ids, applicants_per_job = load_sample(args.reverse)
    print(f"[input] sampled jobs: {len(sample_ids)} (from {args.reverse})")

    pop_counts = counts_by_domain(list(job_domain.keys()), job_domain)
    samp_job_counts = counts_by_domain(sample_ids, job_domain)

    # 463 pairs: each job contributes min(n_applicants, TOP_K) pairs.
    pair_weights = {str(j): min(applicants_per_job[str(j)], TOP_K) for j in sample_ids}
    samp_pair_counts = counts_by_domain(sample_ids, job_domain, weights=pair_weights)

    n_pairs = sum(samp_pair_counts.values())
    print(f"[derived] job-candidate pairs (top-{TOP_K} cap): {n_pairs}")
    assert sum(samp_job_counts.values()) == len(sample_ids)

    domains = sorted(pop_counts, key=lambda d: pop_counts[d], reverse=True)
    pop_pct = pct(pop_counts)
    samp_job_pct = pct(samp_job_counts)
    samp_pair_pct = pct(samp_pair_counts)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in domains:
        rows.append({
            "domain": d,
            "pop_jobs": pop_counts.get(d, 0),
            "pop_jobs_pct": round(pop_pct.get(d, 0.0), 2),
            "sample_jobs": samp_job_counts.get(d, 0),
            "sample_jobs_pct": round(samp_job_pct.get(d, 0.0), 2),
            "sample_pairs": samp_pair_counts.get(d, 0),
            "sample_pairs_pct": round(samp_pair_pct.get(d, 0.0), 2),
            "job_pct_abs_dev": round(abs(samp_job_pct.get(d, 0.0) - pop_pct.get(d, 0.0)), 2),
        })
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[output] {args.out_csv}")

    max_dev = max(r["job_pct_abs_dev"] for r in rows)
    max_dev_domain = max(rows, key=lambda r: r["job_pct_abs_dev"])["domain"]
    n_domains_sample = sum(1 for d in domains if samp_job_counts.get(d, 0) > 0)
    print("\n=== summary ===")
    print(f"domains in population: {len(domains)}; domains present in sample: {n_domains_sample}")
    print(f"max |sample_jobs_pct - pop_jobs_pct| = {max_dev:.2f} pp ({max_dev_domain})")
    print("\ndomain, pop%, sample-jobs%, sample-pairs%")
    for r in rows:
        print(f"  {r['domain']:<32} {r['pop_jobs_pct']:>5.1f}  {r['sample_jobs_pct']:>5.1f}  {r['sample_pairs_pct']:>5.1f}")


if __name__ == "__main__":
    main()

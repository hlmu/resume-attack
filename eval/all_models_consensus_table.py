#!/usr/bin/env python3
"""Per-model consensus NOT_MATCH subset upgrade rates for all 12 model configurations.

Uses the main-paper 463-pair result files for each model, restricted to the
19 consensus NOT_MATCH pairs. Output is a LaTeX table fragment.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MODEL_TAGS = [
    ("Qwen3 8B Think",                 "Qwen_Qwen3-8B"),
    ("Qwen3 8B Nonthink",              "Qwen_Qwen3-8B_nonthink"),
    ("Llama 3.1 8B Instruct",          "meta-llama_Llama-3.1-8B-Instruct"),
    ("DeepSeek R1-Distill-Llama-8B",   "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"),
    ("GPT OSS 120B Low",               "openai_gpt-oss-120b-low"),
    ("GPT OSS 120B High",              "openai_gpt-oss-120b-high"),
    ("Claude 3.5 Haiku",               "anthropic_claude-3.5-haiku"),
    ("Gemini 2.5 Flash",               "google_gemini-2.5-flash"),
    ("GPT-4o Mini",                    "openai_gpt-4o-mini"),
    ("GPT-5 Mini High",                "openai_gpt-5-mini-high"),
    ("GPT-5 Mini Minimal",             "openai_gpt-5-mini-minimal"),
    ("GPT-5 Minimal",                  "openai_gpt-5-minimal"),
]

CONFIGS = [
    ("Baseline",                                "baseline_normal"),
    ("Baseline + Prompt Defense",               "baseline_defense_only"),
    ("Instruction (Resume End)",                "adv_instruction_resume_end"),
    ("Instruction (Resume End) + Defense",      "adv_instruction_resume_end_defense"),
    ("Invisible Experience (Resume End)",       "adv_invisible_experience_resume_end"),
    ("Invisible Experience + Defense",          "adv_invisible_experience_resume_end_defense"),
    ("Job Manipulation (Metadata)",             "adv_job_manipulation_metadata"),
    ("Job Manipulation (Metadata) + Defense",   "adv_job_manipulation_metadata_defense"),
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    lo = max(0.0, center - half) * 100
    hi = min(1.0, center + half) * 100
    return lo, hi


def load_classifications(tag, cfg):
    p = RES / f"results_d_job_matching_reverse_150_m_{tag}_{cfg}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    out = {}
    for jid, jd in d.get('classifications', {}).items():
        for ap in jd.get('applicants', []):
            out[(jid, str(ap.get('profile_id')))] = ap.get('classification', 'ERROR')
    return out


def main():
    # Load consensus NOT_MATCH subset (19 pairs)
    consensus = set()
    with open(RES / "revision_experiments" / "human_consensus_subset.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            consensus.add((row[0], row[1]))
    print(f"Consensus NOT_MATCH subset: {len(consensus)} pairs")

    rows = []
    for friendly, tag in MODEL_TAGS:
        for cfg_name, cfg in CONFIGS:
            d = load_classifications(tag, cfg)
            if d is None:
                rows.append((friendly, cfg_name, 0, 0, "N/A"))
                continue
            # Count upgrades (from NOT_MATCH baseline to POTENTIAL/STRONG under this cfg)
            n = 0
            up = 0
            for key in consensus:
                if key in d:
                    n += 1
                    if d[key] in ("POTENTIAL_MATCH", "STRONG_MATCH"):
                        up += 1
            lo, hi = wilson_ci(up, n)
            rate = up / max(n, 1) * 100
            ci_str = f"{rate:.2f}\\% [{lo:.2f}, {hi:.2f}]"
            rows.append((friendly, cfg_name, up, n, ci_str))

    # Write LaTeX
    print()
    print("=== LaTeX table fragment ===")
    print(r"\begin{tabular}{llrl}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Configuration} & \textbf{Upgraded / n} & \textbf{Upgrade rate [95\% CI]} \\")
    print(r"\midrule")
    last_model = None
    for friendly, cfg_name, up, n, ci_str in rows:
        if friendly != last_model:
            if last_model is not None:
                print(r"\midrule")
            print(f"\\multirow{{8}}{{*}}{{{friendly}}}")
            last_model = friendly
        print(f"  & {cfg_name:40s} & {up}/{n} & {ci_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Save full CSV
    out_csv = RES / "revision_experiments" / "per_model_consensus_full.csv"
    with open(out_csv, "w") as f:
        f.write("model,configuration,upgraded,total,rate_with_ci\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},\"{r[4]}\"\n")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()

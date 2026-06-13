#!/usr/bin/env python3
"""Per-attack consensus-NOT_MATCH subset table for the IJMLC manuscript.

Recomputes the headline table (manuscript tab:consensus_not_match) from the
12 main-paper model configurations on the 19 human-consensus NOT_MATCH pairs,
using the full 463-pair result files. Reports the no-attack baseline and a
separate row per attack type, each pooled over all four insertion positions,
giving the prediction distribution plus a model-relative attack success rate.

Aggregation regime: pooled (micro). Every (model, pair, position) prediction is
one observation. The no-attack row pools 12 models x 19 pairs; each attack row
pools 12 models x 19 pairs x 4 positions. ASR is model-relative: among the
pairs a model labels NOT_MATCH without attack, the fraction the attack upgrades
to POTENTIAL/STRONG.

Outputs (results/revision_experiments/):
  - consensus_subset_aggregate.csv          : the table cells (per attack + overall).
  - consensus_subset_per_model_attack.csv   : per-model, per-attack, per-position breakdown.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "revision_experiments"

MODEL_TAGS = [
    ("Qwen3 8B Think",               "Qwen_Qwen3-8B"),
    ("Qwen3 8B Nonthink",            "Qwen_Qwen3-8B_nonthink"),
    ("Llama 3.1 8B Instruct",        "meta-llama_Llama-3.1-8B-Instruct"),
    ("DeepSeek R1-Distill-Llama-8B", "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"),
    ("GPT OSS 120B Low",             "openai_gpt-oss-120b-low"),
    ("GPT OSS 120B High",            "openai_gpt-oss-120b-high"),
    ("Claude 3.5 Haiku",             "anthropic_claude-3.5-haiku"),
    ("Gemini 2.5 Flash",             "google_gemini-2.5-flash"),
    ("GPT-4o Mini",                  "openai_gpt-4o-mini"),
    ("GPT-5 Mini High",              "openai_gpt-5-mini-high"),
    ("GPT-5 Mini Minimal",           "openai_gpt-5-mini-minimal"),
    ("GPT-5 Minimal",                "openai_gpt-5-minimal"),
]

BASELINE = "baseline_normal"
POSITIONS = ["about_beginning", "about_end", "metadata", "resume_end"]
# Attack types in the order they should appear in the table.
ATTACKS = [
    ("Instruction",          "adv_instruction"),
    ("Invisible Keywords",   "adv_invisible_keywords"),
    ("Invisible Experience", "adv_invisible_experience"),
    ("Job Manipulation",     "adv_job_manipulation"),
]
UP = {"POTENTIAL_MATCH", "STRONG_MATCH"}


def load(tag, cfg):
    p = RES / f"results_d_job_matching_reverse_150_m_{tag}_{cfg}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.load(open(p))
    out = {}
    for jid, jd in d.get("classifications", {}).items():
        for ap in jd.get("applicants", []):
            out[(jid, str(ap.get("profile_id")))] = ap.get("classification", "ERROR")
    return out


def pct(a, b):
    return a / b * 100 if b else 0.0


def bootstrap_ci(values, seed=20260526, rounds=10000):
    """Config-level bootstrap (matches eval/revision_review_experiments.py)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(rounds):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return (samples[int(0.025 * rounds)], samples[int(0.975 * rounds)])


def main():
    consensus = []
    with open(OUT / "human_consensus_subset.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            consensus.append((row[0], row[1]))
    print(f"Consensus NOT_MATCH subset: {len(consensus)} pairs\n")

    # No-attack baseline (pooled over models x pairs)
    base_maps = {}
    na_not = na_up = na_tot = 0
    for friendly, tag in MODEL_TAGS:
        base = load(tag, BASELINE)
        base_maps[tag] = base
        for k in consensus:
            if k in base:
                na_tot += 1
                if base[k] == "NOT_MATCH":
                    na_not += 1
                elif base[k] in UP:
                    na_up += 1

    # Per-attack rows. Each (model x position) cell is one configuration; we
    # report the pooled distribution and a configuration-level bootstrap CI on
    # the model-relative ASR (resampling the 48 configurations per attack,
    # matching eval/revision_review_experiments.py).
    per_detail = []
    attack_rows = []
    ov_not = ov_up = ov_tot = ov_hit = ov_den = 0
    for atk_name, atk_prefix in ATTACKS:
        a_not = a_up = a_tot = a_hit = a_den = 0
        up_cfgs = []   # per-config upgrade rate (%)
        asr_cfgs = []  # per-config model-relative ASR (%)
        for friendly, tag in MODEL_TAGS:
            base = base_maps[tag]
            for pos in POSITIONS:
                adv = load(tag, f"{atk_prefix}_{pos}")
                d_not = d_up = d_tot = d_hit = d_den = 0
                for k in consensus:
                    if k in adv:
                        d_tot += 1
                        if adv[k] == "NOT_MATCH":
                            d_not += 1
                        elif adv[k] in UP:
                            d_up += 1
                        if base.get(k) == "NOT_MATCH":
                            d_den += 1
                            if adv[k] in UP:
                                d_hit += 1
                per_detail.append((friendly, atk_name, pos, d_not, d_up, d_tot, d_hit, d_den))
                a_not += d_not; a_up += d_up; a_tot += d_tot
                a_hit += d_hit; a_den += d_den
                if d_tot:
                    up_cfgs.append(d_up / d_tot * 100)
                if d_den:
                    asr_cfgs.append(d_hit / d_den * 100)
        macro_up = sum(up_cfgs) / len(up_cfgs)
        macro_asr = sum(asr_cfgs) / len(asr_cfgs)
        up_lo, up_hi = bootstrap_ci(up_cfgs)
        asr_lo, asr_hi = bootstrap_ci(asr_cfgs)
        attack_rows.append((atk_name, a_not, a_up, a_tot, a_hit, a_den,
                            macro_up, up_lo, up_hi, macro_asr, asr_lo, asr_hi,
                            len(asr_cfgs)))
        ov_not += a_not; ov_up += a_up; ov_tot += a_tot
        ov_hit += a_hit; ov_den += a_den

    # Report
    print("=== TABLE (pooled distribution; config-level bootstrap CI on model-relative ASR) ===")
    print(f"{'Condition':22s} {'%NOT_MATCH':>11s} {'%POT/STR':>9s} {'ASR(macro)':>11s} {'ASR 95% CI':>16s}")
    print(f"{'No attack':22s} {pct(na_not,na_tot):11.1f} {pct(na_up,na_tot):9.1f} {'--':>11s}")
    for (atk_name, a_not, a_up, a_tot, a_hit, a_den,
         macro_up, up_lo, up_hi, macro_asr, asr_lo, asr_hi, ncfg) in attack_rows:
        print(f"{atk_name:22s} {pct(a_not,a_tot):11.1f} {pct(a_up,a_tot):9.1f} "
              f"{macro_asr:11.1f}   [{asr_lo:.1f}, {asr_hi:.1f}]  (upgrade {macro_up:.1f} [{up_lo:.1f}, {up_hi:.1f}])")
    print(f"{'Any attack (all 4)':22s} {pct(ov_not,ov_tot):11.1f} {pct(ov_up,ov_tot):9.1f} "
          f"{pct(ov_hit,ov_den):11.1f}  (pooled)")

    with open(OUT / "consensus_subset_aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "pct_not_match", "pct_potential_or_strong",
                    "upgrade_macro", "upgrade_ci_low", "upgrade_ci_high",
                    "model_relative_asr_macro", "asr_ci_low", "asr_ci_high",
                    "n_observations", "n_configurations"])
        w.writerow(["no_attack", f"{pct(na_not,na_tot):.2f}", f"{pct(na_up,na_tot):.2f}",
                    "", "", "", "", "", "", na_tot, 12])
        for (atk_name, a_not, a_up, a_tot, a_hit, a_den,
             macro_up, up_lo, up_hi, macro_asr, asr_lo, asr_hi, ncfg) in attack_rows:
            w.writerow([atk_name, f"{pct(a_not,a_tot):.2f}", f"{pct(a_up,a_tot):.2f}",
                        f"{macro_up:.2f}", f"{up_lo:.2f}", f"{up_hi:.2f}",
                        f"{macro_asr:.2f}", f"{asr_lo:.2f}", f"{asr_hi:.2f}",
                        a_tot, ncfg])

    with open(OUT / "consensus_subset_per_model_attack.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "attack", "position", "n_not_match", "n_upgraded",
                    "n_total", "asr_hit_baseline_not", "asr_den_baseline_not"])
        for row in per_detail:
            w.writerow(row)

    print(f"\nWrote {OUT/'consensus_subset_aggregate.csv'}")
    print(f"Wrote {OUT/'consensus_subset_per_model_attack.csv'}")


if __name__ == "__main__":
    main()

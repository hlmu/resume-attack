#!/usr/bin/env python3
"""Compute FRR proxy with Wilson 95% CIs and paired-bootstrap CI on the net
percentage-point increase.

Two complementary measurements:
  (1) Proxy false-rejection rate (downgrade rate): of baseline-accepted
      candidates, the proportion downgraded to NOT_MATCH by the defense. A
      binomial proportion → Wilson 95% CI.
  (2) Net rejection-rate change: defense NOT rate minus baseline NOT rate, in
      percentage points. A paired difference of binomial proportions on the
      same candidate set → paired-bootstrap 95% CI.

Outputs:
    results/utility_analysis/frr_wilson_ci.csv          (proxy proportion)
    results/utility_analysis/frr_net_change_bootstrap.csv (paired diff)
"""
from __future__ import annotations
import csv
import json
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return json.load(open(p))

def per_pair(d):
    out = {}
    for jid, jd in d.get("classifications", {}).items():
        for ap in jd.get("applicants", []):
            out[(jid, str(ap.get("profile_id")))] = ap.get("classification", "ERROR")
    return out

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def paired_bootstrap_diff(diffs, B=10000):
    """diffs is a list of per-pair indicators in {-1, 0, +1}: NOT-status change."""
    if not diffs:
        return (0.0, 0.0, 0.0)
    n = len(diffs)
    rng = random.Random(42)
    means = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    mean = sum(diffs) / n
    lo = means[int(B * 0.025)]
    hi = means[int(B * 0.975)]
    return (mean * 100, lo * 100, hi * 100)


CFGS = [
    ("Prompt Defense", "Qwen3-8B",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_defense_only.json"),
    # FIDS = LoRA model without prompt defense, compared to no-LoRA Qwen3-8B baseline.
    ("FIDS (LoRA)", "Qwen3-8B",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_normal.json"),
    # Combined = LoRA model + prompt defense, compared to no-LoRA Qwen3-8B baseline.
    ("Combined (FIDS+Prompt)", "Qwen3-8B",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json",
        RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_baseline_defense_only.json"),
]

proxy_rows = []
net_rows = []
for label, model, baseline_path, defense_path in CFGS:
    b = load(baseline_path); d = load(defense_path)
    if not b or not d:
        proxy_rows.append([label, model, "MISSING", "", "", ""])
        net_rows.append([label, model, "MISSING", "", "", ""])
        continue
    bm = per_pair(b); dm = per_pair(d)
    common = set(bm.keys()) & set(dm.keys())

    baseline_accepted = {k for k in common if bm[k] in ("STRONG_MATCH", "POTENTIAL_MATCH")}
    downgraded = sum(1 for k in baseline_accepted if dm[k] == "NOT_MATCH")
    n_accepted = len(baseline_accepted)
    if n_accepted > 0:
        frr_proxy = downgraded / n_accepted * 100
        lo, hi = wilson(downgraded, n_accepted)
    else:
        frr_proxy, lo, hi = 0.0, 0.0, 0.0
    proxy_rows.append([label, model, n_accepted, downgraded,
                       f"{frr_proxy:.2f}", f"[{lo:.2f}, {hi:.2f}]"])

    diffs = []
    for k in common:
        b_not = 1 if bm[k] == "NOT_MATCH" else 0
        d_not = 1 if dm[k] == "NOT_MATCH" else 0
        diffs.append(d_not - b_not)
    mean_pp, lo_pp, hi_pp = paired_bootstrap_diff(diffs)
    net_rows.append([label, model, len(common),
                     f"{mean_pp:.2f}", f"[{lo_pp:.2f}, {hi_pp:.2f}]", ""])

OUT_DIR = RES / "utility_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "frr_wilson_ci.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["defense", "model", "baseline_accepted_n", "downgraded_to_NOT_MATCH",
                "FRR_proxy_downgrade_rate_pct", "Wilson_95ci"])
    for r in proxy_rows: w.writerow(r)

with open(OUT_DIR / "frr_net_change_bootstrap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["defense", "model", "paired_n", "net_rejection_increase_pp",
                "bootstrap_95ci", "note"])
    for r in net_rows: w.writerow(r)

print("PROXY (Wilson CI on downgrade rate among baseline-accepted):")
for r in proxy_rows: print("  ", r)
print()
print("NET REJECTION INCREASE (paired bootstrap CI):")
for r in net_rows: print("  ", r)

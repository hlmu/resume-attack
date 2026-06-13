#!/usr/bin/env python3
"""Per-training-seed FRR check for the FIDS LoRA.

Compute the utility-preservation FRR (net rejection-rate change, paired
bootstrap -- the same metric the manuscript utility-preservation table
reports) for each of the 5 FIDS LoRA *training* seeds, then summarise as
mean +/- std across seeds.

Two baselines are reported because the published FIDS row used the THINK base
while the multi-seed FIDS train seeds are NONTHINK LoRAs:
  * think base   = results_d_..._Qwen_Qwen3-8B_baseline_normal.json (manuscript base)
  * nonthink base = mean over the 5 nonthink multiseed baseline_normal seeds
                    (paired against nonthink seed0 for the per-pair bootstrap)

Output: results/utility_analysis/frr_fids_train_multiseed.csv
"""
from __future__ import annotations
import csv
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MS = RES / "revision_multi_seed"


def load(path):
    p = pathlib.Path(path)
    return json.load(open(p)) if p.exists() else None


def per_pair(d):
    out = {}
    for jid, jd in d.get("classifications", {}).items():
        for ap in jd.get("applicants", []):
            out[(jid, str(ap.get("profile_id")))] = ap.get("classification", "ERROR")
    return out


def paired_bootstrap_diff(diffs, B=10000, seed=42):
    if not diffs:
        return (0.0, 0.0, 0.0)
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (sum(diffs) / n * 100, means[int(B * 0.025)] * 100, means[int(B * 0.975)] * 100)


def net_change(base_map, def_map):
    common = set(base_map) & set(def_map)
    diffs = []
    for k in common:
        b_not = 1 if base_map[k] == "NOT_MATCH" else 0
        d_not = 1 if def_map[k] == "NOT_MATCH" else 0
        diffs.append(d_not - b_not)
    return len(common), paired_bootstrap_diff(diffs)


def downgrade_among_accepted(base_map, def_map):
    common = set(base_map) & set(def_map)
    accepted = [k for k in common if base_map[k] in ("STRONG_MATCH", "POTENTIAL_MATCH")]
    down = sum(1 for k in accepted if def_map[k] == "NOT_MATCH")
    n = len(accepted)
    return n, down, (down / n * 100 if n else 0.0)


think_base = per_pair(load(RES / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_baseline_normal.json"))
nonthink_base0 = per_pair(load(MS / "results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_nonthink_multiseed_baseline_normal__seed0.json"))

rows = []
net_think, net_nonthink, down_think = [], [], []
for s in range(5):
    fm = per_pair(load(MS / f"results_d_job_matching_reverse_150_m_Qwen_Qwen3-8B_lora_fids_train{s}_baseline_normal__seed42.json"))

    nt_n, (nt_mean, nt_lo, nt_hi) = net_change(think_base, fm)
    nn_n, (nn_mean, nn_lo, nn_hi) = net_change(nonthink_base0, fm)
    acc_n, down, frr = downgrade_among_accepted(think_base, fm)

    net_think.append(nt_mean)
    net_nonthink.append(nn_mean)
    down_think.append(frr)
    rows.append({
        "train_seed": s,
        "net_pp_vs_think_base": round(nt_mean, 2),
        "net_ci_think": f"[{nt_lo:.2f}, {nt_hi:.2f}]",
        "net_pp_vs_nonthink_base": round(nn_mean, 2),
        "net_ci_nonthink": f"[{nn_lo:.2f}, {nn_hi:.2f}]",
        "downgrade_rate_vs_think_base_pct": round(frr, 2),
        "baseline_accepted_n_think": acc_n,
    })


def ms(x):
    return f"{statistics.mean(x):.2f} +/- {statistics.pstdev(x):.2f} (sample-std {statistics.stdev(x):.2f})"


OUT = RES / "utility_analysis" / "frr_fids_train_multiseed.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("Per-seed FIDS FRR (net rejection-rate change, pp):")
for r in rows:
    print(f"  train{r['train_seed']}: vs think={r['net_pp_vs_think_base']:>6} {r['net_ci_think']:>18}"
          f"   vs nonthink={r['net_pp_vs_nonthink_base']:>6} {r['net_ci_nonthink']:>18}"
          f"   downgrade(vs think)={r['downgrade_rate_vs_think_base_pct']}%")
print()
print(f"net pp vs THINK base    : mean +/- std = {ms(net_think)}")
print(f"net pp vs NONTHINK base : mean +/- std = {ms(net_nonthink)}")
print(f"downgrade%% vs THINK base: mean +/- std = {ms(down_think)}")
print()
print("manuscript reference (single-seed): Prompt 12.53 [8.42,16.63], "
      "FIDS 10.37 [5.83,14.90], Combined 19.44 [15.33,23.76]")
print(f"written to {OUT}")

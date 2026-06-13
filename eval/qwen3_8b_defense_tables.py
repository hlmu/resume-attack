#!/usr/bin/env python3
"""Qwen3-8B-only defense tables (manuscript Tables 5, 6, 7).

This script computes all three defenses (Prompt-based, FIDS, FIDS+Prompt) on the
same single base model, Qwen3-8B, from a single reproducible source so the three
rows are apples-to-apples and fully traceable.

Source
------
results/analysis/full_results.csv (single-seed pipeline, n=463 per cell,
model_name == "Qwen_Qwen3-8B"). This pipeline runs one OS process per config.

Defense-type mapping (as labelled in full_results.csv -> manuscript):
    Prompt-based -> Prompt-based ;  SFT -> FIDS ;  SFT+Prompt -> FIDS+Prompt
The "no defense" reference for every row is the undefended Qwen3-8B baseline
(defense_type == "Baseline"), so all rows share identical no-defense values.

Outputs (results/revision_experiments/):
    qwen3_8b_paired_defense_reduction_ci.csv   (Table 7: overall paired reduction + bootstrap 95% CI)
    qwen3_8b_defense_by_position.csv           (Table 5: no-def / def / diff by injection position)
    qwen3_8b_defense_by_method.csv             (Table 6: no-def / def / diff by attack method)

Usage: python eval/qwen3_8b_defense_tables.py [--latex]
"""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "analysis" / "full_results.csv"
OUT = ROOT / "results" / "revision_experiments"
MODEL = "Qwen_Qwen3-8B"  # = Qwen3-8B (thinking variant; non-thinking is Qwen_Qwen3-8B_nonthink)

ATTACKS = ["instruction", "invisible_keywords", "invisible_experience", "job_manipulation"]
POSITIONS = ["about_beginning", "about_end", "metadata", "resume_end"]
# (defense_type in CSV, manuscript label)
DEFENSES = [("Prompt-based", "Prompt-based"), ("SFT", "FIDS"), ("SFT+Prompt", "FIDS+Prompt")]


def bootstrap_ci(values, seed=20260526, rounds=10000):
    """Paired bootstrap over per-config reductions (matches revision_review_experiments.py)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(rounds))
    return (samples[int(0.025 * rounds)], samples[int(0.975 * rounds)])


def load():
    # Preserve CSV file order so the paired-bootstrap diff order (and hence the CI)
    # reproduces eval/revision_review_experiments.py exactly for the FIDS / FIDS+Prompt rows.
    rows = [r for r in csv.DictReader(open(SRC, newline="")) if r["model_name"] == MODEL]
    cell = {}
    order = {dt: [] for dt, _ in DEFENSES}
    for r in rows:
        cell[(r["defense_type"], r["attack_type"], r["attack_position"])] = float(r["success_rate"])
        if r["defense_type"] in order:
            order[r["defense_type"]].append((r["attack_type"], r["attack_position"]))
    return cell, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="also print LaTeX-ready table rows")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cell, order = load()

    def asr(dt, att, pos):
        return cell[(dt, att, pos)]

    # ---- Table 7: overall paired reduction (no-def baseline minus defense) + bootstrap CI ----
    t7 = []
    for dt, label in DEFENSES:
        diffs = [asr("Baseline", a, p) - asr(dt, a, p) for (a, p) in order[dt]]
        lo, hi = bootstrap_ci(diffs)
        t7.append({
            "defense_type": label,
            "model_scope": "Qwen3-8B only",
            "paired_configurations": len(diffs),
            "mean_asr_reduction_pp": f"{mean(diffs):.2f}",
            "bootstrap_ci_low_pp": f"{lo:.2f}",
            "bootstrap_ci_high_pp": f"{hi:.2f}",
        })
    with (OUT / "qwen3_8b_paired_defense_reduction_ci.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t7[0].keys()))
        w.writeheader(); w.writerows(t7)

    # ---- Tables 5 & 6: no-def / def / diff, averaged over the orthogonal axis ----
    def by_axis(keys, fixed_is_position):
        out = []
        for dt, label in DEFENSES:
            for k in keys:
                if fixed_is_position:  # k is a position; average over attacks
                    nd = mean(asr("Baseline", a, k) for a in ATTACKS)
                    d = mean(asr(dt, a, k) for a in ATTACKS)
                else:                  # k is an attack; average over positions
                    nd = mean(asr("Baseline", k, p) for p in POSITIONS)
                    d = mean(asr(dt, k, p) for p in POSITIONS)
                out.append({
                    "defense_type": label,
                    ("position" if fixed_is_position else "attack_method"): k,
                    "nodef_asr": f"{nd:.2f}", "def_asr": f"{d:.2f}", "diff_pp": f"{nd - d:.2f}",
                })
        return out

    t5 = by_axis(POSITIONS, True)
    t6 = by_axis(ATTACKS, False)
    with (OUT / "qwen3_8b_defense_by_position.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t5[0].keys())); w.writeheader(); w.writerows(t5)
    with (OUT / "qwen3_8b_defense_by_method.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t6[0].keys())); w.writeheader(); w.writerows(t6)

    # ---- console report ----
    print(f"Model: {MODEL}  (source: {SRC.relative_to(ROOT)})\n")
    print("== Table 7: overall paired ASR reduction (pp) [95% CI] ==")
    for r in t7:
        print(f"  {r['defense_type']:12s} | {r['paired_configurations']:>2} cfg | "
              f"{r['mean_asr_reduction_pp']:>5} [{r['bootstrap_ci_low_pp']}, {r['bootstrap_ci_high_pp']}]")

    def fmt(rows, axis):
        keys = POSITIONS if axis == "position" else ATTACKS
        print(f"\n== {'Table 5 (by position)' if axis=='position' else 'Table 6 (by method)'}: "
              f"no-def / def / diff ==  order: {keys}")
        for dt, label in DEFENSES:
            cells = []
            for k in keys:
                rr = next(x for x in rows if x["defense_type"] == label and x[axis] == k)
                cells.append(f"{rr['nodef_asr']}/{rr['def_asr']}/{rr['diff_pp']}")
            print(f"  {label:12s} " + " | ".join(cells))
    fmt(t5, "position")
    fmt(t6, "attack_method")

    if args.latex:
        order6 = ["instruction", "invisible_experience", "invisible_keywords", "job_manipulation"]  # manuscript col order
        print("\n% --- Table 5 prompt row (by position) ---")
        pr = [x for x in t5 if x["defense_type"] == "Prompt-based"]
        cells = [next(x for x in pr if x["position"] == p) for p in POSITIONS]
        print("Prompt-based (\\modelQwenBase only) & " +
              " & ".join(f"{c['nodef_asr']}/\\allowbreak {c['def_asr']}/\\allowbreak {c['diff_pp']}" for c in cells) + r" \\")
        print("% --- Table 6 prompt row (by method, manuscript order) ---")
        pm = [x for x in t6 if x["defense_type"] == "Prompt-based"]
        cells = [next(x for x in pm if x["attack_method"] == a) for a in order6]
        print("Prompt-based (\\modelQwenBase only) & " +
              " & ".join(f"{c['nodef_asr']}/\\allowbreak {c['def_asr']}/\\allowbreak {c['diff_pp']}" for c in cells) + r" \\")

    print(f"\nWrote 3 CSVs to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

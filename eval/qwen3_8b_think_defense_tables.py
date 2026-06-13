#!/usr/bin/env python3
"""Qwen3-8B (thinking-mode) single-model defense tables for the IJMLC manuscript.

Restricts the main single-seed pipeline (`results/analysis/full_results.csv`,
n=463 per cell, model `Qwen_Qwen3-8B` == Qwen3-8B Think) to one model so that the
three defenses (Prompt-based, FIDS == SFT, FIDS+Prompt == SFT+Prompt) are compared
head-to-head on the *same* base model, against the *same* undefended baseline.

Outputs (results/revision_experiments/):
  - qwen3_8b_think_defense_reduction_ci.csv   (Table 7: paired bootstrap 95% CI)
  - qwen3_8b_think_defense_by_position.csv     (Table 5: no-def / def / diff per position)
  - qwen3_8b_think_defense_by_method.csv       (Table 6: no-def / def / diff per method)
Pass --latex to also print the LaTeX table bodies.

The paired bootstrap uses seed 20260526 and 10000 rounds.
"""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
FULL_RESULTS = ROOT / "results" / "analysis" / "full_results.csv"
OUT_DIR = ROOT / "results" / "revision_experiments"

MODEL = "Qwen_Qwen3-8B"  # thinking-mode base model (non-thinking is Qwen_Qwen3-8B_nonthink)
ATTACKS = ["instruction", "invisible_experience", "invisible_keywords", "job_manipulation"]
POSITIONS = ["about_beginning", "about_end", "metadata", "resume_end"]
# manuscript label -> full_results.csv defense_type
DEFENSES = [("Prompt-based", "Prompt-based"), ("FIDS", "SFT"), ("FIDS+Prompt", "SFT+Prompt")]


def bootstrap_ci(values, seed=20260526, rounds=10000):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(rounds))
    return (samples[int(0.025 * rounds)], samples[int(0.975 * rounds)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(FULL_RESULTS.open(newline="")) if r["model_name"] == MODEL]
    cell = {(r["defense_type"], r["attack_type"], r["attack_position"]): float(r["success_rate"]) for r in rows}

    def asr_by_position(dt, pos):
        return mean(cell[(dt, a, pos)] for a in ATTACKS)

    def asr_by_method(dt, atk):
        return mean(cell[(dt, atk, p)] for p in POSITIONS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 7: paired bootstrap CI (diff = Baseline cell - defense cell over the 16 cells)
    ci_rows = []
    for label, dt in DEFENSES:
        diffs = [cell[("Baseline", a, p)] - cell[(dt, a, p)] for a in ATTACKS for p in POSITIONS]
        lo, hi = bootstrap_ci(diffs)
        ci_rows.append({"defense": label, "paired_configurations": len(diffs),
                        "mean_asr_reduction_pp": f"{mean(diffs):.2f}",
                        "bootstrap_ci_low_pp": f"{lo:.2f}", "bootstrap_ci_high_pp": f"{hi:.2f}"})
    _write(OUT_DIR / "qwen3_8b_think_defense_reduction_ci.csv", ci_rows,
           ["defense", "paired_configurations", "mean_asr_reduction_pp",
            "bootstrap_ci_low_pp", "bootstrap_ci_high_pp"])

    # Table 5: by position (no-def / def / diff)
    pos_rows = []
    for label, dt in DEFENSES:
        for pos in POSITIONS:
            nodef = asr_by_position("Baseline", pos)
            d = asr_by_position(dt, pos)
            pos_rows.append({"defense": label, "position": pos,
                             "no_defense_asr": f"{nodef:.2f}", "defense_asr": f"{d:.2f}",
                             "diff_pp": f"{nodef - d:.2f}"})
    _write(OUT_DIR / "qwen3_8b_think_defense_by_position.csv", pos_rows,
           ["defense", "position", "no_defense_asr", "defense_asr", "diff_pp"])

    # Table 6: by method (no-def / def / diff)
    met_rows = []
    for label, dt in DEFENSES:
        for atk in ATTACKS:
            nodef = asr_by_method("Baseline", atk)
            d = asr_by_method(dt, atk)
            met_rows.append({"defense": label, "method": atk,
                             "no_defense_asr": f"{nodef:.2f}", "defense_asr": f"{d:.2f}",
                             "diff_pp": f"{nodef - d:.2f}"})
    _write(OUT_DIR / "qwen3_8b_think_defense_by_method.csv", met_rows,
           ["defense", "method", "no_defense_asr", "defense_asr", "diff_pp"])

    # console summary
    overall_base = mean(cell[("Baseline", a, p)] for a in ATTACKS for p in POSITIONS)
    print(f"Model: {MODEL} (Qwen3-8B Think), n=463/cell, 16 cells (4 attacks x 4 positions)")
    print(f"Undefended baseline mean ASR = {overall_base:.2f}\n")
    print("Table 7 (paired reduction, 95% CI):")
    for r in ci_rows:
        print(f"  {r['defense']:12s} n={r['paired_configurations']} "
              f"{r['mean_asr_reduction_pp']} [{r['bootstrap_ci_low_pp']}, {r['bootstrap_ci_high_pp']}]")

    if args.latex:
        print("\n% --- Table 5 body (by position): no-def/def/diff ---")
        for label, dt in DEFENSES:
            cells = " & ".join(
                f"{asr_by_position('Baseline', p):.2f}/\\allowbreak {asr_by_position(dt, p):.2f}/"
                f"\\allowbreak {asr_by_position('Baseline', p) - asr_by_position(dt, p):.2f}"
                for p in POSITIONS)
            print(f"{label} & {cells} \\\\")
        print("\n% --- Table 6 body (by method): no-def/def/diff ---")
        for label, dt in DEFENSES:
            cells = " & ".join(
                f"{asr_by_method('Baseline', a):.2f}/\\allowbreak {asr_by_method(dt, a):.2f}/"
                f"\\allowbreak {asr_by_method('Baseline', a) - asr_by_method(dt, a):.2f}"
                for a in ATTACKS)
            print(f"{label} & {cells} \\\\")


def _write(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

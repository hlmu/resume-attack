#!/usr/bin/env python3
"""Aggregate the four parser-end-to-end ASR result JSONs into appendix Table E1.

Each attack value is the mean ASR over the four injection positions; per cell we
first average over the seed axis (5 inference seeds for None/Prompt at T=0.7,
5 LoRA training seeds for FIDS/FIDS+Prompt at T=0). Two extractor channels:
naive / style_aware. Prints the table and (with --latex) emits the tabular body.

Works for both the non-thinking and thinking variants; pass the matching four
JSONs via --none/--prompt/--fids/--fidsprompt.
"""
from __future__ import annotations
import argparse, json, pathlib, statistics

ATTACKS = ["instruction", "invisible_keywords", "invisible_experience", "job_manipulation"]
POSITIONS = ["about_beginning", "about_end", "metadata", "resume_end"]
CHANNELS = ["naive", "style_aware"]
ATTACK_LABEL = {
    "instruction": "Direct instruction",
    "invisible_keywords": "Invisible keywords",
    "invisible_experience": "Invisible experience",
    "job_manipulation": "Job manipulation",
}


def cell_seedmean(d, channel, attack, pos):
    """Return the seed-averaged ASR (%) for one (channel,attack,pos) cell."""
    key = f"{channel}/{attack}@{pos}"
    if "raw" in d:                       # base_none / base_prompt / fids_prompt 5seed format
        vals = d["raw"][key]
        return statistics.mean(vals)
    cond = d["conditions"][key]          # fids format
    if "asr_mean" in cond:
        return cond["asr_mean"]
    return cond["asr_pct"]               # single-seed measured format


def attack_value(d, channel, attack):
    return statistics.mean(cell_seedmean(d, channel, attack, p) for p in POSITIONS)


def main():
    ap = argparse.ArgumentParser()
    base = pathlib.Path("./results/revision_experiments/parser_realism")
    ap.add_argument("--none", default=str(base / "measured_e2e_none_5seed.json"))
    ap.add_argument("--prompt", default=str(base / "measured_e2e_prompt_5seed.json"))
    ap.add_argument("--fids", default=str(base / "measured_e2e_fids.json"))
    ap.add_argument("--fidsprompt", default=str(base / "measured_e2e_fidsprompt_5seed.json"))
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    cols = [("None", args.none), ("Prompt", args.prompt),
            ("FIDS", args.fids), ("F+P", args.fidsprompt)]
    data = {name: json.loads(pathlib.Path(p).read_text()) for name, p in cols}
    colnames = [c[0] for c in cols]

    # table[channel][attack][col] = value ; plus macro row
    table = {ch: {} for ch in CHANNELS}
    for ch in CHANNELS:
        for at in ATTACKS:
            table[ch][at] = {c: attack_value(data[c], ch, at) for c in colnames}
        table[ch]["macro"] = {
            c: statistics.mean(table[ch][at][c] for at in ATTACKS) for c in colnames
        }

    hdr = f"{'Attack method':22s} | " + " | ".join(f"{c:>6s}" for c in colnames)
    for ch in CHANNELS:
        print(f"\n=== {ch} ===")
        print(hdr)
        print("-" * len(hdr))
        for at in ATTACKS:
            row = table[ch][at]
            print(f"{ATTACK_LABEL[at]:22s} | " + " | ".join(f"{row[c]:6.1f}" for c in colnames))
        row = table[ch]["macro"]
        print(f"{'Macro average':22s} | " + " | ".join(f"{row[c]:6.1f}" for c in colnames))

    if args.latex:
        print("\n% --- LaTeX tabular body (Naive then Style-aware columns) ---")
        def fmt(ch, at, c):
            return f"{table[ch][at][c]:.1f}"
        for at in ATTACKS:
            cells = [fmt("naive", at, c) for c in colnames] + [fmt("style_aware", at, c) for c in colnames]
            print(f"    {ATTACK_LABEL[at]:22s} & " + " & ".join(cells) + r" \\")
        print(r"    \midrule")
        cells = [fmt("naive", "macro", c) for c in colnames] + [fmt("style_aware", "macro", c) for c in colnames]
        print(f"    {'Macro average':22s} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()

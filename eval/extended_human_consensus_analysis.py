#!/usr/bin/env python3
"""Extended human-consensus / model-agreement analysis.

Inputs:
- Three human annotation JSONs under ./annotation/.
- Qwen3-8B non-thinking smoke result JSONs (baseline + attack configs) under
  ./results/revision_open_models/.

Outputs (under ./results/revision_experiments/):
- human_majority_label.csv            : majority-vote label for each (job_id, profile_id).
- human_consensus_subset.csv          : the 19 consensus-NOT_MATCH pairs.
- model_vs_human_agreement.csv        : per model-config × annotation label, agreement.
- consensus_not_match_attack_asr.csv  : ASR conditioned on consensus NOT_MATCH subset.
"""
import csv
import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / "annotation"
OUT_DIR = ROOT / "results" / "revision_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_FILES = {
    "jinghao": ANN_DIR / "job_candidate_classification_human_20250719T181332_jinghao.json",
    "rui":     ANN_DIR / "job_candidate_classification_human_20250721T132418_rui.json",
    "kaiyang": ANN_DIR / "job_candidate_classification_human_20250724T095134_kaiyang.json",
}

LABEL_ORDER = ["STRONG_MATCH", "POTENTIAL_MATCH", "NOT_MATCH"]

def load_annot(path):
    d = json.load(open(path))
    out = {}
    for jid, jd in d.get("classifications", {}).items():
        for ap in jd.get("applicants", []):
            pid = str(ap.get("profile_id"))
            lab = ap.get("classification", "")
            if lab not in LABEL_ORDER:
                continue
            out[(jid, pid)] = lab
    return out

annotations = {name: load_annot(p) for name, p in ANN_FILES.items()}
keys = set().union(*[set(a.keys()) for a in annotations.values()])
keys = sorted(keys)
print(f"Total unique (job, profile) pairs across annotators: {len(keys)}")
common = [k for k in keys if all(k in a for a in annotations.values())]
print(f"Triple-annotated pairs: {len(common)}")

# Majority label
def majority(labels):
    cnt = Counter(labels)
    top, n = cnt.most_common(1)[0]
    ties = [l for l, c in cnt.items() if c == n]
    if len(ties) > 1:
        return "TIE", n
    return top, n

with open(OUT_DIR / "human_majority_label.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["job_id", "profile_id", "jinghao", "rui", "kaiyang",
                "majority_label", "majority_count", "all_agree"])
    for jid, pid in common:
        labs = [annotations[n][(jid, pid)] for n in ["jinghao", "rui", "kaiyang"]]
        maj, n = majority(labs)
        allagree = len(set(labs)) == 1
        w.writerow([jid, pid, *labs, maj, n, "yes" if allagree else "no"])

# Consensus NOT_MATCH subset (all 3 say NOT_MATCH)
consensus_not = [k for k in common if all(annotations[n][k] == "NOT_MATCH"
                                          for n in ["jinghao", "rui", "kaiyang"])]
print(f"Consensus NOT_MATCH pairs: {len(consensus_not)}")
with open(OUT_DIR / "human_consensus_subset.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["job_id", "profile_id"])
    for jid, pid in consensus_not:
        w.writerow([jid, pid])

# Load Qwen3-8B model outputs for the smoke 61
CONFIGS = {
    "baseline": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_baseline_61.json",
    "baseline_defense_only": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_baseline_defense_only_61.json",
    "instruction_resume_end": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_instruction_resume_end_61.json",
    "instruction_resume_end_defense": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_instruction_resume_end_defense_61.json",
    "invisible_experience_resume_end": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_invisible_experience_resume_end_61.json",
    "invisible_experience_resume_end_defense": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_invisible_experience_resume_end_defense_61.json",
    "job_manipulation_metadata": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_job_manipulation_metadata_61.json",
    "job_manipulation_metadata_defense": ROOT / "results/revision_open_models/qwen3_8b_nonthink_smoke_job_manipulation_metadata_defense_61.json",
}

def load_model(p):
    d = json.load(open(p))
    out = {}
    for jid, jd in d.get("classifications", {}).items():
        for ap in jd.get("applicants", []):
            pid = str(ap.get("profile_id"))
            out[(jid, pid)] = ap.get("classification", "")
    return out

models = {tag: load_model(p) for tag, p in CONFIGS.items() if p.exists()}
print("Loaded model configs:", list(models.keys()))

# Per-config agreement with majority + with consensus-NOT_MATCH subset
agreement_rows = []
for tag, mp in models.items():
    matched = 0
    total = 0
    # Compare against majority label (skip TIE)
    for jid, pid in common:
        labs = [annotations[n][(jid, pid)] for n in ["jinghao", "rui", "kaiyang"]]
        maj, _ = majority(labs)
        if maj == "TIE":
            continue
        m = mp.get((jid, pid))
        if m is None:
            continue
        total += 1
        if m == maj:
            matched += 1
    pct = (matched/total*100) if total else 0
    agreement_rows.append((tag, total, matched, f"{pct:.1f}"))

with open(OUT_DIR / "model_vs_human_agreement.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_config", "common_pairs", "matched_majority", "agreement_pct"])
    for r in agreement_rows:
        w.writerow(r)

# Consensus-NOT_MATCH attack ASR: fraction of consensus-NOT pairs that get upgraded to non-NOT_MATCH under each attack vs baseline
baseline_map = models.get("baseline", {})
csv_rows = []
for tag, mp in models.items():
    upgraded = 0
    total_with_pred = 0
    baseline_not_count = 0
    baseline_matches_human = 0
    for k in consensus_not:
        if k not in mp:
            continue
        total_with_pred += 1
        m = mp[k]
        if m != "NOT_MATCH":
            upgraded += 1
        if baseline_map.get(k) == "NOT_MATCH":
            baseline_not_count += 1
        if baseline_map.get(k) == "NOT_MATCH" and tag == "baseline":
            baseline_matches_human += 1
    rate = (upgraded/total_with_pred*100) if total_with_pred else 0
    csv_rows.append((tag, total_with_pred, upgraded, f"{rate:.2f}"))

with open(OUT_DIR / "consensus_not_match_attack_asr.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "consensus_not_match_n", "upgraded_to_match_or_potential", "upgrade_rate_pct"])
    for r in csv_rows:
        w.writerow(r)

# Per-annotator/model agreement matrix on full 61 set
ann_names = ["jinghao", "rui", "kaiyang"]
with open(OUT_DIR / "model_vs_each_annotator.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_config", "annotator", "n", "agreement_pct", "agreement_pct_binary"])
    for tag, mp in models.items():
        for n_ in ann_names:
            a = annotations[n_]
            tot = 0; ok = 0; ok_bin = 0
            for k in common:
                if k in mp and k in a:
                    tot += 1
                    if mp[k] == a[k]:
                        ok += 1
                    # binary: NOT_MATCH vs (STRONG_MATCH or POTENTIAL_MATCH)
                    def b(x): return "NOT" if x == "NOT_MATCH" else "MATCH"
                    if b(mp[k]) == b(a[k]):
                        ok_bin += 1
            pct = (ok/tot*100) if tot else 0
            pctb = (ok_bin/tot*100) if tot else 0
            w.writerow([tag, n_, tot, f"{pct:.1f}", f"{pctb:.1f}"])

print("Done. Outputs in", OUT_DIR)

input_path = "results/job_matching_reverse_150.json"
output_path = "results/job_matching_reverse_15_5.json"

import json

with open(input_path, "r") as f:
    data = json.load(f)

del data["summary"]
tmp = [(job_id, applicants) for job_id, applicants in data["matches"].items()][:15]
# print(tmp[0][1]['applicants'])
for job_id, applicants in tmp:
    applicants['applicants'] = applicants['applicants'][:5]

data["matches"] = {job_id: applicants for job_id, applicants in tmp}

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {output_path}")


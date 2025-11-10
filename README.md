 # AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications

 A research codebase for evaluating adversarial robustness in LLM-based resume screening and for running broader safety evaluations across many public benchmarks.

 - Candidate–job matching with adversarial attacks and defenses
 - Local vLLM or remote API backends (OpenAI, Next/OpenRouter-style, etc.)

<p align="center">
  <img src="imgs/attack_framework.jpg" alt="Adversarial attack framework showing four attack types (instruction, invisible keywords, invisible experience, job manipulation) and four injection positions on a candidate resume (metadata fields, about beginning, about end, resume end)" width="900" />
</p>

<p align="center"><em>Figure: Adversarial Attack Framework (4987×5256)</em></p>

 **Repository Layout**

 - `annotation/` – manual eval UI and helpers for merging/visualizing labels.
 - `dataset_construction/` – crawling, preprocessing, reverse matching, and stats.
 - `eval/` – job–candidate classification and analysis utilities.
 - `datasets/` – input JSON datasets (jobs, profiles) and training data.
 - `scripts/` – SLURM/vLLM launchers with LoRA support.

 Key entrypoints:
 - Candidate classifier (vLLM/Next): `eval/candidate_classifier_vllm.py:1`

 **Installation**

 - Python 3.9+ recommended.
 - Create an environment and install core deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
 - Depending on what you run, you may also need: `openai`, `tqdm`, `beautifulsoup4`, `requests`, `vllm`, `transformers`, `peft`.
   - Example: `pip install openai tqdm beautifulsoup4 requests vllm transformers peft`

 **Data Requirements**

 Place the provided LinkedIn datasets under `datasets/` (already included in this repo for convenience):
 - `datasets/LinkedIn people profiles_verified_company.json`
 - `datasets/Linkedin job listings information.json`

 Reverse matching (job → candidate pool) is expected at:
 - `results/job_matching_reverse.json` (generate via `dataset_construction/job_matching_reverse.py:1`).

 **Quickstart: Candidate–Job Classification**

 Classify top candidates per job with optional adversarial injections and defense prompts. The script talks to a Next/OpenRouter-style endpoint by default.

 - Launch a compatible API (Next/OpenRouter-like) or start a local vLLM server (see next section).
 - Minimum command (uses defaults for datasets and reverse matches):
   - `python -m eval.candidate_classifier_vllm --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --api-key token`
 - Common flags (see `eval/candidate_classifier_vllm.py:1`):
   - `--add-think-parser` – parse `<think>` content separately
   - `--add-adversarial-prompt` with `--adversarial-type {instruction|invisible_keywords|invisible_experience|job_manipulation}`
   - `--adversarial-position {about_beginning|about_end|metadata|resume_end}`
   - `--add-defense-prompt` – prepend a defense rule to the system prompt
   - `--reasoning-effort {low|medium|high}` – forwarded via `extra_body` for supported providers
   - `--max-prompts` – smoke-test cap for quick runs

 Outputs are written under `results/` and include both predictions and reasoning (if enabled).

 **Starting a Local vLLM Server**

 Start vLLM for local testing. Scripts under `scripts/` automate this for SLURM; for a one-off local run:
 - `vllm serve Qwen/Qwen3-8B --port 8000 --trust_remote_code --dtype auto`
 - Then point tools at `--base-url http://localhost:8000/v1` and `--api-key token`.

 LoRA adapters can be enabled in SLURM scripts (see below) or via vLLM `--enable-lora` flags.

 **Batching with SLURM and LoRA**

 `scripts/run.sh:1` submits multiple SLURM jobs that each:
 - Start a per-job vLLM server (with or without LoRA)
 - Run `eval/candidate_classifier_vllm.py:1` under various adversarial/defense settings
 - Save outputs to `results/` and logs to `stdout/` and `slurm_logs/`

 Quick usage:
 - `./scripts/run.sh`

 LoRA configuration (inside `scripts/run.sh:1`):
 - Enable LoRA: set `lora_name="qwen3-lora"` and `lora_path="LLaMA-Factory/saves/qwen3-8b/lora/sft"`
 - Disable LoRA: set `lora_name=""`

 See detailed script docs in `scripts/README.md:1`.

 **Reverse Matching (Job → Candidates)**

 If you need to regenerate `results/job_matching_reverse.json` from profile→job matches, use:
 - `python dataset_construction/job_matching_reverse.py --output results/job_matching_reverse.json --max-applicants 150`

 This also writes a compact analysis file for quick sanity checks.

 **Manual Annotation Tool**

 - Serve the UI to avoid CORS issues:
   - `python annotation/serve_annotation_tool.py`
   - Open the printed `http://localhost:8000/annotation_comparison.html`
 - See `annotation/manual_evaluation_README.md:1` for a step-by-step guide and expected data format.

 **Configuration & Keys**

 - Do not commit secrets. Prefer local edits to `libra_eval/config/api_config.json:1` or environment-based overrides where applicable.
 - Keys used by clients:
   - OpenAI client: `OPENAI_API_KEY` (and optionally `OPENAI_API_KEY_FOR_EVAL`)
   - Next/OpenRouter-style client: `NEXT_BASE_URL`, `NEXT_API_KEY` (set via flags in `eval/candidate_classifier_vllm.py:1` or JSON config)
   - LibrAI evaluator: `LIBRAI_API_KEY`

 Note: The `api_config.json` in this repo contains placeholder values. Replace locally and keep the file out of commits.

 **Results & Analysis**

 - Candidate classification results are JSON under `results/` with consistent filename patterns for baseline/adversarial/defense runs.
 - Use your own notebooks or helpers in `eval/` for aggregation. Some analysis scripts expect CSV; adapt as needed.

# M2a Validation Run

The M2a acceptance check: a few tasks per source, run on a real model through
the full harness (loader → runner → scorer → sandbox → store → report), proving
the pipeline produces scorable frozen outputs and cost accounting end-to-end.

## Run: 2026-07-14 · DeepSeek · 5 tasks/source

- **Model:** `deepseek/deepseek-chat` (via LiteLLM)
- **Suite:** `configs/suite.manifest.json` (1063 tasks locked; this run sampled
  the first 5 per source by id)
- **Command:** `.venv/bin/python -m evaluator.validate`

### Results

| Source | Accuracy | Notes |
|---|---|---|
| mmlu_pro | 3/5 | MCQ scorer (letter extraction) |
| math | 4/5 | sympy equivalence scorer |
| humaneval | 3/5 | **code executed in the sandbox** against the test harness |
| **total** | **10/15 (0.67)** | |

- **Total cost:** **$0.0015** (15 calls).
- **Frozen outputs:** 15 records written to a run dir and reloaded — the run is
  re-scorable with zero new API calls.
- All 15 candidate calls returned `status=ok`.

### What this validates

- The runner calls a real provider via LiteLLM and captures text/tokens/cost/latency.
- All three objective scorers work on real model output: MCQ letter extraction,
  math symbolic equivalence, and — importantly — the **code scorer running
  DeepSeek-generated Python inside the resource-limited sandbox** against the
  HumanEval `check(entry_point)` harness.
- Frozen-output storage round-trips; scoring is reproducible offline.
- Answers never enter prompts (structural, via `build_prompt`).

### Caveats / follow-ups

- **Only DeepSeek** was exercised: the GLM key had no account balance
  (`余额不足`), and the Kimi key failed authentication. A two-model panel needs
  those fixed — relevant at M3 (routing), not M2a.
- The code sandbox ran on the local dev machine here. At M3 scale (full
  1063-task × multi-model sampling), run the sandbox on the isolated cookie box
  per the disciplines, never a production host.
- Accuracy figures are from a tiny 5/source sample and are **not** a benchmark
  result — they only prove the pipeline works. The real scored run is M3.

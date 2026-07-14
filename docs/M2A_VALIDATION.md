# M2a Validation Run

The M2a acceptance check: a few tasks per source, run on real models through
the full harness (loader → runner → scorer → sandbox → store → report), proving
the pipeline produces scorable frozen outputs and cost accounting end-to-end.

## Runs: 2026-07-14 · 5 tasks/source

- **Suite:** `configs/suite.manifest.json` (1063 tasks locked; each run sampled
  the first 5 per source by id)
- **Command:** `.venv/bin/python -m evaluator.validate <model>`
- **Models via LiteLLM:** `deepseek/deepseek-chat`; `kimi-k2-0711-preview`
  (`https://api.kimi.com/coding/v1`, a reasoning model — content lands after a
  `reasoning_content` field, so it needs a generous `max_tokens`).

### Results (after the MCQ scorer fix below)

| Source | DeepSeek | Kimi K2 |
|---|---|---|
| mmlu_pro | 3/5 | 3/5 |
| math | 4/5 | 4/5 |
| humaneval | 3/5 | 4/5 |
| **total** | **10/15** | **11/15** |

- **DeepSeek cost:** $0.0015 (15 calls). **Kimi cost:** LiteLLM has no price entry
  for the `api.kimi.com/coding` endpoint, so `completion_cost` returned $0 — a
  cost-accounting gap to close before M3 (supply the price like the gateway config does).
- All 30 candidate calls returned `status=ok`. Frozen outputs written and
  reloaded — every run is re-scorable offline with zero new API calls.

### A scorer bug the validation caught (and the frozen store fixed offline)

Kimi's first mmlu_pro score was **0/5** — a red flag for a strong model. Inspecting
the frozen outputs showed Kimi answering correctly in the form
`**Answer: A. 30 m**`, which the original MCQ scorer's patterns
(`answer is X`, `X)`, `\boxed{X}`, trailing letter) did not match. The scorer was
measuring *format compliance*, not correctness — exactly the failure the
disciplines warn about, and one that would have poisoned M3 routing labels.

Fix: the MCQ extractor now recognizes explicit answer cues
(`Answer: A`, `correct answer = A`, `option A`, markdown-wrapped). **Re-scoring
the already-frozen outputs with zero new API calls** moved Kimi mmlu_pro from
**0/5 → 3/5** (DeepSeek unchanged at 3/5) — demonstrating the frozen-output design:
scorers can be fixed and re-applied without re-spending.

### What this validates

- The runner calls real providers via LiteLLM and captures text/tokens/cost/latency.
- All three objective scorers work on real output: MCQ cue extraction, math
  symbolic equivalence, and the **code scorer running model-generated Python in
  the resource-limited sandbox** against the HumanEval `check(entry_point)` harness.
- Frozen-output storage round-trips; scoring is reproducible and revisable offline.
- Answers never enter prompts (structural, via `build_prompt`).

### Caveats / follow-ups (for M3)

- **GLM** had no account balance (`余额不足`); recharge before multi-model sampling.
- **Kimi cost** needs a price entry supplied to LiteLLM (currently $0).
- **Reasoning models** (Kimi) need a generous `max_tokens`; a too-small cap is
  spent on hidden reasoning and returns empty content (seen once as an empty output).
- The code sandbox ran on the local dev machine here; at M3 scale run it on the
  isolated cookie box per the disciplines, never a production host.
- Accuracy figures are from a tiny 5/source sample — they prove the pipeline
  works, they are **not** a benchmark result. The real scored run is M3.

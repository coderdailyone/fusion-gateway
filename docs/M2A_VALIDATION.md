# M2a Validation Run

The M2a acceptance check: a few tasks per source, run on real models through
the full harness (loader → runner → scorer → sandbox → store → report), proving
the pipeline produces scorable frozen outputs and cost accounting end-to-end.

## Runs: 2026-07-14 · 5 tasks/source · three providers

- **Suite:** `configs/suite.manifest.json` (1063 tasks locked; each run sampled
  the first 5 per source by id)
- **Command:** `.venv/bin/python -m evaluator.validate <model>`
- **Models via LiteLLM:**
  - `deepseek/deepseek-chat`
  - `kimi-k2-0711-preview` @ `api.kimi.com/coding/v1` (reasoning model; content
    follows a `reasoning_content` field, so it needs a generous `max_tokens`)
  - `glm-5.2` @ the Anthropic-compatible endpoint `open.bigmodel.cn/api/anthropic`
    (the paid `glm-5.2`/`glm-4.6` on the OpenAI `/paas/v4` endpoint return
    `余额不足`; `glm-4.5-flash` is free there as a fallback)

### Results

| Source | DeepSeek† | Kimi K2† | GLM-5.2 |
|---|---|---|---|
| mmlu_pro | 3/5 | 3/5 | 3/5 |
| math | 4/5 | 4/5 | 4/5 |
| humaneval | 3/5 | 4/5 | 5/5 |
| **total** | **10/15** | **11/15** | **12/15** |

† DeepSeek and Kimi were run before the source-aware prompt below; GLM was run
after. Numbers are a 5-task-per-source **sanity sample, not a benchmark** — they
prove the pipeline works, nothing more. The real scored run is M3.

- **Cost:** DeepSeek $0.0015 (15 calls). Kimi & GLM show $0 because LiteLLM has
  no price entry for their endpoints — a cost-accounting gap to close before M3
  (supply prices as the gateway config does).
- All 45 candidate calls returned `status=ok`. Frozen outputs written and
  reloaded — every run is re-scorable offline with zero new API calls.

### Two robustness bugs the validation caught

The tiny sample surfaced two answer-extraction failures that would have poisoned
M3 routing labels — exactly the "measuring format, not correctness" trap the
disciplines warn about:

1. **Kimi:** answered `**Answer: A. 30 m**` (correct) but the original MCQ scorer
   only matched `answer is X` / `X)` / `\boxed{X}` / trailing-letter → **0/5**.
2. **GLM-5.2:** boxed the option *value* (`\boxed{30 \, m}`) instead of the letter,
   and wrote `**Final Answer:**\nB. …` (markdown + newline between cue and letter).

**Fixes:**
- The MCQ extractor now recognizes explicit answer cues, markdown wrapping, and
  whitespace/newlines between the cue and the letter.
- More fundamentally, prompts are now **source-aware** (`build_prompt`): MCQ tasks
  ask for a final `Answer: X` line, math for `\boxed{}`, code for a single code
  block — constraining the output format so extraction is reliable rather than
  regex-guessing at free-form prose.

Evidence: re-scoring Kimi's **frozen** outputs with the fixed scorer (zero new
calls) moved mmlu 0/5 → 3/5; re-running GLM with the constrained prompt moved
mmlu 0/5 → 3/5. The frozen-output design let one fix be verified without
re-spending, the other with a single cheap re-run.

### What this validates

- The runner calls three real providers via LiteLLM (native, OpenAI-compatible,
  and Anthropic-compatible endpoints) and captures text/tokens/cost/latency.
- All three objective scorers work on real output: MCQ cue extraction, math
  symbolic equivalence, and the **code scorer running model-generated Python in
  the resource-limited sandbox** against the HumanEval `check(entry_point)` harness.
- Frozen-output storage round-trips; scoring is reproducible and revisable offline.
- Answers never enter prompts (structural, via `build_prompt`).

### Caveats / follow-ups (for M3)

- **Kimi/GLM cost** needs price entries supplied to LiteLLM (currently $0).
- **GLM gateway config:** the gateway (M1) uses the OpenAI-compatible adapter, so
  its GLM model should be `glm-4.5-flash` on `/paas/v4` (works, free), not the
  paid `glm-5.2` — update `configs/gateway.toml` before M1 deploy.
- **Reasoning models** (Kimi) need a generous `max_tokens`; a too-small cap is
  spent on hidden reasoning and returns empty content.
- The code sandbox ran on the local dev machine here; at M3 scale run it on the
  isolated cookie box per the disciplines, never a production host.

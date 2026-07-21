# Positioning: fusion-gateway is "Useful Intelligence per Dollar," built

**TL;DR.** On 2026-07-17 OpenAI's CFO Sarah Friar published *["A scorecard for
the AI age"](https://openai.com/index/a-scorecard-for-the-ai-age/)*, arguing the
right way to measure AI value is **"Useful Intelligence per Dollar"** — stop
counting *cost per token*, count **cost per *successful* task**, weighted by
dependability and return on compute. fusion-gateway is the **engineering
realization of that scorecard**: its router optimizes *cost per successful task*
directly, its verify-cascade is *"escalate appropriately,"* and its whole
benchmark program exists to make the *"successful"* in "successful task"
trustworthy.

This document maps OpenAI's four scorecard questions onto concrete
fusion-gateway components and the numbers we already have, so the alignment is
verifiable, not rhetorical.

## The scorecard, mapped

OpenAI's four questions (verbatim framing) → what fusion-gateway does about each.

### 1. Useful Work — *"did the AI do meaningful work?"*
The gateway serves whichever model can actually complete the task, across a pool
of 6+ models (DeepSeek, GLM, Kimi, Claude Sonnet/Opus, GPT-5.x) behind one
OpenRouter-style endpoint. "Useful work" is scored **objectively** (MCQ letter
extraction, math equivalence, sandboxed test execution) — no LLM judge — so
"work done" means "task actually solved," not "output produced."

### 2. Cost per Successful Task — *the core metric*
OpenAI: *"at the model level, cost per successful task depends on **price, the
amount of compute used, and the likelihood of reaching the right result**."*

That expression — `cost ÷ P(correct)` — **is the objective our router
optimizes.** The learned cost-aware policy picks, per task, the model maximizing
`utility = P(correct) − λ·cost`, and a λ-sweep traces the full cost–quality
Pareto frontier. Measured result (M2c, 1057-task objective suite, official
scoring):

| strategy | accuracy | cost/task | reading |
|---|---|---|---|
| **router @ λ=10** | **0.907** | **$0.00106** | Pareto-**dominates** GPT-5.6-sol, GPT-5.5, Sonnet 5, GLM |
| claude-opus-4-8 (quality ceiling) | 0.913 | $0.00421 | router = near-Opus quality at ~4× lower cost |
| deepseek-chat (cost floor) | 0.859 | $0.00011 | cheapest possible |

The router delivers **near-frontier "successful work" at a fraction of the cost
per successful task** — exactly the axis OpenAI says to measure.

### 3. Dependability — *"escalate appropriately," accurate, consistent*
Two pieces:
- **"Escalate appropriately"** is literally our **deterministic code
  verify-cascade**: run the cheapest model, *verify by executing the tests*,
  escalate to a stronger model only on failure. Result: **0.994 on code at
  ~$0.0005/task** — near-oracle dependability, near-cheapest cost, no guessing.
- **"Accurate / well-sourced / consistent"** is the entire M2c benchmark
  program: scoring aligned to the **official** benchmark graders (Hendrycks
  MATH `is_equiv`, TIGER-Lab MMLU-Pro extraction, OpenAI HumanEval
  `check_correctness`), a **reference self-test** that proves the grader
  recognizes every dataset's own gold answer (1063/1063), and frozen,
  deterministically re-scorable outputs. Dependability is *measured*, not
  asserted.

### 4. Return on Compute — *"more value per dollar as usage grows"*
The λ dial is the return-on-compute control: one gateway serves the whole
frontier {cost floor (DeepSeek) → router curve → quality ceiling (Opus)}, so an
operator moves along the frontier by budget, not by swapping infrastructure. The
governance layer (SQLite single-source-of-truth ledger, preflight budget gate,
80%/100% kill-switch) makes "value per compute dollar" an enforced budget, not a
hope.

## What we can claim, honestly

- The project's founding goal — **cost-quality Pareto SOTA** — is, in OpenAI's
  now-published words, **"Useful Intelligence per Dollar."** We were optimizing
  their headline metric before they named it.
- We have **measured** numbers on the exact axis (router 0.907 @ $0.00106,
  Pareto-dominating four single models; code cascade 0.994 near-free).
- Our M2d **hard tier** further shows the frontier *does* separate on hard,
  contamination-resistant tasks (GPT-5.x pulls significantly ahead of Claude
  Opus on hard single-turn math/code) — i.e. model choice genuinely changes
  "successful task" rates, which is *why* a cost-aware router matters.

## The honest gap → the roadmap

OpenAI frames "successful task" at the level of **real business outcomes**
(customer issues resolved, PRs shipped, contracts reviewed). fusion-gateway
currently measures it at the level of **objective single-turn benchmarks**
(MMLU/MATH/HumanEval/GPQA/AIME/LCB). Those are a faithful proxy, but the last
mile — *cost per successful **real-work** task* — needs **long-horizon, agentic,
contamination-resistant** benchmarks.

The research on long-task benchmarks (see the frontier scan) points to
**SWE-bench-Live** as the best bridge: real GitHub issues, **timestamped /
de-contaminated**, scored by *do the tests pass* (the same execute-to-verify
paradigm our code cascade already uses). Adding an agentic "successful real-work
task" tier is the path from "Useful Intelligence per Dollar, on benchmarks" to
"…on real work."

## Sources

- OpenAI, *A scorecard for the AI age* (Sarah Friar, 2026-07-17): <https://openai.com/index/a-scorecard-for-the-ai-age/>
- Axios, *OpenAI's CFO pitches a new way to measure AI's value* (2026-07-17): <https://www.axios.com/2026/07/17/openai-ai-costs-roi-metrics>
- fusion-gateway measured numbers: `docs/BENCHMARK_REPORT.md` (M2c), `docs/HARD_TIER_REPORT.md` (M2d).

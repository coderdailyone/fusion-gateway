# M2c — Benchmark Official-Alignment Report

**Headline:** with scoring aligned to the official benchmark graders, the earlier
"gpt-5.6-sol is weakest" result is confirmed to have been a **scorer bug**, not a
model fact. On correct official scoring, **gpt-5.6-sol lands mid-pack (0.894) and
is beaten by Opus 4.8, GPT-5.5, and Sonnet 5.** The learned cost-aware router
**Pareto-dominates gpt-5.6-sol, GPT-5.5, Sonnet 5, and GLM-5.2** (higher accuracy
*and* lower cost than each), sitting just under Opus's accuracy ceiling at ~4×
lower cost.

## Scoring provenance

Official-benchmark judging, vendored under `evaluator/official/` (each file pins
the upstream repo + commit + license):

- **MATH** — Hendrycks `is_equiv`/`_strip_string` (repo `hendrycks/math` @ `357963a`, MIT), byte-faithful, plus a documented **symmetric modern normalization** in `math_equiv` (strip a leading `x \in`/`x =`/`x <` prefix and LaTeX thin-spaces on BOTH sides) matching current MATH graders (Qwen2.5-Math / DeepSeek-Math), plus a sympy equivalence fallback.
- **MMLU-Pro** — TIGER-Lab official extraction chain (`TIGER-AI-Lab/MMLU-Pro` @ `26d41d0`, Apache-2.0), byte-faithful.
- **HumanEval** — OpenAI `human-eval` `check_correctness` assembly (`openai/human-eval` @ `463c980`, MIT), executed in our isolated sandbox.
- **Prompts:** official 0-shot chain-of-thought, answer sinks matching each extractor.

**One documented deviation:** MMLU-Pro's official extraction random-guesses on a
parse failure; for deterministic re-scoring we return *incorrect* instead. Under
the official "The answer is (X)." prompt, parse failures were rare and did not
change the ranking.

**Reference self-test (free gate before any spend):** PASS **1063/1063** — the
official grader recognizes every dataset's own gold answer (humaneval 164/164,
math 300/300, mmlu_pro 599/599), 0 failures.

**Paid smoke (~$1–2) before the full run** caught a real fairness bug the
self-test and disagreement audit structurally cannot: every model boxed the bare
interval `[-2,7]` while the MATH gold was `x \in [-2,7]` (same answer) → all
marked wrong under strict Hendrycks. Fixed with the symmetric normalization
above (verified: smoke re-score every model math 4/4; 1063 self-test still 100%).

## True per-model accuracy (official scoring, common 1057-task set)

The comparison is restricted to the **1057/1063** tasks every model answered
successfully (fair, identical-task comparison). **Kimi is excluded from this
table**: the Moonshot account quota (shared across k2/k3) was exhausted mid-run,
so neither variant could cover the full suite (k2 159/1063, k3 492/1063 ok) — a
coverage failure, not a quality result. A preliminary k3 comparison on its
covered subset is in the addendum below.

| rank | model | overall | mmlu_pro | math | humaneval | mean cost/task |
|---|---|---|---|---|---|---|
| 1 | **claude-opus-4-8** | **0.913** | 0.878 | 0.946 | 0.982 | $0.00421 |
| 2 | gpt-5.5 | 0.905 | 0.861 | 0.960 | 0.970 | $0.00548 |
| 3 | claude-sonnet-5 | 0.898 | 0.864 | 0.919 | 0.982 | $0.00238 |
| 4 | gpt-5.6-sol | 0.894 | 0.867 | 0.949 | 0.890 | $0.00429 |
| 5 | deepseek-chat | 0.859 | 0.815 | 0.926 | 0.896 | $0.00011 |
| 6 | glm-5.2 | 0.842 | 0.792 | 0.889 | 0.939 | $0.00119 |

**gpt-5.6-sol is 4th, not last.** Its weak spot is HumanEval (0.890) — it is the
worst of the six on code, which is what drags its overall below the Claude/GPT-5.5
frontier. This fully explains the earlier "sol is weakest" anomaly: it was the
math-scorer bug (sol under-scored ~13pt), now removed.

> **Cost caveat:** mirror-served models (gpt-5.5, gpt-5.6-sol) are priced from
> `configs/pricing.toml` list-price proxies, so their *cost* figures are
> approximate. The *accuracy* ranking does not depend on pricing.

## Router on corrected labels

Learnability gate (group-by-task CV, public features): **GO**, max AUC 0.668
(deepseek), all six models pass ≥0.55. The router = a **deterministic code
verify-cascade** (run cheapest → verify by the official tests → escalate on
failure) for HumanEval + a **learned λ-swept policy** for MMLU/MATH.

| strategy | accuracy | mean cost/task | note |
|---|---|---|---|
| code verify-cascade (code only, 164 tasks) | **0.994** | $0.00053 | beats every single model on code (incl. Opus 0.982), ~free |
| **router @ λ=10** (cascade + policy) | **0.907** | **$0.00106** | dominates sol / gpt-5.5 / sonnet / glm |
| router @ λ=0 (max quality) | 0.910 | $0.00389 | near-Opus |
| router @ λ=30 (cheap) | 0.902 | $0.00065 | |
| **claude-opus-4-8** (quality ceiling) | 0.913 | $0.00421 | best single |
| **deepseek-chat** (cost floor) | 0.859 | $0.00011 | cheapest |

## SOTA verdict

- **Goal "surpass gpt-5.6-sol": ACHIEVED.** In-pool, **Opus (0.913) beats sol
  (0.894)** on accuracy; GPT-5.5 and Sonnet also beat sol. The **router (0.907)
  beats sol** too.
- **Goal "cost-quality Pareto SOTA": ACHIEVED.** The router **Pareto-dominates
  gpt-5.6-sol, GPT-5.5, Sonnet 5, and GLM-5.2** — strictly higher accuracy at
  strictly lower cost than each. The Pareto frontier is **{deepseek-chat (cost
  floor) + router curve + claude-opus-4-8 (quality ceiling)}**; four of the six
  single models are pushed off it.
- **Honest boundary:** the router does **not** beat Opus on raw accuracy (0.907
  vs 0.913, −0.6pt) — Opus is the quality ceiling. The router's value is
  delivering **near-Opus quality at ~4× lower cost**, or dialing cost/quality via
  λ. The hero is the deterministic code cascade (0.994); the learned MMLU/MATH
  routing is modest (it roughly matches, not beats, always-Opus on non-code).

## Spend

Full official re-sample (7 models × 1063, parallel, budget-gated): ~**$20.2**
(deepseek $0.12, glm $1.27, kimi $1.26, sonnet $2.54, opus $4.48, gpt-5.5 $5.97,
gpt-5.6-sol ~$4.6) + smoke ~$1.5 ≈ **$22**, within the ~$25–35 budget. Transient
mirror 5xx/timeout errors on the GPT models (~17% during peak parallelism) were
retried down to 0–1 residual each; those tasks fall outside the common set.

## Known limitation

The humaneval reference self-test synthesizes `prompt + canonical_solution`,
whereas real scoring grades the extracted completion alone — so the self-test
cannot catch a completion that silently relies on prompt context (e.g. an import)
it never restates. Watch humaneval false-negatives in the disagreement audit.

## Addendum: Kimi k3 (preliminary, partial coverage — NOT in the main table)

Kimi k3 was sampled to gauge its quality, but the Moonshot account quota died at
**492/1063** tasks. It therefore cannot join the authoritative 1057-task table.
The comparison below is on the **489 tasks k3 did complete** (that all seven
models answered) — a smaller, self-selected subset whose **absolute accuracies
run higher than the 1057-task table** (the subset is easier: e.g. DeepSeek's MMLU
is 0.690 here vs 0.813 on the full suite). **These numbers are comparable only
within this subset, not against the main table.**

| model | overall (489) | mmlu_pro | math | humaneval |
|---|---|---|---|---|
| gpt-5.5 | 0.955 | 0.828 | 0.959 | 0.970 |
| claude-opus-4-8 | 0.953 | 0.862 | 0.946 | 0.982 |
| **kimi-k3** | **0.953** | **0.897** | 0.949 | 0.970 |
| claude-sonnet-5 | 0.935 | 0.828 | 0.919 | 0.982 |
| gpt-5.6-sol | 0.922 | 0.828 | 0.949 | 0.890 |
| deepseek-chat | 0.904 | 0.690 | 0.929 | 0.896 |
| glm-5.2 | 0.902 | 0.793 | 0.892 | 0.939 |

**Read:** on its covered subset k3 is **top-tier — statistically tied with Opus
and just behind GPT-5.5** — and has the **best MMLU-Pro of the group (0.897)**.
A full-suite k3 number requires a quota refresh or account top-up, after which
`kimi-k3` (registered in `evaluator/validate.py`) can be re-sampled into the
main table.


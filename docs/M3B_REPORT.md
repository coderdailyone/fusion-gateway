# M3b — Cost-Aware Router Report

> **⚠️ Superseded by M2c.** This 2-model (DeepSeek+Sonnet) experiment's numbers
> were computed on a buggy math scorer and are **retracted**. The authoritative,
> official-scored, 6-model result is in **`BENCHMARK_REPORT.md`** (M2c): on
> correct scoring the router @λ=10 reaches 0.907 @ $0.00106 and Pareto-dominates
> gpt-5.6-sol / GPT-5.5 / Sonnet / GLM; Opus (0.913) is the accuracy ceiling.
> The inline ⚠️ markers below flag the specific bug-inflated claims.

**⚠️ RETRACTED (bug-inflated; pending M2c official recompute): Headline: the
learned dynamic policy Pareto-dominates the best single model.**
On the full 1063-task objective suite, routing DeepSeek (cheap) ↔ Claude Sonnet 5
(strong) — with a deterministic verify-cascade for code and a learned classifier
for math/MCQ — reaches **⚠️ RETRACTED (bug-inflated; pending M2c official
recompute): higher accuracy than always-Sonnet at ~2.6× lower cost**,
and beats always-DeepSeek on accuracy. `always-Sonnet` is pushed off the Pareto
frontier.

## Learnability gate (free — on M3a pilot data)

Before paid sampling, we tested the router's premise: can public features
(problem TF-IDF + source + length) predict each model's per-task correctness?
150-task pilot, group-by-task CV, per-fold featurizer refit.

| model | CV AUC |
|---|---|
| deepseek-chat | 0.66 |
| claude-sonnet-5 | 0.50 |

**GO** — DeepSeek's *failures* are predictable (0.66), so the router can learn
"escalate when the cheap model is likely wrong." (Sonnet's correctness is
near-unpredictable at 0.50, which is fine — routing only needs to predict when
the cheap model fails.)

## Full run (1063 tasks × DeepSeek + Sonnet, ~$2.27)

Pool: DeepSeek (cheap) + Claude Sonnet 5 (strong). Router CV AUC on the full data:
DeepSeek **0.68**, Sonnet 0.58. Split: 899 math/MCQ routed by the learned policy +
164 code by the deterministic verify-cascade.

### ⚠️ RETRACTED (bug-inflated; pending M2c official recompute): Cost–quality points

| strategy | accuracy | cost / task |
|---|---|---|
| always DeepSeek | 0.849 | $0.00010 |
| always Sonnet (best single) | 0.867 | $0.00204 |
| **learned router @ λ=3** | **0.892** | **$0.00077** |
| learned router @ λ=10 | 0.891 | $0.00061 |
| learned router @ λ=1e6 (cheapest) | 0.852 | $0.00012 |
| oracle (upper bound) | 0.919 | $0.00023 |

### Result

- **Router beats the best single model on BOTH axes:** at λ=3, **⚠️ RETRACTED
  (bug-inflated; pending M2c official recompute): +2.4pt accuracy
  over always-Sonnet (0.892 vs 0.867) at ~2.6× lower cost** ($0.00077 vs $0.00204).
  → the dynamic policy **Pareto-dominates Sonnet** (envelope check: `dominated =
  [claude-sonnet-5]`).
- It also **beats always-DeepSeek on accuracy** (0.892 vs 0.849) but costs **~7×
  MORE** than DeepSeek ($0.00077 vs $0.00010) — a genuine quality/cost trade-off,
  NOT a free win. The router is **never** cheaper than always-DeepSeek: DeepSeek is
  the absolute cost floor ($0.0001 — nothing beats always-using-the-cheapest). So
  the real Pareto frontier is **{DeepSeek endpoint} + {router curve}**, and
  always-Sonnet is off it.
- vs the historical DRACO-20 result ("dynamic routing never beat the best static
  policy on quality") — this **does** beat it, because the pool finally has a
  genuine cost–quality spread and code is runtime-verifiable.

### Honest attribution (what's doing the work)

- **The deterministic code verify-cascade is the hero** (164 tasks): run DeepSeek,
  run its code against the tests, escalate to Sonnet only on failure → near-oracle
  quality on code at near-DeepSeek cost, no learning needed.
- **The math/MCQ learned router is modest** (CV AUC 0.58–0.68): it captures *some*
  of the "escalate when DeepSeek likely wrong" signal, not all. The oracle→router
  gap (0.919 → 0.892) is the room a better router/features could still recover.
- Net: the win is real and deployable, driven mostly by verifiable-task cascading
  plus a modest learned escalation on non-verifiable tasks.

## Acceptance

- ✅ Dynamic policy envelopes the strong static baseline (Sonnet) and beats the
  cheap one on quality; always-Sonnet is off the frontier.
- ✅ Honest non-oracle numbers reported (router 0.892 vs oracle 0.919).
- ✅ group-by-task CV; features public-only; evaluator isolated from the gateway.
- Total spend: **~$2.27** (DeepSeek $0.10 + Sonnet $2.16).

## Not done here (deliberately)

- **Opus 4.8 / gpt-5.6-sol legs:** deferred. Merged PR #1's behavioral eval found
  Sonnet-5 ≈ Opus-4.8 on objective correctness, so Opus likely adds no quality
  signal here (redundant, like GLM was to DeepSeek). Adding them (~$6, ~4h) would
  enrich the frontier only if a *cheaper-yet-sometimes-better* strong model exists —
  worth a small check, not a blind full run.
- RouteLLM external comparison (M5); wiring the router into the online gateway (M6).

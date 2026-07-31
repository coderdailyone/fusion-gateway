# M6 — Self-Consistency Resampling + Objective Voting: Result

**Status:** stopped at the subset gate 2026-07-26 · **Milestone:** M6
**Question:** does sampling each domestic model 10× and aggregating by objective
rules beat claude-opus-4-8 (0.913) — with no frontier model in the loop?

## TL;DR

**No, and the full run was not worth its cost.** On a 150-task stratified subset
at k=10 (4500 samples, ~$4.6), objective voting peaks at **0.9133 at k=5** — only
**0.7 points (1 task) above the best single domestic model** (kimi-k3, 0.9067) —
and *declines* to 0.9000 at k=10. The spec's stop rule technically passed (the
10-sample oracle rose 0.9308 → **0.9733**), but the thing that rule was protecting
against happened anyway one layer down: **the answers are in the pool and the
aggregator cannot extract them.** The full 1063-task run ($32, ~100 h) was
cancelled on this evidence.

## The decisive numbers (150 tasks, k=10, official scoring)

| k | oracle | voting accuracy | tie rate |
|---:|---:|---:|---:|
| 1 | 0.9533 | 0.8933 | 11.3% |
| 3 | 0.9667 | 0.9067 | 4.7% |
| **5** | 0.9667 | **0.9133** ← peak | 4.7% |
| 10 | **0.9733** | 0.9000 ← *declines* | 2.0% |

| single model (sample 0) | accuracy |
|---|---:|
| kimi-k3 | 0.9067 |
| glm-5.2 | 0.8800 |
| deepseek-chat | 0.8733 |

Within-model diversity at k=10: **1.79 distinct ballot keys** per (task, model),
up from 1.05–1.10 measured at k=3 in the smoke.

## What this shows

1. **Resampling works as advertised — the premise was right.** Diversity rose
   (1.05 → 1.79 distinct keys) and the oracle rose with it (0.9308 → 0.9733).
   More samples genuinely surface answers the 3-candidate panel did not have.
2. **Voting does not convert that into accuracy.** Peak voting (0.9133) beats the
   best single model by exactly one task. The gap between the oracle (0.9733) and
   the vote (0.9133) is **6 points that the aggregator cannot reach** — the same
   failure M5 hit with an LLM fuser, now reproduced with objective plurality.
   Changing the aggregation *mechanism* did not change the outcome.
3. **Marginal return dies at k=5, then goes negative.** k=5 → k=10 *loses* 1.3
   points while the oracle still rises. This is the amplification failure the
   spec warned about: extra samples concentrate votes on confidently-wrong
   answers faster than they rescue right ones. More compute is the wrong lever.
4. **The stop rule was necessary but not sufficient.** It asked "did resampling
   surface new answers?" (yes) rather than "can voting use them?" (no). A future
   gate should test the *extraction*, not the ceiling.

## Decision: full run cancelled

Extrapolating 150 tasks to 1063 at $32 and ~100 hours of wall time, to land near
0.90–0.91 — under opus's 0.913 and within noise of kimi-k3 alone — is not a
justified spend. Total M6 outlay: **~$4.8** (probe + smoke + subset), versus
$32 planned.

## What the two fusion milestones together establish

| approach | result vs best single domestic |
|---|---|
| M5 — LLM fuser over 3 candidates | 0.8975 vs 0.8905 (+0.7pt, p=0.39, n.s.) |
| M6 — objective voting over 30 samples | 0.9133 vs 0.9067 (+0.7pt, 150-task subset) |

Two independent aggregation mechanisms — a subjective LLM rewriter and an
objective plurality vote — converge on the **same +0.7-point ceiling** over the
best pool member, while the pool's oracle sits 4–6 points higher. That
consistency is the finding: **on this pool the limit is not the aggregator's
design, it is that the models' errors are correlated.** When they are wrong, they
are wrong together (M5 measured 79% unanimity on mmlu_pro at 0.851 accuracy), and
no vote-counting rule can recover an answer nobody produced correctly *and
distinguishably*.

## The honest answer to the original goal

The goal was "several domestic models scored and fused to reach GPT-5.6 / Opus-5
level."

- **GPT-5.6 level: reached — but not by fusion.** kimi-k3 alone scores 0.9067 on
  this subset (0.884 on the full standard tier), against gpt-5.6-sol's 0.894.
- **Opus-5 level (0.913): not reached.** Best fusion/voting is ~0.9133 on a
  150-task subset (±0.05 at this n) and 0.897 on the full tier — at or below opus,
  never significantly above it.
- **Fusion's measured contribution is ~+0.7 points and not significant** in
  either mechanism.

## Where the remaining headroom actually is

The oracle says 4–6 points sit unclaimed. Two levers remain, neither of which is
"more samples" or "a better fuser prompt":

1. **Decorrelate the pool.** Every gain here is bounded by how often the models
   fail *together*. Adding a genuinely different model — different pretraining
   lineage, not another Chinese chat model tuned on similar data — raises the
   oracle in a way resampling cannot.
2. **Objective verification where it exists.** Voting on code was the weakest
   branch precisely because only 45.7% of HumanEval problems (and 0% of
   LiveCodeBench) expose public doctests to check against. The project's proven
   0.994 code result came from executing *real* tests, not from counting votes.
   Expanding objective checks beats expanding ballots.

## Reproducibility

Frozen samples: `evaluator/runs/m6_consistency/subset150/k{0..9}_s{0..3}/`
(4500 samples, 3666 ok). Re-voting and re-scoring cost **$0**.
Driver `scripts/run_consistency.py` (`probe` | `sample` | `report`);
voting machinery `evaluator/consistency/{normalize,vote,sampler}.py`;
metrics helper `runs/m6_scratch/decide.py`.
Design: `docs/superpowers/specs/2026-07-26-self-consistency-voting-design.md`.

**Temperature note:** deepseek-chat and glm-5.2 honour `temperature=0.8`;
**kimi-k3 rejects it outright** (HTTP 400, "only 1 is allowed for this model") and
was sampled without it, relying on its own sampling variance — it still produced
distinct outputs, so its ballots are real votes, but the legs are not
temperature-matched.

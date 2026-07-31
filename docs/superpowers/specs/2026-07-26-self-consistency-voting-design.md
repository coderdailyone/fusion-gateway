# M6 — Self-Consistency Resampling + Objective Voting

**Status:** design approved 2026-07-26
**Milestone:** M6 (fusion family; supersedes M5's LLM-fuser approach for the
cases where an objective rule exists)

## Why

M5 established two things. First, **a domestic panel already contains
frontier-grade answers**: the 3-model oracle is 0.9308 against opus's 0.913.
Second, **an LLM fuser cannot extract them**: `fix` and `break` are coupled —
adding a majority-copy rule cut `break` 25→21 but dropped `fix` 32→28, leaving
net gain flat at +7, and a stronger solver (kimi-k3) made a *worse* fuser.

A free probe then showed the sharper problem: on mmlu_pro, **plain majority
voting over the 3 existing candidates scores 0.8514 — identical to the LLM
fusion's 0.851**. The expensive fuser adds nothing over free voting. And
**475/599 (79%) of mmlu_pro tasks already have unanimous agreement**, so with
only 3 votes the ballot is nearly saturated: there is almost nothing left to
aggregate.

M6's thesis: **the missing ingredient is independent samples, not a smarter
judge.** Sampling each model 10× turns 3 correlated opinions into a real
distribution, and aggregating that distribution with **objective rules** (letter
counts, math equivalence, test execution) avoids the LLM-rewrite trap entirely.

**Goal:** beat claude-opus-4-8 (0.913) on the standard tier using only domestic
models.

## Positioning (locked with the user)

- **Optimize for accuracy, not cost.** Compute is not a constraint this
  milestone; a budget ceiling exists only as a runaway backstop.
- **Domestic only.** deepseek-chat, glm-5.2, kimi-k3 — in the panel, in the
  tie-break fuser, everywhere. No frontier model in any role.
- **10 samples per model per task, all 1063 tasks** (≈32k calls): 30 votes/task.
- **Objective voting first; the LLM fuser is a tie-break of last resort.**
- Same disciplines as M2c/M2d/M5: official scorers only (no LLM judge), frozen
  re-scorable outputs, `evaluator/` isolated from `gateway.*`, secrets only in
  `runs/secrets/.env`, both manifests byte-unchanged.

## Baselines to beat (M2c/M5, same suite, same graders)

| | accuracy |
|---|---:|
| **claude-opus-4-8 — the target** | **0.913** |
| gpt-5.5 | 0.905 |
| claude-sonnet-5 | 0.898 |
| M5 LLM fusion | 0.897 |
| gpt-5.6-sol | 0.894 |
| kimi-k3 (best single domestic) | 0.884 |
| *3-candidate panel oracle (M5 ceiling)* | *0.9308* |

## Architecture — `evaluator/consistency/`

Three units; none imports `gateway.*`.

```
evaluator/consistency/
  sampler.py    # k samples per (task, model) at temperature T; every sample frozen
  normalize.py  # answer -> comparable "ballot key", per source type
  vote.py       # aggregate ballots by objective rules -> final answer + tally
```

**Data flow per task:** sample (3 models × 10) → normalize each sample to a
ballot key → vote → final answer → scored by the **existing official scorers**
and frozen.

## Voting rules (objective, per source)

| source | normalization | rule |
|---|---|---|
| `mmlu_pro` | official letter extraction (`evaluator/official/mmlu_extract`) | plurality over letters |
| `math` | `math_equiv` equivalence classes (`0.5` and `1/2` are one candidate) | plurality over classes |
| `humaneval` | **run the task's public tests in the sandbox** | passing samples win; among passers, plurality by text; if none pass, plurality by text |

The code rule is the important one: **do not vote on code, execute it.** This is
the project's proven verify-cascade move (0.994 on code) and uses the task's
*public* tests — never the hidden grader.

## Tie-break (the only LLM in the loop)

A genuine tie (e.g. 12–12) or a task where no ballot can be formed falls through
to the existing `evaluator/fusion/fuse.py` fuser (glm-5.2, domestic), which
receives the candidates **plus the vote tally**. Expected to fire rarely — with
3 votes 79% of mmlu_pro is already unanimous, and 30 votes makes exact ties
rarer still. The tie-break rate is reported.

## Sampling parameters

- **k = 10 per model per task**, all 1063 tasks, all three sources.
- **temperature 0.8** for deepseek-chat and glm-5.2 — enough diversity without
  degenerate output. **kimi-k3 is a reasoning model and may ignore or reject a
  temperature parameter**; the implementer probes this first and, if it does,
  relies on the model's own sampling variance and records that in the report.
- Sampling is sharded and resumable (the M2c/M5 pattern); every sample frozen.
- **Budget ceiling $40** as a runaway backstop, not a target.

## Yield curve and stop rule (free, from frozen samples)

Because every sample is frozen, the vote can be replayed with the first 1, 3, 5,
and 10 samples at **$0**, producing a **votes-vs-accuracy curve**. This:

1. shows where marginal returns die (if 5→10 is flat, 5 is the right k);
2. gives the **stop rule**: if the **new 30-sample oracle does not exceed
   0.9308**, resampling produced no answers the panel did not already have —
   stop and report, rather than pushing further compute at it.

**The failure mode to watch:** self-consistency amplifies *systematic* error.
mmlu_pro is 79% unanimous yet only 0.851 — the models are often confidently
wrong *together*, and more samples will make such ballots look more certain, not
less. The report therefore tracks the count of **"unanimous but wrong"** tasks
before and after; if that count does not fall, this technique is the wrong tool
for those tasks and the next lever is per-task compute (longer reasoning on
disputed items), not more samples.

## Report — `docs/M6_VOTING_REPORT.md`

- Voting accuracy with **Wilson 95% CI**, versus opus 0.913, M5 fusion 0.897,
  and kimi-k3 0.884; **McNemar** against opus and against M5 fusion.
- New 30-sample oracle vs the M5 3-candidate oracle (0.9308).
- The votes-vs-accuracy yield curve (k = 1, 3, 5, 10).
- Per-source breakdown (mmlu_pro / math / humaneval).
- Tie-break rate, spoiled-ballot rate, "unanimous but wrong" before/after.
- Verdict: does a domestic pool beat opus under objective voting.

## Error handling

- A failed/timed-out sample is discarded and counted; the vote proceeds on the
  remaining samples (no re-sampling to fill gaps).
- A task whose samples all fail falls back to the M5 frozen candidates and is
  flagged.
- An answer the extractor cannot parse is a **spoiled ballot** — discarded from
  the tally and counted in the spoiled-ballot rate.
- Sandbox timeout/exception on a code sample ⇒ that sample counts as
  "not passing" and voting continues.
- Transient mirror 5xx/timeout ⇒ prune-and-retry, resumable (M2c pattern).
- Budget ceiling reached ⇒ stop cleanly; frozen samples stay re-votable at $0.

## Testing

- Unit: `normalize` for each source (letter extraction; `math_equiv` merging
  `0.5`/`1/2` into one class; code pass-signature), including unparseable input.
- Unit: `vote` — plurality, exact tie detection, all-spoiled ballots, and the
  code rule preferring a passing sample over a more numerous failing one.
- Unit: yield-curve slicing (first-k sub-vote) over fixture ballots.
- Isolation test: `evaluator/consistency/*` imports no `gateway.*`.
- Determinism: re-voting frozen samples twice is identical.
- All unit tests stub the completion fn — no network.

## Acceptance criteria

1. 10 samples per model per task sampled for all 1063 standard-tier tasks,
   frozen, resumable, under the $40 backstop; sampling failures reported.
2. Objective voting implemented per source (letters / math-equivalence /
   test execution), with the LLM tie-break firing only on genuine ties.
3. Yield curve (k = 1, 3, 5, 10) and the new oracle computed at $0; the stop
   rule is evaluated and its outcome recorded.
4. `docs/M6_VOTING_REPORT.md` published with Wilson CIs and McNemar vs opus
   (0.913) and vs M5 fusion (0.897), the per-source breakdown, tie-break and
   spoiled-ballot rates, and the "unanimous but wrong" before/after count.
5. No frontier model in any role; `evaluator/` isolated; manifests unchanged;
   scoring deterministic and re-runnable at $0.

## Non-goals

- Not revisiting M5's conclusions or artifacts; M6 is an additional strategy.
- Not wiring voting into the live gateway (a later milestone, gated on M6).
- No LLM judge in scoring; the tie-break fuser produces an *answer*, never a grade.
- Hard tier (657 tasks) is out of scope until the standard-tier verdict lands.

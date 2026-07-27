# M4 — SWE-bench-Live Agentic Tier: Pilot Report

**Status:** pilot complete 2026-07-24 · **Milestone:** M4 (long-task routing)
**Question:** does a *task-level escalation cascade* (cheap model → proxy verifier
→ escalate to strong on failure) reach near-strong resolve rates at materially
lower **cost per successful task** on real long-horizon coding tasks?

## TL;DR

On a 20-instance sample of **SWE-bench-Live (verified split)**, graded by the
**official SWE-bench harness** (do the hidden `FAIL_TO_PASS`/`PASS_TO_PASS` tests
pass), the deepseek→opus cascade delivered **89% of opus's resolves (8 vs 9) at
71% of opus's cost, i.e. ~20% lower cost-per-successful-task** — the routing
thesis, extended from single-turn to agentic long tasks, holds directionally.
The **self-report escalation gate is the bottleneck** (precision 2/6) and is the
clear next improvement. Sample is small and execution was lossy; this is an
indicative pilot, not a leaderboard number.

## Result

| configuration | resolved | total cost (est.) | **cost / successful task** |
|---|---:|---:|---:|
| deepseek-chat (cost floor) | 2 | $0.18 | $0.09 |
| claude-opus-4-8 (quality ceiling) | **9** | $11.29 | $1.26 |
| **cascade (deepseek → opus)** | **8** | $8.03 | **$1.00** |

- **Cascade vs opus:** 8/9 resolves at $8.03 vs $11.29 → **−20% cost per
  successful task** ($1.00 vs $1.26). Near-ceiling quality, materially cheaper —
  the intended Pareto move.
- **Cost accounting:** the cascade runs deepseek on all instances ($0.18) and
  opus only on the escalated ones (9 of 15 → $7.84), not opus-on-everything.
- **deepseek alone** is cheap but weak in the agent loop (2 resolves) — it
  frequently failed to produce a well-formed submission (see caveats).

## How the cascade routes (escalation trigger)

Per instance: **deepseek runs the whole task** via SWE-agent; a **proxy
verifier** decides pass/fail; only on **fail** does opus re-run the whole task.
The verifier **never reads the hidden grader** (`FAIL_TO_PASS`) — that would be
cheating. In this pilot the gate was the pragmatic **self-report signal**:
*deepseek exited `submitted` with a non-empty patch* → keep deepseek; else
escalate. Of 15 completed instances, **6 kept deepseek, 9 escalated to opus.**

## The verifier is the bottleneck

The self-report gate's **precision was 2/6**: of the six deepseek patches it kept,
only two actually resolved. The four false-keeps cost the cascade real resolves —
concretely, `Kozea__Radicale-1766` was solved by opus but the cascade wrongly
kept deepseek's unresolved patch, which is the entire 8-vs-9 gap. A stronger gate
directly lifts the cascade toward (and potentially past, at lower cost) the opus
ceiling. **The designed repro-test-first verifier** (agent authors a reproduction
test; keep only if it goes red→green with no regression) is the intended fix and
the top follow-up.

## Execution caveats (honest)

- **Small, lossy sample.** 20 instances attempted; 15 produced a gradeable
  prediction from at least one model. Losses: 3 image-build races (images not
  pre-pulled → concurrent pull contention through the China registry mirror; now
  fixed by pre-pulling), 2 command-timeouts on large repos, and — notably —
  deepseek repeatedly hitting `exit_format` (the cheap model struggles with
  SWE-agent's strict submission format). N this small is directional only.
- **Contamination.** The `verified` split spans 2024-07…2025-04, which predates
  the pool models' training cutoffs, so this pilot does **not** establish
  contamination resistance — it validates the *cost/quality routing mechanics*.
  A real de-contaminated run should sample the freshest monthly `test` split.
- **Cost units.** opus cost is litellm's estimate at Anthropic list price; the
  actual mirror charge is likely lower. deepseek cost is real. The **ratio**
  (cascade ~20% cheaper per success) is robust to this; absolute dollars are an
  upper bound.

## Infrastructure (reproducibility)

Dedicated eval box, **rootless Docker** (isolated from the host's other Docker),
image store on a 3.1 TB SSD. SWE-bench-Live instance images pulled from DockerHub
`starryzhang/` via a registry mirror; dataset via an HF mirror; agent = **SWE-agent
1.1.0** driving the models through LiteLLM (deepseek direct, opus via an
Anthropic-compatible mirror + proxy); grading = the official swebench fork. All
model calls, patches, and trajectories are frozen for zero-cost re-grading.

## Verdict & go/no-go

**Directionally GO on the thesis, but upgrade before scaling.** The pilot proves
the pipeline end-to-end and shows the cascade beating the strong model on
cost-per-successful-task. Before a fuller run:

1. **Implement the repro-test-first verifier** (the self-report gate's 2/6
   precision is the main quality leak).
2. **Harden execution:** pre-pull images, raise per-command/instance timeouts,
   and improve the cheap model's submission-format reliability (or swap the cheap
   leg).
3. **Then** run a larger, freshest-month (de-contaminated) sample and report
   with Wilson CIs + McNemar significance.

## Sources

- Standard/hard-tier context: `docs/BENCHMARK_REPORT.md`, `docs/HARD_TIER_REPORT.md`
- Positioning (cost per successful task): `docs/POSITIONING.md`
- Design: `docs/superpowers/specs/2026-07-22-swebench-live-agentic-tier-design.md`

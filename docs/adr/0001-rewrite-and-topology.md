# 0001. Full Rewrite of the Gateway Core, and VPS/Cookie Compute Topology

## Status

Accepted — 2026-07-14

## Context

Two prior lines of work had each hit a ceiling, and neither, taken as-is,
was the right foundation to build forward from:

- **the prior online-routing research line** (`the GPU research workspace` on the cookie GPU
  machine) is the active research line. Real DRACO-20 experiments proved
  that dynamic routing can cut cost 24–26% and latency 13–14% versus static
  baselines, but quality never beat the best static single-model policy
  within the model pool. Two blockers were named: (1) fast-judge variance
  too high — fixed-output repeat scoring reached only 0.63–0.74
  sign-agreement, leaving fusion-gain labels untrainable; and (2) the
  white-box routing representations (Qwen-0.5B last-token/256) had only
  been validated at n=20 pilot scale. This line proves an idea and carries
  hard-won methodological lessons, but lacks production engineering rigor
  and a validated policy to serve traffic with.
- **the prior offline control-plane line** (VPS image, paused) is an offline control
  plane with strong engineering discipline — event-sourced and replayable
  trace, a cost ledger, 58 tests, reviewer-governed — but its design center
  is offline execution plus a human approval gate, which is the opposite of
  the 7×24 online-serving goal this project needs. This line has the rigor
  but the wrong shape for online serving.

Master decided, in conversation on 2026-07-14, that the right move is not
to patch or merge either codebase, but to converge both lines into one new
system: a self-use production gateway plus a reproducible cost-quality
benchmark report, targeting cost-quality Pareto SOTA.

## Decision

**D1 — Full rewrite of the gateway core.** What is carried forward is
discipline, not code:

- From the prior offline control-plane line: event-sourced and replayable trace design,
  "a STOP event must carry an explicit stop condition," and cost ledger
  semantics (preflight before any real call, settle after).
- From the prior online-routing research line's own discipline (its AGENTS.md): leakage rules
  (judge/rubric data never enters routing input; candidates only ever see
  public `id`/`domain`/`problem`-class fields), group-by-task validation,
  and secrets handling.
- These are captured as full prose in `docs/DISCIPLINES.md` (Task 2 of the
  M0+M1 plan), so that every later task in this project can cite the rules
  directly without depending on either legacy repository remaining
  available or unfrozen.

**Old lines are demoted, not deleted:**

- the prior online-routing research line's run directories and frozen outputs — the DRACO-20
  experiment artifacts living on the cookie GPU machine under
  `the GPU research workspace` — are kept exactly where they are, untouched,
  and are repurposed as evaluation and replay assets. Future benchmark work
  may consume them read-only for comparison; they are not a live codebase
  to extend, and they must not be mutated.
- the prior offline control-plane line stays paused, **permanently**. It is not
  scheduled to be unfrozen or resumed as a codebase under this project. It
  survives only as a discipline source (already captured in full in
  `docs/DISCIPLINES.md`) and as reference material for anyone who needs to
  understand the original design intent. Any future desire to unfreeze it
  is superseded by this decision — recording that supersession is a
  primary purpose of this ADR.

**Topology:**

- The gateway process binds `127.0.0.1:8800` on the VPS (production host:
  4 core / 8G) and serves real OpenAI-compatible traffic 7×24. It is
  reached only through VPS-local services or an SSH tunnel, and is never
  exposed directly to the public internet.
- Training, batch evaluation, and heavy or repeated experiments run on the
  cookie GPU machine (RTX 3050 4GB). The cookie machine never serves live
  production traffic.
- The laptop is a development and orchestration surface only; it never
  serves 7×24 production traffic and nothing on the serving path may
  depend on it being online.

## Consequences

- **No code reuse from either legacy line.** Every module under `gateway/`,
  `evaluator/`, and `router_training/` is written fresh against this
  repo's own spec and tests; nothing is imported or copy-pasted from
  `fusion-research` or from the the prior offline control-plane line codebase. This
  costs implementation time that reuse would have saved, but avoids
  inheriting either line's structural mismatch with the 7×24 online-serving
  goal — the prior offline control-plane line's offline/human-gate shape, and Fusion
  Research's unvalidated-at-scale policy and unreliable judge.
- **Disciplines are ported as text, not as an executable dependency.**
  `docs/DISCIPLINES.md` is the durable artifact going forward. Later tasks
  (from M2 onward, per the M0+M1 plan) implement the actual enforcement —
  leakage tests, the group-by-task CV harness, judge repeat-agreement
  checks — natively in this repository, rather than importing test code
  from either legacy line.
- **DRACO-20 runs and other the prior online-routing research line outputs are kept frozen on the
  cookie machine as evaluation assets.** They remain available for
  comparison and replay, but are explicitly not a base to build the
  router/training pipeline on top of, and must not be mutated by this
  project's work.
- **the prior offline control-plane line's codebase is off this project's critical
  path** under every milestone in the current plan. Reviving it as active
  development would require a new ADR that explicitly supersedes this one.
- **Compute placement is now a load-bearing constraint on every later
  task.** Any task whose design would make the gateway's serving path
  depend on cookie-machine or laptop availability is out of spec under
  this ADR and must be rejected or escalated rather than implemented as
  proposed.

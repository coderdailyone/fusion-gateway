# M4 — SWE-bench-Live Agentic Tier (long-task routing)

**Status:** design approved 2026-07-22
**Milestone:** M4 (extends M3 router + M2d hard tier to long-horizon agentic tasks)

## Why

Every existing tier (standard M2c, hard M2d) is **single-turn QA**: one prompt,
one answer, objective grade. OpenAI's scorecard (see `docs/POSITIONING.md`)
frames the real metric as *cost per successful **real-work** task* — customer
issues resolved, PRs shipped — and the single-turn frontier is saturating. The
last mile is a **long-horizon, agentic, contamination-resistant** tier.

**SWE-bench-Live** is the bridge: real GitHub issues, **timestamped /
de-contaminated** (monthly snapshots of issues created after model cutoffs),
scored by *do the hidden tests pass* — the same execute-to-verify paradigm our
code verify-cascade already uses. This milestone extends the project's signature
move — **"escalate appropriately"** (the deterministic verify-cascade that gave
0.994 on code at ~$0.0005/task) — from single-turn code to a full agent loop.

**The thesis under test:** a *task-level escalation cascade* (cheap model
attempts the whole task; a cheap proxy verifier gates; escalate the whole task
to a strong model only on failure) reaches **near-strong-model resolve rates at
materially lower cost-per-successful-task** on real long tasks. That is the
router's value proposition, measured on real work instead of a proxy.

## Positioning (locked with the user)

- **This is a routing experiment on long tasks**, not just a model benchmark.
  The deliverable is the cascade's cost-quality vs single-model baselines.
- **Pilot first, gated.** Build the full pipeline, run a **~30-task pilot**
  end-to-end, measure (resolve rate / cost-per-successful-task / escalation rate
  / verifier-vs-grader agreement), then decide whether to run the full split.
  Matches the project's "cheap/free gate before paid spend" discipline.
- **Same disciplines as M2c/M2d/M3:** `evaluator/` isolated from `gateway.*`,
  frozen re-gradeable outputs, official upstream grading, preflight budget gate,
  secrets only in `runs/secrets/.env`. The standard (`configs/suite.manifest.json`)
  and hard (`configs/suite.hard.manifest.json`) manifests are **byte-unchanged**.
- **Task-level cascade only** this milestone. Per-step (mid-loop model swap)
  routing is explicitly deferred — it breaks context continuity and has no clean
  per-step escalation signal, and it is not needed to test the thesis.

## Execution environment (a hard prerequisite)

SWE-bench-Live needs Docker: each instance runs its repo + hidden tests in a
per-instance container, and the agent loop runs commands in that container while
solving. This dev box has **no Docker**. Decision: a **new dedicated eval box**
(≥16 GB RAM / ≥100 GB disk, Docker installed), user-provisioned, SSH access
given to the implementer. It is isolated from the production VPS (which runs
Prism/crabot and must not be touched) and does not share its resources. The box
pulls SWE-bench-Live's official **pre-built per-instance images** (the ~tens of
GB of disk). All commands the agent runs, and the final grading, happen there —
never on the dev box, never on the production VPS. The pipeline is written to be
box-agnostic: "given one Docker host, it runs"; the host is a config dependency.

## Architecture — four isolated units under `evaluator/agentic/`

None import `gateway.*`. Each has one job and a typed interface.

```
evaluator/agentic/
  runner.py     # drive SWE-agent on ONE instance with ONE model → (patch, trajectory, cost)
  verifier.py   # proxy verifier: "did this attempt pass?" (NEVER the hidden grader) → bool + evidence
  cascade.py    # orchestrate: cheap.run → verify → pass? keep : strong.run(fresh) → keep; sum cost
  grade.py      # write predictions.jsonl → official SWE-bench-Live harness (Docker) → resolved map
```

**Data flow (one instance):**
1. `runner.run(instance, "deepseek-chat")` → SWE-agent solves in the instance's
   Docker container, emits a `git diff` patch + trajectory + measured cost.
2. `verifier.verify(instance, patch, trajectory)` → runs in the same container:
   patch non-empty **AND** the agent-authored reproduction test goes red→green
   **AND** the repo's existing/affected tests don't regress. Returns pass/fail +
   the evidence used (never touches `FAIL_TO_PASS`/`PASS_TO_PASS`).
3. `cascade`: if verifier passes → accept the cheap patch. Else the strong model
   runs the whole task **fresh** (a clean re-attempt, not a resume of the cheap
   trajectory — cleaner cost attribution and matches "escalate the whole task")
   → accept the strong patch. Total cost = cheap + (if escalated) strong.
4. `grade`: collect the accepted patches into `predictions.jsonl`, run the
   **official** SWE-bench-Live evaluation harness → `resolved` per instance.

**Baselines run alongside** (same instances, for the cost-quality comparison):
`deepseek-chat` alone and `claude-opus-4-8` alone. A fresh opus attempt on an
instance *is* exactly the cascade's escalation step, so the pilot runs each model
**once per instance** and **derives** the cascade from the two frozen runs plus
the verifier gate — cascade accepts the cheap patch when the verifier passes,
else the opus patch — with **no redundant third run** (halving strong-model
spend). `cascade.py` still implements the real cheap→verify→strong orchestration
(unit-tested with stubs); the pilot driver wires it over the shared frozen runs.
So one pilot yields three curves — cheap-only, strong-only, cascade — from two
model runs per instance.

**Freezing / reproducibility:** trajectories, patches, and `predictions.jsonl`
are frozen (extending the `evaluator/store` pattern); re-grading is zero-cost.
The SWE-bench-Live snapshot is pinned to one monthly release; only instances
whose issue/PR date is after the models' cutoffs are included (de-contamination).

## The SWE-agent integration (the technical crux)

SWE-agent is driven by LiteLLM, and our model registry
(`evaluator/validate.py` `MODELS`) already wraps LiteLLM with per-model
`api_base`/`api_key`/`max_tokens`. `runner.py` configures SWE-agent to call each
model through its registry mirror, so the *same* endpoints M2c/M2d/M3 used drive
the agent loop. Per-instance cost is read from LiteLLM (`completion_cost`, summed
over the trajectory) and frozen — the same accounting the rest of the evaluator
uses. SWE-agent (Docker per-instance, standard `predictions.jsonl` output) is
used unmodified except for this model wiring; OpenHands is the fallback if
SWE-agent's mirror wiring proves unworkable.

## The proxy verifier (the escalation gate — and the cheat boundary)

**Iron rule:** the verifier must **never** read the official hidden tests
(`FAIL_TO_PASS` / `PASS_TO_PASS`). Those are the final grader; leaking them into
the escalation decision would both cheat and contaminate the metric. The
verifier uses only signals a real autonomous deployment would have:

- **Reproduction test (primary):** as part of solving, the agent authors a test
  that reproduces the issue. Verifier re-runs it: it must fail on the original
  code and pass after the patch (red→green).
- **Regression guard (necessary):** the repo's existing / touched tests must
  still pass after the patch.
- **Non-empty patch (floor):** an empty diff is an automatic fail.

Pass = non-empty patch **AND** repro red→green **AND** no regression. If the
agent could not produce a reproduction test, fall back to regression-guard +
self-report, and **flag** the instance (a weak-signal case).

The verifier is deliberately **imperfect** — it can pass what the hidden grader
fails and vice-versa; that is realistic (a real router escalates on an imperfect
cheap signal). So the pilot **measures the verifier against the official grader**
(precision = of verifier-passes, fraction officially resolved; recall = of
officially-resolved cheap attempts, fraction the verifier caught). A useless
verifier shows up here as a number, not a surprise in production.

## Sampling scope (pilot)

- **~30 instances** from the pinned de-contaminated SWE-bench-Live snapshot,
  stratified across repos so no single repo dominates.
- **Two model runs** per instance (`deepseek-chat`, `claude-opus-4-8`); the
  cascade (deepseek→opus) is *derived* from them plus the verifier gate, so the
  pilot reports three curves — cheap-only, strong-only, cascade — from those two
  runs, with no redundant third run.
- **Budget gate ~$50** for the whole pilot. **First gate (before real spend):**
  a **3-instance paid smoke** that confirms each mirror (deepseek, opus) actually
  sustains SWE-agent's multi-turn tool-use loop and that the Docker box grades a
  known instance correctly. After the smoke, report **measured per-task cost +
  escalation rate**, then spend the remainder only on approval. Resumable +
  transient-error retry as in M2c/M2d.

## Report — `docs/M4_AGENTIC_TIER_REPORT.md`

- Cascade vs `deepseek` alone vs `opus` alone: **resolve rate** (with Wilson 95%
  CI), **cost-per-successful-task**, and **escalation rate**.
- Verifier-vs-grader agreement (precision / recall) and the count of weak-signal
  (no-repro-test) instances.
- Verdict: does the cascade reach near-opus resolve rate at materially lower
  cost-per-successful-task, and **a go/no-go recommendation on the full run**.

## Error handling

- SWE-agent crash / container failure on an instance → that instance is recorded
  as an error row and retried (prune-and-retry, M2c pattern); a runaway cannot
  hang the batch (per-instance timeout + container teardown).
- Mirror can't sustain the multi-turn tool loop → caught by the 3-instance smoke
  *before* real spend; swap the model (glm-5.2 cheap / gpt-5.x strong fallbacks).
- Verifier tooling error (repro test won't run) → treat as verifier-fail with the
  instance flagged, never a crash; the cascade escalates (safe default).
- Official harness disagreement / grading error on an instance → surfaced in the
  report, never silently dropped.
- Budget ceiling hit → sampling stops cleanly at the gate; frozen work is
  re-gradeable with zero new spend.

## Testing

- Unit: `cascade` orchestration (pass→keep-cheap, fail→escalate, cost summed)
  with a stubbed runner/verifier — no network, no Docker.
- Unit: `verifier` decision logic (non-empty ∧ red→green ∧ no-regression) with
  fixture patches; confirm it reads no hidden-test fields.
- Isolation test: `evaluator/agentic/*` imports no `gateway.*`.
- Sanity (on the Docker box): the official harness grades a known instance's
  gold patch as `resolved` before any model runs.
- Determinism: re-grading a frozen `predictions.jsonl` twice is identical.

## Acceptance criteria (pilot)

1. Dedicated Docker eval box stands up; the official SWE-bench-Live harness grades
   a known instance's gold patch to `resolved` (harness sanity).
2. `runner.py` drives **both** deepseek and opus through SWE-agent end-to-end on
   ≥1 instance producing a valid patch (the 3-instance paid smoke passes) under
   the budget gate.
3. Cascade + two single-model baselines run on ~30 de-contaminated instances;
   cheap→verify→escalate→official grade; frozen, resumable, errors retried.
4. `docs/M4_AGENTIC_TIER_REPORT.md` published: cascade vs singles (resolve rate +
   Wilson CI, cost-per-successful-task, escalation rate) + verifier-vs-grader
   agreement + a documented go/no-go on the full run.
5. Isolation held: no `gateway.*` import; standard + hard manifests byte-unchanged;
   secrets only in `runs/secrets/.env`; costs frozen and re-gradeable.

## Non-goals

- **Not the full SWE-bench-Live run** — the full split is gated on the pilot's
  cost/quality result and its own go decision.
- **Not per-step / mid-loop routing** — task-level cascade only this milestone.
- **No LLM judge** — grading is the official hidden-test execution; the verifier
  is a separate, non-grader escalation signal.
- **No gateway or standard/hard-tier changes** — this is a new, isolated tier.
- **Not a production agent product** — a measurement deliverable, like M2d.

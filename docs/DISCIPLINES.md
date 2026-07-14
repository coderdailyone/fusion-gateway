# Disciplines

This document is the durable record of engineering and research disciplines
that fusion-gateway is built on. It is **ported text, not a reference** — the
rules below are written out in full so that every later task in this project
can cite this file directly, without depending on either legacy repository
staying available, unfrozen, or even readable.

## Provenance

Fusion-gateway is a full rewrite (see `docs/adr/0001-rewrite-and-topology.md`)
that converges two prior lines of work, both of which had hit a ceiling on
their own:

- **the prior online-routing research line** (`the GPU research workspace` on the cookie GPU machine)
  — the active research line. Real DRACO-20 experiments demonstrated that
  dynamic routing can cut cost 24–26% and latency 13–14% versus static
  baselines, but quality never beat the best static single-model policy
  within the model pool. Two blockers were identified: (1) fast-judge
  variance was too high — fixed-output repeat scoring only reached
  0.63–0.74 sign-agreement, which made fusion-gain labels untrainable; and
  (2) the white-box representations used for routing features (Qwen-0.5B
  last-token/256) had only been validated at n=20 pilot scale. The
  discipline ported from this line covers leakage, validation methodology,
  and judge reliability — the exact areas where the line's own failures
  were diagnosed.
- **the prior offline control-plane line** (VPS image, paused) — an offline control
  plane with strong engineering rigor (event-sourced and replayable trace,
  a cost ledger, 58 tests) governed by a human reviewer, but whose design
  center — offline execution plus a human approval gate — is the opposite
  of the 7×24 online-serving goal this project needs. The discipline ported
  from this line covers event sourcing, the cost ledger, and secrets
  handling — the parts of its engineering that are correct regardless of
  serving topology.

Neither line is reused as code. What follows is discipline, extracted and
rewritten in full, that every later task in this project must follow.

---

## ① Leakage

Candidate and routing inputs — anything the policy engine, feature
extractor, or candidate-generation code path can see when deciding how to
route or what to send to a model — may only ever contain public fields:
an `id`-class identifier, a `domain`-class tag, and the `problem`-class
content of the task itself. Nothing else is admissible in that pipeline.

Judge outputs, rubric text, and reference or gold answers must never enter
routing inputs, under any code path, at any stage — not the current turn's
inputs, not a historical feature derived from a past evaluation run, not a
cached representation computed from an evaluation artifact. If a value was
produced by, or depends on, the evaluation/judging side of the system, it
is not allowed to flow into the routing/candidate side.

This rule exists because leakage of exactly this kind is what silently
inflates offline numbers until they stop meaning anything online: a router
that indirectly "sees" rubric or answer signal — even through an innocuous
looking derived feature — will show optimistic offline gains that do not
hold once the same policy runs on genuinely unseen input. the prior online-routing research line's
own inability to beat static baselines on quality is entangled with exactly
this class of measurement risk, which is why this discipline is treated as
load-bearing rather than as a checklist item.

Enforcement: from M2 onward, an automated test must fail the build if any
rubric-like field — keys or content resembling `rubric`, `judge_notes`,
`reference_answer`, `gold`, `criteria`, or equivalent — appears anywhere in
what is passed to candidate or routing code paths. The discipline itself
binds immediately: no task before M2 may write code that threads judge or
rubric data into a routing decision, even though the automated check that
catches violations is not wired in until M2's evaluator harness lands.

## ② Event sourcing

The event log is append-only. Events are never updated or deleted in
place; any correction to the record happens by appending a new event, never
by rewriting history. This is what makes the log a source of truth rather
than a mutable status field.

Every event may carry a `parent_seq` pointing at a causally prior event
within the same request, so the full causal graph of a request — not just
its flat chronological order — can be reconstructed from the log alone.
Replaying the same ordered event stream for a request must always produce
the same decision sequence, deterministically. That determinism is the
actual payoff of event sourcing here: it buys audit and reproducibility,
not merely a chronological log of what happened.

STOP and refusal events must carry an explicit, machine-readable stop
condition — for example `token_budget_exhausted`, `upstream_error`,
`policy_refusal`, or `user_cancelled` — recorded in the event payload.
Token budget exhaustion **alone is not a valid stop reason**: running out
of tokens is a resource-exhaustion fact, not a decision, and recording it
as if it were a self-explanatory stop reason destroys the ability to later
distinguish "the system chose to stop here" from "the system was cut off
here." Any code path that halts generation or a call chain because of a
budget must pair that fact with the actual policy decision that followed
from it (e.g., "budget exhausted, and policy says stop rather than retry
with a smaller model") and record both.

This discipline is ported from the prior offline control-plane line, where the
event-sourced, replayable trace design was sound engineering independent
of that line's offline/human-gate mismatch with the current serving goal.

## ③ Ledger

Every real, billable provider call must have a ledger **preflight** row
inserted before the call is made, carrying an estimated cost computed from
the call's expected token usage and the model's price. The call is only
allowed to proceed if that estimate does not push consumption over the
active budget cap. This preflight-then-settle pattern exists specifically
so that a cost commitment is recorded synchronously with intent — before
the network call happens — rather than reconstructed after the fact, where
a crash mid-call could otherwise leave spend untracked.

After the call completes, the row is **settled** with the actual cost,
token counts, and latency observed. Calls that fail are marked `failed`,
not `settled`, and failed rows do not consume budget — a provider error
should never be charged against the ledger as if real work happened.

When a settled row's actual cost diverges from its preflight estimate by
more than 20%, that row is routed to a reconciliation queue rather than
silently accepted into the running total. This bounds how far a bad
estimation function (wrong token-counting heuristic, stale pricing, etc.)
can silently corrupt the ledger's picture of spend, and gives a concrete,
checkable trigger for investigating estimation drift instead of relying on
someone noticing the bill looks wrong at the end of the month.

This discipline is ported wholesale, as design language, from Industrial
Fusion Router's cost ledger semantics — one of the genuinely solid pieces
of engineering that line produced, even though none of its code is reused.

## ④ Validation

All training and evaluation of routing or fusion policies uses
**group-by-task cross-validation**: the unit of the split is the task (or
task-family), and every row derived from a given task — every model's
attempt at it, every feature computed from it — goes entirely into either
the train fold or the eval fold, never split across both.

**Row-wise leave-one-out validation is explicitly rejected as invalid**
whenever rows from the same task can land on both sides of a split. Even
though row-wise LOO looks like it validates — it holds out individual rows
— when those rows share a task with rows in the training fold, information
about that task (its answer pattern, its idiosyncrasies, its difficulty
signature) leaks across the split boundary. The resulting offline number is
not generalization signal; it is the model partially memorizing the task
through a side channel that resembles cross-validation but isn't. Any
result computed this way must be treated as invalid, not merely
"optimistic" — it does not tell you what will happen on a genuinely unseen
task.

This is the same underlying discipline as leakage (①), applied specifically
to validation methodology rather than to feature/input content, and it is a
direct, hard-won lesson from the prior online-routing research line's own experimentation.

## ⑤ Judges

Any judge used to score candidate or model outputs must be tested for
repeat-scoring consistency before its output is trusted for anything beyond
exploratory inspection: run the judge on the same fixed output at least
twice and measure the **sign-agreement** — whether repeated runs land on
the same side of a threshold or comparison.

The floor is **0.85 sign-agreement** on repeats. A judge that cannot clear
this bar is producing noise close to a coin flip relative to what a
reliable judge would need to achieve, and noise at that level cannot
function as a rubric no matter how sophisticated its stated criteria are.

A judge whose fixed-output repeat sign-agreement falls below 0.85 must not
be used to produce training labels — its outputs cannot be trusted enough
to teach a router or model anything — and must not be used to support
positive claims — it cannot be cited as evidence that one policy or
strategy outperforms another, because a judge this unreliable validates
nothing about which side actually won.

This threshold is not arbitrary: the prior online-routing research line's own DRACO-20 fast judge
measured only 0.63–0.74 repeat sign-agreement, and that measurement was one
of the two named reasons its fusion-gain labels were untrainable. 0.85 is
set as the hard floor going forward precisely because 0.63–0.74 was
empirically demonstrated, in this project's own prior work, to be
inadequate for anything load-bearing.

## ⑥ Secrets

Secrets — API keys, tokens, credentials of any kind — are never committed
to the repository and never printed: not to logs, not to stdout, not in
error messages, not in ledger rows, not in event payloads. If a value could
be used to authenticate as this system to a paid API, it does not appear in
anything that gets persisted, logged, or displayed outside of the process
that needs it to make the call.

Remote environment files that hold secrets — for example the VPS `.env`
carrying `DEEPSEEK_API_KEY`, `GLM_API_KEY`, and `GATEWAY_TOKENS` — must be
mode `600` (owner read/write only) on the remote host. They are created
manually or out-of-band on the host, not shipped through `rsync` or `git`
alongside the rest of the deployment.

Practically: before any commit or push, staged changes are diffed for
secret-shaped strings (for example, provider key prefixes like `sk-`)
before they leave the local machine. `.gitignore` covers `.env` and other
local secret files at the repository root so they cannot be added by
accident.

## ⑦ Compute placement

The VPS is the only place that **serves**. The gateway process binds
`127.0.0.1:8800` on the VPS and answers real OpenAI-compatible traffic
7×24, reached only through VPS-local services or an SSH tunnel — it is
never exposed directly to the public internet. This is the single process
in the whole system that production clients depend on being up.

The cookie GPU machine (RTX 3050 4GB) is where training, batch evaluation,
and heavy or repeated experiments run. It is not a serving target: no path
in this project's architecture should ever have the cookie machine
answering live production chat requests, and nothing on the serving path
should have a runtime dependency on the cookie machine being reachable.

The laptop never serves 7×24. It is a development and orchestration
surface only — a place to write code, run tests, and drive deploys from —
and there is no scenario in this project's design where a laptop process is
depended upon to keep production traffic flowing.

This placement mirrors the reason each legacy line ended up demoted:
the prior offline control-plane line's offline-plus-human-gate design didn't fit a
7×24 online-serving goal, which needs a dedicated always-on host — the VPS.
the prior online-routing research line's GPU-bound experimentation belongs on the machine that has
the GPU and no serving obligation — cookie. Production feature extraction
on the VPS is deliberately restricted to CPU-cheap small models (Qwen-0.5B
class, ONNX/quantized) precisely because the VPS is the serving host, not a
training host, and routing decisions themselves must stay cheap and fast
enough to run inline on every request.

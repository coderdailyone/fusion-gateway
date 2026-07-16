# Claude Code — A Behavioral Evaluation for Cost-Aware Routing

- Date: 2026-07-15
- Status: research note. Independent behavioral evaluation of the Claude
  Code CLI agent harness, focused on the question this gateway cares
  about most: **when does a stronger, more expensive model actually earn
  its price?** Findings are observations, not commitments.
- Subject: `claude` CLI 2.1.204–2.1.210, driven headlessly via
  `claude -p --output-format stream-json` across ~30 scripted runs.
- Models compared: `claude-haiku-4-5`, `claude-sonnet-5`,
  `claude-opus-4-8`, and `claude-opus-4-8 --effort max`.

## Why this note exists

A cost-aware routing gateway lives or dies on one empirical question: for
a given task, is the cheaper capable model *good enough*, and when it is
not, does escalation buy correctness or merely burn budget? The most
mature single-process coding agent available is a useful natural
laboratory for that question, because it exposes model choice directly
and its behavior can be scored against independently computed ground
truth. This note records what ~30 scripted probes found, with the
routing implications separated from the raw behavior.

## Method

- **Objective scoring wherever possible.** Ground truths were computed by
  brute force *before* probing — `fib(10)=55`, the derangement count
  `D8=14833`, the digit sum of `2^100`, and multi-case interpreter/
  language batteries the models never saw. "No gap" below is a checked
  null result over scored cases, not an impression.
- **Falsifiable probes.** Each probe was built so a specific failure
  *would* have shown: a recovery probe that would reveal a re-read or a
  wrong recollection; a puzzle rigged to elicit a specific hallucinated
  answer; a battery with adversarial error cases.
- **Headless, reproducible.** All runs were `claude -p` with structured
  JSON output, so tool calls, turn counts, and metered cost are read from
  the stream record rather than inferred.

Limitations are stated inline and collected at the end. The headline
results are single-run-per-cell where noted; treat multipliers as
directional, not variance-bounded.

## Headline — correctness vs. efficiency across models

### No correctness gap across ~20 hard tasks

Across roughly twenty objectively scored tasks — subtle-bug detection,
frontier reasoning (Burnside counting, derangements, the digit sum of
`2^100`), and full programming-language builds — `sonnet-5`, `opus-4.8`,
and `opus-4.8 --effort max` solved **every** objective case. Max-effort
reasoning caught nothing the mid-tier model missed. The single most
persuasive probe was a snail-in-a-well puzzle rigged to elicit an
off-by-one "day" answer; **both models resisted it identically**, which
is stronger evidence than a task both merely solved.

### Efficiency is task-shaped, and it *inverts*

Two autonomous test-driven builds were run to completion on `sonnet-5`
and `opus-4.8 --effort max`, then scored by an independent battery the
models never saw.

**Task 1 — expression interpreter (medium):**

|                        | sonnet-5   | opus-4.8 max            |
| ---------------------- | ---------- | ----------------------- |
| objective battery (15) | 15/15      | 15/15                   |
| model's own tests      | 38 pass    | pass                    |
| tool calls             | 5          | 5                       |
| implementation size    | 312 lines  | 406 lines               |
| cost                   | **$0.16**  | $0.36 (**2.3× more**)   |

**Task 2 — "MiniLang" (larger: recursion, loops, functions, strings):**

|                        | sonnet-5         | opus-4.8 max              |
| ---------------------- | ---------------- | ------------------------- |
| objective battery (19) | 19/19            | 19/19                     |
| model's own tests      | 62 pass          | 42 pass                   |
| turns / tool calls     | 19 / 18          | **5 / 4**                 |
| impl / test size       | 708 L / 502 L    | 421 L / 95 L              |
| cost                   | $1.04            | **$0.23 (4.5× cheaper)**  |

Same correctness on every case. But the cost relationship **flips with
task size**:

- On the **small** one-shot task, max-effort thinking is pure overhead —
  **2.3× more** for the identical answer.
- On the **larger** build, the stronger model was **4× fewer turns and
  4.5× cheaper**: it produced a complete, correct language in 4 tool
  calls, while the cheaper model reached the same score through 18
  iterative red-green turns.

Neither model dominates. The cheaper model's path is thorough (62 tests,
708 lines, many cycles); the stronger model's is decisive (42 tests, 421
lines, few turns). Both land correct, so the price gap is **not a fixed
premium** — it depends on how much iteration the task demands.

### Reading for a routing policy

- **Do not escalate for correctness.** On every objective case here, the
  mid-tier model matched max-effort. A router that escalates *expecting a
  better answer* will usually pay for nothing.
- **Escalate on demonstrated failure, not by default.** The evidence for
  a capability gap has to come from a failed check on the actual task,
  not from a prior about "hard" tasks.
- **Weigh expected iteration count, not per-token price.** On a large
  single deliverable a smaller model reaches through many round trips, a
  stronger, decisive model can *lower* total cost by cutting turns. Per-
  token price is the wrong routing signal in isolation; expected turns to
  a passing result is the right one.
- **Caveat.** These are single runs per cell; the turn-count gap partly
  reflects the cheaper model writing more tests (62 vs 42) — thoroughness
  and decisiveness are entangled, not a clean capability signal. The
  inversion motivates iteration-aware routing; it does not by itself
  establish a general rule.

## Supporting behavioral findings

These are secondary to the routing result but explain *why* the harness
behaves as it does, and several matter for anyone building a durable
agent runtime.

### Recovery and durability

- **Resume replays committed history, not a lossy summary.** A resumed
  session recalled its exact prior wording with **zero** tool calls —
  recovery is replay of durable state, not reconstruction from a
  compressed memory.
- **Interrupt recovery reconciles against reality.** A hard-killed build,
  on resume, rebuilt its progress from filesystem/git state rather than a
  self-reported checkpoint; atomic writes left no partial file.
- **Compaction is real, can fail, and is not a correctness substrate.**
  Headless compaction fires automatically (`status: compacting`),
  **fails** with `too_few_groups` on a handful of huge messages (leading
  to a terminal `"Prompt is too long"`), **succeeds** when there are many
  smaller groups, and on success preserves early facts verbatim — but it
  never bounds unbounded ingestion. A summary frame should accelerate a
  client, never replace committed history.
- **A single in-memory process is memory-fragile.** Long runs were
  repeatedly OS-killed (exit 137, OOM) on a 7.7 GB machine; the growing
  transcript is an unbounded in-memory substrate that hard-fails. On a
  15 GB / 16-core machine the same long builds completed. The clearest
  argument for bounded, durable, resumable job state over one long-lived
  process.

### Multi-agent topology

- **Fan-out then integrate, improvised per task.** A parallel request
  dispatched several self-contained sub-agents in one turn and then
  integrated their results — a Manager → Workers → Integrator shape
  arising ad hoc. Sub-agent failure propagated cleanly child → parent →
  user.
- **Sub-agents persist by identity.** Each child gets its own
  `<session>/subagents/agent-<id>.jsonl`; the parent thread keeps only
  the dispatch and the summary. A clean handoff shape: the parent holds a
  reference, the child's detail is separately retrievable.

### Tools, effects, and work discipline

- **Deferred tool discovery scales.** Tool *names* appear in the
  capability frame, but full schemas are fetched on demand; with 8 tools
  available the model fetched only the one it needed. Names-up-front,
  schemas-on-demand keeps the context small as the tool surface grows.
- **A pre-execution hook is a viable gate.** A `PreToolUse` hook cleanly
  intercepts before a tool runs — a sound shape for an approval or
  effect-authorization boundary.
- **Verify per step, not amortized.** Every "fixed/passing" claim was
  backed by an observed command result in the same transcript; on a
  46-turn documentation pass, the syntax check ran per file.
- **The model checkpoints by default.** A normal prompt did one increment
  and stopped ("once approved, I'll proceed"); only an explicit autonomy
  contract drove a full build. Do not expect a model to self-sustain a
  long job — the *driver* has to keep it going.
- **Fails closed only under total denial.** With any alternative
  reachable, the harness degrades gracefully; the pathological thrash
  appears only when *every* capability is denied, where failing closed
  and reporting is the correct behavior.
- **Locates by search, not bulk reads.** Token-frugal by default; it
  avoids the reads that would fill the window.

## Methodology reflection

**What held.** Computing ground truth before probing turned "seems as
good" into checked null results. Falsifiable design is why the recovery
conclusion stands — a lossy summary or a re-read would have shown, and
neither did. Rounds that merely re-confirmed "no gap" were reported as
null results rather than dressed up.

**What to do differently.** Harness-level behavior (compaction, resume,
hooks) is model-agnostic, so it should be probed on the *cheapest* model
— one early round wasted spend by exercising harness behavior on an
expensive model before internalizing this; the decisive compaction
results later cost cents on the cheapest model. And "construct a bounded
task long enough to separate the models" repeatedly failed because
capable models one-shot moderately complex builds; the genuinely-long,
multi-hour, open-ended regime was never reached and resists headless
scripting.

The errors shared one root: **trusting an impression instead of
re-deriving from an artifact** — assuming a probe would be cheap,
assuming a command executed, assuming "long enough." Every good result
came from the opposite move: compute the truth first, read the actual
stream record, check the exit code after the last edit.

## Open questions (do not close silently)

- **Genuinely multi-hour open-ended work** — the one regime where a model
  capability gap is most plausible was never reached; it resists headless
  scripting and either OOMs or one-shots on everything smaller.
- **Beyond-window compaction quality under real load** — the mechanism
  and one successful-recall case were observed, not a systematic sweep of
  which facts survive versus drop.
- **Per-model compaction quality** — compaction is harness-level; whether
  a stronger model summarizes better than a cheaper one was not compared.
- **Variance** — the head-to-head cells are single runs; the efficiency
  multipliers are directional, not variance-bounded.

## Relevance to this gateway

The routing thesis this project is built on — a cheap-and-strong model
pool with escalation governed by evidence — is consistent with what the
evaluation found:

1. **Correctness rarely justifies escalation.** Across ~20 objective
   tasks the mid-tier model matched the strongest configuration. A router
   should treat "escalate for a better answer" as the exception that must
   be *earned* by an observed failure, not the default.
2. **Cost is task-shaped, so route on iterations, not price.** The
   efficiency inversion — a stronger model *cheaper* on a large
   deliverable because it needs far fewer turns — means a per-token price
   comparison mis-ranks models on exactly the tasks where routing matters
   most. Expected turns to a passing result is the signal to model.
3. **Score against ground truth, not judgments, wherever a task allows
   it.** The most trustworthy findings here came from batteries computed
   before the model ran. A routing evaluation should prefer objective,
   pre-computed checks over model-graded rubrics wherever the task admits
   them.

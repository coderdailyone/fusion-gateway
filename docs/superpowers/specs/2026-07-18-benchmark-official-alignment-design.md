# M2c — Benchmark Official Alignment (判分严谨化)

**Status:** design approved 2026-07-18
**Milestone:** M2c (evaluation family; retroactively strengthens M3a/M3b)

## Why

The M3b SOTA/Pareto conclusions rest on **self-written scorers**. Those scorers
have mis-judged a *strong* model **three times**:

1. Kimi MMLU 0/5 — reasoning output truncated at 2048 tokens + scorer missed
   the `Answer: A` format.
2. GLM MMLU — GLM boxed the option *value* not its letter; markdown `**Final
   Answer:**` broke extraction.
3. gpt-5.6-sol math **−13.2pt** — scorer missed `\boxed{[-2,7]}` (interval,
   correct without an `x \in` prefix) and `\boxed{3\text{ treeks}}` (correct
   value with a unit label). The user caught this: a frontier model scoring
   weakest is a scorer smell, not a model fact.

Every one of these was a *scoring* bug, not a model result. A SOTA claim that
survives outside scrutiny cannot rest on "we wrote the grader ourselves." This
milestone replaces our scoring with **official-benchmark scoring, end to end**,
re-samples every feasible model through the official pipeline, and recomputes
the real numbers. It pays down the evaluation debt the routing work was built on.

## Decisions (locked with the user)

- **Rigor bar:** align to the **official graders** (not just harden ours).
- **Alignment depth:** **full pipeline** — official prompt + official extraction +
  official grading. We re-sample; we do not re-score the old frozen outputs.
- **Model scope:** **all feasible models** — DeepSeek, Claude Sonnet 5, Claude
  Opus 4.8, gpt-5.6-sol, gpt-5.5, GLM-5.2, and Kimi-k2 *if its quota allows*.
- **Prompt protocol:** **0-shot CoT** with each benchmark's official answer-format
  instruction. All models here are chat/reasoning models, for which 0-shot CoT is
  the accepted official protocol; few-shot would multiply token cost for no
  fidelity gain on chat models. (This is the one point the user delegated.)
- **Task suite:** unchanged — the locked 1063-task suite (mmlu599 / math300 /
  humaneval164). It is a frozen asset; we do not touch it.
- **Budget:** ~$25–35, gated by a preflight budget check before any paid call.

## Non-goals

- No LLM judge. Scoring stays objective and deterministic (extraction + symbolic
  equivalence + sandbox execution). The disagreement audit is human-eyeball only.
- No new benchmarks / tasks. Suite is frozen.
- No gateway changes. `evaluator/` and `router/` stay isolated from `gateway.*`
  and never touch the gateway SQLite.
- No few-shot prompting (see decision above).

## Architecture

### 1. `evaluator/official/` — vendored official graders (pinned source)

Each benchmark's **official judging core** is ported into the repo as pinned
source (not a pip dependency). Each file carries a header naming its upstream
**repo + commit/tag + license**. All three upstreams are MIT, so vendoring with
attribution is license-clean and keeps the public repo self-contained and
offline-reproducible.

| File | Ported from | Responsibility |
|---|---|---|
| `official/math_grade.py` | Hendrycks `MATH` `math_equivalence.py` (MIT) | `is_equiv(a, b)` + `_strip_string` normalization (fracs, `\text`/units, `\sqrt`, `\left/\right`, spaces) |
| `official/mmlu_extract.py` | TIGER-Lab `MMLU-Pro` `evaluate_from_local.py` (MIT) | official answer-letter extraction regex chain from CoT |
| `official/humaneval_exec.py` | OpenAI `human-eval` `execution.py` (MIT) | official `check_correctness`: assemble `prompt + completion + test + check(entry_point)`, execute isolated with timeout |
| `official/prompts.py` | each benchmark's official prompt spec | 0-shot CoT prompt templates per source (answer-format instruction only) |

**Provenance requirement:** every vendored file names the exact upstream commit
and license. Ported logic is copied faithfully; local edits are limited to
Python-version fixes and isolation shims, and each such edit is commented.

**One honest deviation (documented in code + report):** MMLU-Pro's official
extraction falls back to a *random* option when it cannot parse a letter, which
injects noise. Because our outputs are frozen and deterministically re-scorable,
a random fallback would make scores non-reproducible. We therefore run the
official extraction regex chain **verbatim** but treat a parse failure as
**incorrect** (conservative), rather than a random guess. This is stated in the
report as our single intentional divergence from the letter of the official
harness, with the count of affected items.

### 2. `evaluator/scorers/` — thin wrappers over `official/`

`scorers/{math,mcq,code}.py` are refactored to **delegate** to `official/`,
returning the existing `Score` type unchanged. The `Score` interface and
`SCORERS` mapping stay stable, so `router/`, `report.py`, and `pilot.py` need no
changes. The old hand-written normalization/extraction logic is removed (its job
is now the vendored graders'); the regression tests that pinned old bugs are
kept and re-pointed at the new path.

### 3. `evaluator/audit/reference_selftest.py` — the core audit weapon

For **every task in the suite**, construct the dataset's own gold answer as a
synthetic "model output" and assert the scorer marks it **correct**:

- **mmlu_pro:** synthesize `The answer is (GOLD_LETTER).` → must extract & match.
- **math:** synthesize `... \boxed{GOLD}` → `is_equiv(extracted, gold)` must be true.
- **humaneval:** run the dataset's `canonical_solution` through `check_correctness`
  → must pass (validates our sandbox equals the official execution protocol).

A scorer that cannot recognize the benchmark's *own* correct answer is provably
broken. Running this across all 1063 tasks surfaces every answer format the
grader mishandles at once — systematically, offline, for free — instead of
discovering them one paid model at a time.

- **Acceptance: 100% pass**, or a listed set of documented exceptions (each with
  task id + root cause + why it is acceptable, e.g. a known-malformed gold answer
  in the upstream dataset).
- This self-test becomes a **permanent CI regression gate**. The three historical
  bugs each get a dedicated regression case.

### 4. Official prompt + re-sample

`runner.build_prompt` is refactored to delegate to `official/prompts.py` while
remaining the **single leakage-guarded entry point** (it still consumes only
`task.problem`, never `task.answer`/`task.tests`; the existing leakage test is
retained). The re-sample reuses the existing resumable `sampler.sample` /
`pilot.run_pilot` driver (skips already-frozen `(task, model)` pairs) and writes
a **new** run dir — old frozen runs are kept as historical, not overwritten.

- Models: `deepseek-chat, claude-sonnet-5, claude-opus-4-8, gpt-5.6-sol,
  gpt-5.5, glm-5.2` and `kimi-k2` if its quota allows (skip-with-note if not).
- **Budget gate (preflight):** estimate cost from token counts + `configs/pricing.toml`
  before any paid call; warn at 80% of budget, hard-stop at 100%. Matches the
  project's ledger-preflight discipline.
- max_tokens: keep the reasoning-model-safe cap (8192) so reasoning models are
  not truncated (root cause of bug #1).

### 5. Recompute the real numbers

Score the new frozen outputs with the official scorers → a corrected result
matrix. Feed it to the existing `router/` pipeline (`ResultMatrix.from_rows` →
train / policy / cascade / pareto) and **retrain the M3b router on corrected
labels**. Produce:

- **`docs/BENCHMARK_REPORT.md`** (new): official-grader provenance + the one
  documented MMLU deviation; reference-answer self-test 100% pass; **true
  per-model accuracy** for all sampled models; whether absolute-SOTA-over-pool
  (including sol) and Pareto-dominance hold on corrected numbers.
- **`docs/M3B_REPORT.md`** (revised): retract the bug-inflated "~3×/2.6× cheaper"
  wording; replace with the recomputed router cost–quality numbers.

### 6. `evaluator/audit/disagreements.py` — lightweight human-eyeball audit

Surface the most informative cases for a human spot-check without an LLM judge:
tasks scored wrong for a model where another model scored correct, and tasks
where the scorer's method was `none` (unparseable). Emits a small ranked report;
no automated re-judging. Guards against a residual systematic extraction bias
that the reference self-test (which only tests gold answers) cannot see.

## Data flow

```
suite (locked 1063)  ──►  official/prompts  ──►  build_prompt (leakage-guarded)
                                                        │
                                                        ▼
              re-sample all feasible models (resumable, budget-gated)
                                                        │
                                                        ▼  frozen outputs (new run dir)
   scorers/*  ──delegate──►  official/{math_grade,mmlu_extract,humaneval_exec}
                                                        │
                        ┌───────────────────────────────┼───────────────────────────┐
                        ▼                               ▼                             ▼
        reference_selftest (gold→correct,      corrected ResultMatrix        disagreements
        100% gate, CI)                          → router retrain → Pareto     (human spot-check)
                                                        │
                                                        ▼
                                   BENCHMARK_REPORT.md + revised M3B_REPORT.md
```

## Error handling

- **Unparseable model output** → scored incorrect with `method="none"`, captured
  for the disagreements audit (never a random guess; never a crash).
- **Sandbox timeout / exception** (code track) → task incorrect, isolated per the
  official `check_correctness` subprocess guard; a runaway does not hang the run.
- **Endpoint slow / flaky / quota-exhausted** (gpt-5.5, GLM, Kimi) → the sampler
  is resumable; a model that cannot complete is dropped from the run with an
  explicit note in the report rather than silently partial.
- **Budget breach** → hard stop at 100% before the offending call; partial frozen
  outputs remain valid and resumable.

## Testing

- **Unit tests** for each `official/` grader, including the **three historical
  bugs as regression cases** (interval/set answers, `\text` units, `x \in`
  prefix; GLM value-boxing / markdown `**Final Answer:**`; Kimi `Answer: A`).
- **Reference self-test** over all 1063 tasks: 100% pass (or documented
  exceptions) — wired as a CI gate.
- **Isolation test:** `evaluator/official/` and `evaluator/audit/` import no
  `gateway.*` and touch no gateway SQLite (existing leakage/isolation tests
  extended).
- **Determinism:** re-scoring the same frozen output twice yields the same Score
  (no random fallback anywhere in the path).

## Acceptance criteria

1. Reference-answer self-test passes on all 1063 tasks (or exceptions listed with
   id + root cause), and runs as a CI regression gate.
2. Every vendored grader carries upstream repo + commit/tag + license; the single
   MMLU extraction deviation is documented in code and report.
3. All feasible models re-sampled through the official 0-shot-CoT pipeline and
   frozen in a new run dir; any dropped model is noted with the reason.
4. `docs/BENCHMARK_REPORT.md` published with true per-model numbers and the
   recomputed router/Pareto/SOTA verdict; `docs/M3B_REPORT.md` revised to retract
   the bug-inflated cost claims.
5. `evaluator/official/` + `evaluator/audit/` isolated from the gateway; the
   three historical bugs covered by regression tests; scoring deterministic.
6. Total spend ≤ budget, enforced by a preflight budget gate.

# M2d — Hard Tier + Contamination-Resistant Benchmark

**Status:** design approved 2026-07-19
**Milestone:** M2d (evaluation family; builds on M2c official scoring)

## Why

The M2c result showed the frontier models bunched under a saturated ceiling:
HumanEval 0.97–0.98 (no discrimination), MATH compressing at 0.95–0.96,
overall spread so tight that "who is #1" (Opus 0.913 vs GPT-5.5 0.905 vs the
k3 subset) sits **within statistical noise** (~1σ). Two consequences:

1. **No discrimination at the top** — a saturated suite can't separate frontier
   models, so the ranking is noise.
2. **Contamination is unfalsifiable** — MMLU-Pro / MATH / HumanEval are public
   and old; a newer model may have trained on them, and we can't tell memorized
   answers from reasoning.

M2d adds a **separate hard tier** that (a) is hard enough to spread the frontier
and (b) includes **timestamped / fresh** sources so contamination becomes a
measurable signal instead of an unfalsifiable caveat.

## Positioning (locked with the user)

- **The standard tier (1057, M2c) is untouched.** It stays the authoritative
  cost-quality / router-Pareto stage — routing's value (cheap on easy, escalate
  on hard) is best shown on mixed difficulty; an all-hard suite would understate
  it. The hard tier is a *separate* deliverable for frontier discrimination +
  contamination, not a replacement.
- **Same M2c discipline:** vendored official scorers with provenance, a locked
  content-hashed manifest, `evaluator/` isolated from `gateway.*`, frozen
  re-scorable outputs, reference self-test before paid sampling.
- **Full hard tier in one milestone** (not phased): 4 hard sources.
- **Both hard-math sources:** AIME (fresh) + MATH level-5 (volume).

## Sources — new locked manifest `configs/suite.hard.manifest.json`

| source | dataset (intended) | N | scorer | contamination |
|---|---|---|---|---|
| `livecodebench` | `livecodebench/code_generation_lite`, filtered to `release_date ≥ 2024-08-01` | ~150 (stratified by difficulty) | **new**: official LCB execution | **resistant** (timestamped) |
| `gpqa_diamond` | `Idavidrein/gpqa` (config `gpqa_diamond`) | 198 (full) | reuse MCQ (A–D) + seeded option shuffle | public (flagged) |
| `aime` | AIME 2024 + AIME 2025 (HF, pinned at build) | ~60 (full) | reuse official math grader (integer exact) | **resistant** (2025 post-cutoff) |
| `math_l5` | MATH `Level 5` subset (`hendrycks/competition_math`, `level=="Level 5"`) | ~250 (stratified by subject) | reuse official math grader | public (flagged) |

Total ≈ 660 hard tasks. Exact HF dataset ids/revisions are resolved and pinned
at build time via `dataset_info(...).sha` (as `build_manifest.py` already does);
the implementer verifies each dataset is reachable and adjusts the id if a named
one is unavailable, keeping the same shape.

**Effort leverage:** AIME and MATH-L5 answers go straight through the existing
`evaluator/official/math_grade` (integer/expression exact). GPQA is 4-option MCQ
and reuses the existing `evaluator/official/mmlu_extract` letter chain (A–J
already covers A–D). **The only genuinely new scorer is LiveCodeBench execution.**

## Scorers

### `evaluator/official/livecodebench_exec.py` (new — the technical crux)

LiveCodeBench problems carry test cases in two shapes:
- **functional / call-based:** a `fn_name` + `(input, expected_output)` pairs; the
  candidate defines that function and it is called with parsed inputs.
- **stdin/stdout:** the candidate reads stdin and its stdout is compared to
  expected (trimmed).

Vendor LCB's official `code_generation` evaluation protocol (repo
`LiveCodeBench/LiveCodeBench`, pin real commit + license in the header — verify
license against upstream at implement time): build the check program per shape,
run it in the existing `evaluator.sandbox.run_code` (isolation discipline, not
LCB's in-process exec), enforce the per-test timeout, pass@1 = all public+private
tests pass. The fetcher normalizes LCB's `public_test_cases`/`private_test_cases`
(JSON strings) into `task.tests` tuples of the shape `evaluator/scorers/code.py`
already consumes (`{"kind": "stdin", stdin, expected_stdout}` or
`{"kind": "pyfunc", test, entry_point}`), so the code scorer's execution loop is
reused; only the LCB-specific program assembly + input parsing is new.

### GPQA-Diamond — reuse MCQ

The fetcher builds the question with its 4 options (1 correct + 3 incorrect from
the dataset) in a **seeded deterministic shuffle** (recorded in the manifest, so
the content hash pins the exact letter↔option mapping), records the correct
letter, and uses the official 0-shot-CoT MMLU-style prompt ("finish with 'The
answer is (X)'"). Scoring reuses `evaluator/scorers/mcq.py` →
`evaluator/official/mmlu_extract`.

### AIME + MATH-L5 — reuse official math grader

Both reuse `evaluator/scorers/math.py` → `evaluator/official/math_grade`
(extraction of the last `\boxed{}` + `math_equiv`). AIME answers are integers
0–999; MATH-L5 answers are MATH-format. No new scorer.

## Fetchers — `evaluator/hf_fetchers.py` + `evaluator/suite/loader.py`

Add `extract` / `stratum` for `livecodebench` (release-window filter + difficulty
stratum), `gpqa_diamond` (seeded 4-option shuffle → record correct letter;
stratum = subject), `aime` (stratum = year), `math_l5` (`level==5` filter;
stratum = subject). `parse_task` already stubs `livecodebench`; add `gpqa_diamond`
(MCQ-shaped like mmlu_pro but A–D), `aime` and `math_l5` (math-shaped). The
seeded shuffle for GPQA lives in the fetcher so it is captured by the content
hash and reproducible.

## Manifest builder

Extend the manifest tooling to build the hard manifest without touching the
standard one: a `HARD_SOURCES` list (source, dataset, split, size, plus any
filter predicate like the LCB date window / MATH level) and a
`build_hard()` that writes `configs/suite.hard.manifest.json` (version 1). The
`Manifest`/`SourceSpec`/`content_sha`/`load` types are unchanged; a source with
a filter records only the ids that survive the filter, so the hash still pins the
exact record set. `load_suite` is unchanged (loads any manifest path).

## Contamination probe (the differentiating value)

The report turns "contamination is unfalsifiable" into a **relative signal**:

- Partition the hard sources into **fresh** {`aime` (2025 subset), `livecodebench`}
  and **public** {`math_l5`, `gpqa_diamond`}.
- For each model compute its rank (and accuracy) within each partition.
- A model whose **public rank is much higher than its fresh rank** (relative to
  peers) is a **memorization suspect** — it looks strong on data it may have
  trained on but drops on genuinely-unseen problems. Report a per-model
  fresh-vs-public delta table; large positive public−fresh deltas are flagged.

This does not *prove* contamination for any single model, but it makes the
frontier ranking honest: conclusions are cross-checked on the fresh partition.

## Sampling scope

Frontier cluster + cheap baseline: `deepseek-chat`, `claude-sonnet-5`,
`claude-opus-4-8`, `gpt-5.6-sol`, `gpt-5.5`, `glm-5.2`. **Kimi is skipped**
(Moonshot account quota exhausted; see M2c). Reuse the M2c per-model-parallel +
budget-gated + sharded sampling infrastructure (`scripts/resample_official.py`
`run_budgeted`, the shard wrapper). Hard problems produce longer outputs and
reasoning models are slow, so budget-gate at ~$25 and expect slower wall-clock;
resumable + error-retry as in M2c.

## Report — `docs/HARD_TIER_REPORT.md`

- Per-model, per-source hard-tier accuracy **with Wilson 95% confidence
  intervals**, and a pairwise significance note (paired/McNemar) on whether the
  frontier actually separates — directly answering the M2c "ranking is within
  noise" problem.
- The fresh-vs-public contamination table.
- Verdict: does the hard tier de-tie the frontier, who is genuinely strongest,
  and which models are memorization suspects.

## Error handling

- Unparseable / unexecuted → scored incorrect, captured for the disagreement
  audit (never a crash, never a random guess).
- Transient mirror 5xx/timeout during sampling → prune-and-retry (M2c pattern).
- A dataset that is unreachable / renamed at build time → the implementer pins an
  equivalent and notes it; the milestone does not silently ship a missing source.
- LCB execution timeout/exception → that test fails (pass@1 = false), isolated in
  the sandbox subprocess; a runaway cannot hang the run.

## Testing

- Unit tests for the new `livecodebench_exec` assembly (both functional + stdin
  shapes) and for the GPQA seeded-shuffle fetcher (deterministic, correct letter
  tracked).
- Reference self-test extended to the hard sources: LCB canonical/reference
  solutions pass; AIME/MATH-L5 gold answers recognized; GPQA gold letter
  recognized. Run at $0 before paid sampling.
- Isolation test: new modules import no `gateway.*`.
- Determinism: re-scoring a frozen hard-tier output twice is identical; the GPQA
  shuffle is seed-stable.

## Acceptance criteria

1. `configs/suite.hard.manifest.json` locked + content-hash verified on load;
   reference self-test passes on all four hard sources (or documented exceptions).
2. LiveCodeBench + (GPQA reuse) scorers vendored/aligned with upstream provenance
   (repo + real commit + license); AIME/MATH-L5 reuse the M2c math grader.
3. Frontier models (6, no Kimi) re-sampled through the official hard-tier pipeline
   under a preflight budget gate; frozen, resumable, errors retried.
4. `docs/HARD_TIER_REPORT.md` published with Wilson CIs + pairwise significance
   and the fresh-vs-public contamination analysis.
5. The standard tier (`configs/suite.manifest.json`, M2c results) is unchanged;
   `evaluator/` stays isolated from the gateway; scoring deterministic.

## Non-goals

- Not replacing or re-scoring the standard tier.
- No new router/fusion work here — the hard tier is a measurement deliverable; a
  hard-tier router experiment is a possible later milestone, not this one.
- No LLM judge; scoring stays objective (extraction + math equivalence + sandbox
  execution).
- Kimi not sampled (account quota); can be added after a quota refresh.

# M2d — Hard-Tier Benchmark Report

> **STATUS: SKELETON — NOT YET FILLED.** This file is a placeholder created
> ahead of Task 8 (manual, paid) of the M2d hard-tier plan
> (`docs/superpowers/plans/2026-07-19-hard-tier-benchmark.md`). Every number
> below is a `TBD` placeholder, not a real measurement. Do not cite any
> figure from this file until Task 8 replaces the placeholders with real
> results from `scripts/hard_report.py` run against
> `evaluator/runs/m2d_hard/`.

**Headline:** TBD — filled after the full hard-tier sample + report run.

## Scoring provenance

Same official-alignment discipline as M2c (`docs/BENCHMARK_REPORT.md`),
extended to the four hard sources:

- **LiveCodeBench** — `evaluator/official/livecodebench_exec.py`, matching the
  evaluation protocol in `LiveCodeBench/LiveCodeBench` (repo + commit + license
  pinned in that module's docstring); executed in this project's isolated
  sandbox (`evaluator/sandbox.py::run_code`), not LCB's in-process harness.
- **GPQA-Diamond** — reuses the M2c MMLU-Pro chain
  (`evaluator/official/mmlu_extract.py`) unchanged; options are a
  seeded-deterministic shuffle recorded in the manifest's content hash.
- **AIME (2024 + 2025)** and **MATH level-5** — reuse the M2c Hendrycks math
  grader (`evaluator/official/math_grade.py`) unchanged.
- **Reference self-test (free gate before any spend):** TBD/PASS — result of
  running the self-test on `configs/suite.hard.manifest.json` (Task 8 Step 1).
  Record the pass count per source here (e.g. `gpqa_diamond N/N`, `aime N/N`,
  `math_l5 N/N`, `livecodebench N/N` — LiveCodeBench is a best-effort smoke,
  not a "gold recognized" guarantee; see `evaluator/audit/reference_selftest.py`).

## Per-model accuracy + Wilson 95% CIs (placeholder)

Common-task-set accuracy across the 6 sampled models (`deepseek-chat`,
`claude-sonnet-5`, `claude-opus-4-8`, `gpt-5.6-sol`, `gpt-5.5`, `glm-5.2`; Kimi
excluded, per-model quota — see M2c). Fill from `scripts/hard_report.py`'s
per-model accuracy + `wilson_ci` output.

| rank | model | overall | gpqa_diamond | aime | math_l5 | livecodebench | 95% CI (overall) |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Pairwise significance

Top-pair McNemar test (`scripts/hard_report.py::mcnemar_p`, normal
approximation over discordant pairs) on whether the hard tier actually
separates the frontier, directly answering the M2c "ranking is within noise"
problem.

- Top pair: TBD vs TBD
- Discordant counts: b=TBD, c=TBD
- McNemar p-value: TBD
- Interpretation: TBD (significant separation vs statistically tied)

## Fresh-vs-public contamination table (placeholder)

Partition: **fresh** = `{aime, livecodebench}` (2025 AIME is post-cutoff for
most models; LiveCodeBench is filtered to `release_date >= 2024-08-01`),
**public** = `{gpqa_diamond, math_l5}`. A model whose public accuracy is much
higher than its fresh accuracy (relative to peers) is a memorization suspect.

| model | fresh accuracy | public accuracy | delta (public − fresh) | flag |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD |

## Verdict

- **Does the hard tier de-tie the frontier?** TBD.
- **Who is genuinely strongest?** TBD.
- **Which models are memorization suspects** (large positive public−fresh
  delta relative to peers)? TBD.
- **Spend:** TBD (budget ceiling ~$25, per the M2d plan).

## Known limitations

- LiveCodeBench's `code_generation_lite` ships no gold/canonical completion,
  so its reference self-test is a best-effort smoke (tests execute and are
  non-empty), not a "gold recognized" guarantee like the other three sources
  get — see `evaluator/audit/reference_selftest.py`.
- The fresh-vs-public partition is a relative signal, not proof of
  contamination for any single model (see
  `docs/superpowers/specs/2026-07-19-hard-tier-benchmark-design.md`).

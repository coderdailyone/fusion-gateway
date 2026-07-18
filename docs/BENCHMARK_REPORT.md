# M2c — Benchmark Official-Alignment Report

**Scoring:** official-benchmark judging, vendored under `evaluator/official/`:
MATH `is_equiv`/`_strip_string` (Hendrycks, MIT) + sympy fallback; MMLU-Pro
extraction chain (TIGER-Lab, Apache-2.0); HumanEval `check_correctness` assembly
(OpenAI human-eval, MIT). Prompts: official 0-shot CoT.

**One documented deviation:** MMLU-Pro official extraction falls back to a
random option on parse failure; for deterministic re-scoring we treat a parse
failure as incorrect. Affected items: <N> (from the disagreement audit).

**Reference self-test:** PASS 1063/1063 — every dataset's own gold answer is
recognized by the official grader (humaneval 164/164, math 300/300, mmlu_pro
599/599), 0 failures. Run offline at $0 as the gate before paid sampling.

## True per-model accuracy (official scoring, official prompts)

| model | accuracy | mean cost/task |
|---|---|---|
| ... | ... | ... |

## Router on corrected labels

<paste render_report output: static baselines, lambda curve, envelope verdict>

## SOTA verdict

<does any strategy beat every single model incl. gpt-5.6-sol? Pareto dominance?>

## Disagreement audit

<count + notable human-checked cases>

**Known limitation:** the humaneval reference self-test synthesizes
`prompt + canonical_solution` as its check, whereas real scoring
(`evaluator/scorers/code.py`) grades the extracted completion alone (no
prompt prepend) — so the self-test cannot catch a completion that silently
relies on prompt context (e.g. an import) it never restates. Watch for
humaneval false-negatives in the disagreement audit as a result.

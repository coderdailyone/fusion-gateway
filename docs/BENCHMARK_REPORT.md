# M2c — Benchmark Official-Alignment Report

**Scoring:** official-benchmark judging, vendored under `evaluator/official/`:
MATH `is_equiv`/`_strip_string` (Hendrycks, MIT) + sympy fallback; MMLU-Pro
extraction chain (TIGER-Lab, MIT); HumanEval `check_correctness` assembly
(OpenAI human-eval, MIT). Prompts: official 0-shot CoT.

**One documented deviation:** MMLU-Pro official extraction falls back to a
random option on parse failure; for deterministic re-scoring we treat a parse
failure as incorrect. Affected items: <N> (from the disagreement audit).

**Reference self-test:** <PASS 1063/1063 | list exceptions>.

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

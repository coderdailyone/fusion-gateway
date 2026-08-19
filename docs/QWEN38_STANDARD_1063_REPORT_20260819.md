# Qwen 3.8 Max — Standard 1063 Benchmark Report

**Status:** complete · **Run date:** 2026-08-19 · **Mode:** single-model baseline  
**Model route:** Token Plan OpenAI-compatible endpoint, model `qwen3.8-max`  
**Scope:** MMLU-Pro (599) + MATH-500 (300) + HumanEval (164) = **1063** tasks
**Repository record:** versioned with the project's benchmark reports.

## Summary

`qwen3.8-max` completed **1063/1063** requests successfully. Official offline
scoring gives **948/1063 = 0.8918** accuracy (95% Wilson CI
`[0.8717, 0.9091]`).

This is a **single-model** result. It does not call the `fusion` or
`fusion-single` endpoint, so it is a clean baseline for later fusion
architecture work.

On the historical M2c six-model common set (1057 tasks), Qwen scores
**946/1057 = 0.8950**. It is statistically indistinguishable from both
`gpt-5.6-sol` (0.8940, one fewer correct task; McNemar p=1.000) and
`claude-sonnet-5` (0.8978, three more correct tasks; p=0.826). It is
significantly better than the historical DeepSeek and GLM runs, but remains
below Opus 4.8 (p=0.023).

## Scoring provenance

The scoring path follows `docs/BENCHMARK_REPORT.md`:

- **MMLU-Pro:** vendored TIGER-Lab official answer extraction.
- **MATH-500:** vendored Hendrycks-style equivalence grader with the project's
  symmetric normalization.
- **HumanEval:** vendored OpenAI HumanEval execution harness.

All scoring was offline over `frozen.jsonl`; no scoring step made model API
calls. The locked dataset revisions were downloaded through `hf-mirror.com`,
then reloaded with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.

## Full 1063-task result

| source | correct | tasks | accuracy | 95% Wilson CI |
|---|---:|---:|---:|---|
| MMLU-Pro | 501 | 599 | 0.8364 | [0.8046, 0.8639] |
| MATH-500 | 288 | 300 | 0.9600 | [0.9314, 0.9770] |
| HumanEval | 159 | 164 | 0.9695 | [0.9306, 0.9869] |
| **overall** | **948** | **1063** | **0.8918** | **[0.8717, 0.9091]** |

Coverage and request health:

- `1063/1063` unique `(task_id, model)` pairs were frozen.
- All `1063` rows have `status=ok`.
- No successful response was empty.
- The run completed normally (`RC=0`); the configured `$8` ceiling was not
  approached.

## Strict comparison with historical single-model baselines

`docs/BENCHMARK_REPORT.md` compares historical M2c models only where every
model returned successfully. Reconstructing that same intersection yields
1057 tasks. Qwen is re-scored on exactly those tasks below.

| rank | model | correct / 1057 | accuracy |
|---:|---|---:|---:|
| 1 | claude-opus-4-8 | 965 | 0.9130 |
| 2 | gpt-5.5 | 957 | 0.9054 |
| 3 | claude-sonnet-5 | 949 | 0.8978 |
| 4 | **qwen3.8-max** | **946** | **0.8950** |
| 5 | gpt-5.6-sol | 945 | 0.8940 |
| 6 | deepseek-chat | 908 | 0.8590 |
| 7 | glm-5.2 | 890 | 0.8420 |

Pairwise, continuity-corrected McNemar tests on the same 1057 tasks:

| comparison | Qwen-only correct | other-only correct | net Qwen | p-value | interpretation |
|---|---:|---:|---:|---:|---|
| Qwen vs gpt-5.6-sol | 34 | 33 | +1 | 1.000 | indistinguishable |
| Qwen vs claude-sonnet-5 | 40 | 43 | -3 | 0.826 | indistinguishable |
| Qwen vs gpt-5.5 | 20 | 31 | -11 | 0.161 | no significant difference at 5% |
| Qwen vs claude-opus-4-8 | 22 | 41 | -19 | 0.023 | Qwen is significantly worse |
| Qwen vs deepseek-chat | 68 | 30 | +38 | 0.00019 | Qwen is significantly better |
| Qwen vs glm-5.2 | 87 | 31 | +56 | <0.000001 | Qwen is significantly better |

The substantive conclusion is narrow: Qwen is in the historical
Sonnet/`gpt-5.6-sol` band on this suite, but this run does not establish a
quality win over either model.

## Runtime, tokens, and estimated cost

The sampler used 8 workers and ran from `13:53:38` to `14:18:56` JST:
**25m18s** wall-clock.

| source | input tokens | output tokens | mean latency | p50 latency | max latency | estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 156,441 | 435,121 | 11.50 s | 7.55 s | 189.92 s | $0.32325 |
| MATH-500 | 36,379 | 384,564 | 15.34 s | 7.10 s | 97.64 s | $0.27835 |
| HumanEval | 30,340 | 32,061 | 2.60 s | 2.41 s | 11.39 s | $0.02517 |
| **overall** | **223,160** | **851,746** | — | **6.74 s** | **189.92 s** | **$0.62677** |

- Mean estimated cost: `$0.000590` per task, `$0.000661` per correct answer.
- Serial sum of request latency: 11,920.8 seconds. The observed
  wall-clock/serial ratio was 7.85×, consistent with the configured 8 workers.
- Five outputs reached the `max_tokens=8192` cap (four MATH, one MMLU-Pro).

**Cost limitation:** `$0.62677` is calculated from the Qwen placeholder rates
in `configs/pricing.toml` (`$0.072/M` input, `$0.717/M` output). The upstream
response exposed no per-request price, so `frozen.jsonl` stores `cost_usd=0`.
This figure is a planning estimate, not an audited Token Plan bill; reconcile
it against the Token Plan account statement before making cost claims.

## Failure analysis

There were 115 incorrect tasks:

- **98 MMLU-Pro:** all produced a parseable official option letter, but the
  letter did not match the gold answer. There were no MMLU formatting failures.
- **12 MATH-500:** a mix of substantive errors and final-answer extraction /
  normalization sensitivity. For example, one response gave `10080` while the
  gold was `10,\!080`, and another used `\textbf{(B)}` where the gold was
  `\text{(B)}`. These should be audited before interpreting all 12 as model
  reasoning failures. Other errors were plainly substantive or did not place
  the intended final expression in a machine-extractable final answer.
- **5 HumanEval:** concrete code failures, including undefined helper functions
  in `HumanEval/10`, `HumanEval/38`, and `HumanEval/50`; incorrect behavior in
  `HumanEval/127`; and an ordering/tie-break error in `HumanEval/145`.

The main performance limitation is MMLU-Pro (83.6%), not code or MATH.

## Relation to earlier fusion work

`docs/M5_FUSION_REPORT.md` records a fusion row with **948 correct** but reports
accuracy `0.8901`; that decimal corresponds to a 1065-row denominator, whereas
this Qwen run has exactly 1063 unique task/model pairs and uses
`948/1063 = 0.8918`. Therefore the two reports should **not** be ranked by their
displayed accuracy without normalizing task identity and duplicate handling.

The useful architecture implication is that Qwen is strong enough to be a
serious primary-solver candidate in the proposed `fusion-single` design:
single-model Qwen already reaches the historical middle/frontier-adjacent band.
The next meaningful experiment is not another Qwen baseline; it is a paired
test over the same frozen task set comparing:

1. Qwen alone;
2. Qwen as primary solver with the other models reviewing/editing;
3. the existing multi-candidate fusion path.

This will isolate whether review/fusion adds value beyond Qwen's baseline,
rather than confounding the result with model selection.

## Reproducibility and artifacts

Run directory:

```text
/home/yangjia/fusion/workload/qwen38_standard_1063_20260819
```

Artifacts:

```text
frozen.jsonl          # 1063 raw model outputs
official_score.json   # full-suite official scoring aggregate
status.env            # start/end times and return code
```

The runner used:

```bash
export HF_HOME=/home/yangjia/fusion/workload/hf_cache_probe
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

set -a
source .env
set +a

PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/sample_one_par.py \
  qwen3.8-max 1063 8 \
  /home/yangjia/fusion/workload/qwen38_standard_1063_20260819 8
```

The `qwen3.8-max` model registry entry in `evaluator/validate.py` points
directly to Token Plan and sets
`enable_thinking=false`, matching the gateway Qwen configuration used in the
previous smoke test.

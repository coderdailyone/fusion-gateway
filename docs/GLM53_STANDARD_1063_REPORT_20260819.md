# GLM 5.3 — Standard 1063 Benchmark Report

**Status:** complete · **Run date:** 2026-08-19 · **Mode:** single-model baseline<br>
**Model route:** direct GLM Anthropic-compatible endpoint, model `glm-5.3`<br>
**Scope:** MMLU-Pro (599) + MATH-500 (300) + HumanEval (164) = **1063** tasks
**Repository record:** versioned with the project's benchmark reports.

## Summary

`glm-5.3` completed **1063/1063** requests successfully. Official offline
scoring gives **900/1063 = 0.8467** accuracy (95% Wilson CI
`[0.8238, 0.8671]`).

This is a **single-model direct-API** result. It does not call `fusion` or
`fusion-single`, so it is a baseline only; it does not establish that GLM 5.3
should replace GLM 5.2 as the fusion quorum member, reviewer, or fuser.

On the historical M2c six-model common set (1057 tasks), GLM 5.3 scores
**896/1057 = 0.8477**, versus GLM 5.2's **890/1057 = 0.8420**. The +6-task
difference is not statistically significant (paired McNemar p=0.673).

## Scoring provenance

The scoring path follows `docs/BENCHMARK_REPORT.md`:

- **MMLU-Pro:** vendored TIGER-Lab official answer extraction.
- **MATH-500:** vendored Hendrycks-style equivalence grader with the project's
  symmetric normalization.
- **HumanEval:** vendored OpenAI HumanEval execution harness.

All scoring was offline over `frozen.jsonl`; no scoring step made model API
calls. The locked datasets were loaded from the local cache with
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.

## Full 1063-task result

| source | correct | tasks | accuracy | 95% Wilson CI |
|---|---:|---:|---:|---|
| MMLU-Pro | 497 | 599 | 0.8297 | [0.7975, 0.8577] |
| MATH-500 | 252 | 300 | 0.8400 | [0.7943, 0.8771] |
| HumanEval | 151 | 164 | 0.9207 | [0.8691, 0.9531] |
| **overall** | **900** | **1063** | **0.8467** | **[0.8238, 0.8671]** |

Coverage and request health:

- `1063/1063` unique `(task_id, model)` pairs were frozen.
- All `1063` rows have `status=ok`.
- No successful response was empty.
- The run completed normally (`RC=0`); the configured `$5` ceiling was not
  approached.

## GLM 5.3 versus GLM 5.2 and Qwen

The source-level profile changed rather than improving uniformly:

| source | GLM 5.2 | GLM 5.3 | delta 5.3−5.2 | Qwen 3.8 Max |
|---|---:|---:|---:|---:|
| MMLU-Pro | 0.7880 (472/599) | **0.8297 (497/599)** | **+0.0417** | 0.8364 |
| MATH-500 | **0.8867 (266/300)** | 0.8400 (252/300) | **-0.0467** | 0.9600 |
| HumanEval | **0.9390 (154/164)** | 0.9207 (151/164) | **-0.0183** | 0.9695 |
| full suite | 0.8391 (892/1063) | **0.8467 (900/1063)** | **+0.0075** | 0.8918 |

Interpretation:

1. GLM 5.3 gains **25 MMLU-Pro** answers over the historical GLM 5.2 output,
   but loses **14 MATH-500** and **3 HumanEval** answers. The full-suite gain is
   only eight tasks.
2. Qwen retains a substantial advantage: it answered 948/1063 correctly versus
   GLM 5.3's 900/1063, a 48-task full-suite gap.
3. Therefore the observed change is a profile trade, not enough evidence to
   claim a reliable quality upgrade for the current fusion design.

## Strict comparison with historical single-model baselines

`docs/BENCHMARK_REPORT.md` compares historical M2c models only where every
model returned successfully. Reconstructing that same intersection yields
1057 tasks. GLM 5.3 is re-scored on exactly those tasks below.

| rank | model | correct / 1057 | accuracy |
|---:|---|---:|---:|
| 1 | claude-opus-4-8 | 965 | 0.9130 |
| 2 | gpt-5.5 | 957 | 0.9054 |
| 3 | claude-sonnet-5 | 949 | 0.8978 |
| 4 | qwen3.8-max | 946 | 0.8950 |
| 5 | gpt-5.6-sol | 945 | 0.8940 |
| 6 | deepseek-chat | 908 | 0.8590 |
| 7 | **glm-5.3** | **896** | **0.8477** |
| 8 | glm-5.2 | 890 | 0.8420 |

Paired, continuity-corrected McNemar tests on the same 1057 tasks:

| comparison | GLM 5.3-only correct | other-only correct | net GLM 5.3 | p-value | interpretation |
|---|---:|---:|---:|---:|---|
| GLM 5.3 vs GLM 5.2 | 73 | 67 | +6 | 0.673 | no significant difference |
| GLM 5.3 vs DeepSeek | 62 | 74 | -12 | 0.346 | no significant difference |
| GLM 5.3 vs Qwen 3.8 Max | 34 | 84 | -50 | 0.000006 | GLM 5.3 is significantly worse |
| GLM 5.3 vs gpt-5.6-sol | 36 | 85 | -49 | 0.000013 | GLM 5.3 is significantly worse |
| GLM 5.3 vs claude-sonnet-5 | 33 | 86 | -53 | 0.000002 | GLM 5.3 is significantly worse |

The strict conclusion is that GLM 5.3 is in the existing DeepSeek/GLM tier on
this suite. It does not close the gap to Qwen or the frontier references, and
it does not establish a statistically reliable improvement over GLM 5.2.

## Runtime, tokens, and estimated cost

The sampler used 8 workers and ran from `20:14:59` to `20:23:09` JST:
**8m10s** wall-clock.

| source | mean input tokens | mean output tokens | mean latency | p50 latency | max latency | estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 233.95 | 150.91 | 3.38 s | 2.28 s | 89.27 s | $0.28296 |
| MATH-500 | 104.37 | 250.76 | 4.93 s | 2.47 s | 151.59 s | $0.18429 |
| HumanEval | 161.55 | 103.29 | 1.95 s | 1.76 s | 6.57 s | $0.05316 |
| **overall** | **186.21** | **171.75** | — | **2.20 s** | **151.59 s** | **$0.52041** |

- Total tokens: 197,942 input + 182,566 output.
- Mean estimated cost: `$0.000490` per task, `$0.000578` per correct answer.
- Serial sum of request latency: 3,824.6 seconds. The wall-clock/serial ratio
  was 7.81×, consistent with the configured 8 workers.
- No output reached the `max_tokens=8192` cap.

Relative to the Qwen 3.8 Max run, GLM 5.3 finished about 3.1× faster and used
about 4.7× fewer output tokens. This operational advantage did not translate
into comparable accuracy: Qwen scored 0.8918 versus GLM 5.3's 0.8467.

**Cost limitation:** `$0.52041` is a budget estimate using the historical
GLM-5.2 rates copied into `configs/pricing.toml` (`$0.60/M` input,
`$2.20/M` output). The GLM 5.3 response exposed no per-request price, so
`frozen.jsonl` stores `cost_usd=0`. This is not an audited GLM 5.3 bill.

## Failure analysis

There were 163 incorrect tasks:

- **102 MMLU-Pro:** all produced a parseable official option letter, but the
  letter did not match the gold answer. There were no MMLU formatting failures.
- **48 MATH-500:** primarily substantive answer errors. At least one failure is
  grading-normalization-sensitive (`10080` versus gold `10,\!080`), but that
  cannot explain the scale of the MATH regression against GLM 5.2.
- **13 HumanEval:** both semantic and execution failures. Examples include
  missing `typing.List` imports, calls to undefined helpers such as
  `is_palindrome`, `encode_cyclic`, and `poly`, plus incorrect implementation
  behavior on valid inputs.

The main change from GLM 5.2 is a stronger MMLU-Pro profile offset by a
materially weaker MATH-500 result.

## Architecture implication

Do **not** replace `glm-5.2` with `glm-5.3` in the current fusion quorum or
fuser based on this run alone:

- The measured overall lift over GLM 5.2 is not statistically significant.
- The new model trades away math and code performance, both important to SWE
  and agentic use.
- This direct-API benchmark did not initially validate the gateway adapter,
  agent tool calls, or containerized SWE execution.

Those operational prerequisites are now partially covered by the separate
20-instance SWE-bench-Live pilot below. It validates normal tool calls,
gateway ledger accounting, patch production, retry/merge handling, and
official SWE-bench grading. It does not independently cover streaming or
unusual Anthropic thinking-block formats.

The next useful comparison is a controlled paired `fusion-single` evaluation,
with GLM 5.3 serving only as reviewer or fuser, against the current GLM 5.2
path. It must use the same instances, prompts, images, and stopping policy so
the changed variable is the model version rather than execution coverage.

## SWE-bench-Live gateway pilot — 2026-08-20

The direct GLM-5.3 route was then exercised through Fusion Gateway and
SWE-agent on 20 locally cached `SWE-bench-Live/verified` instances. The
official harness resolved 12 of the 20 submitted patches. One additional
instance, `Aider-AI__aider-3806`, exceeded the harness's 1,800-second test
timeout; therefore the completed-instance rate is 12/19 rather than 12/20.

| metric | result |
|---|---:|
| official `verified` split size | 500 |
| pilot predictions submitted | 20 |
| completed by official grader | 19 |
| resolved | 12 |
| unresolved | 7 |
| grader errors | 1 |
| submitted-pilot resolution rate | 12/20 = 60.0% |
| completed-instance resolution rate | 12/19 = 63.2% |
| empty patches | 0 |
| remaining Docker containers | 0 |
| gateway requests | 2,227 |
| provisional gateway ledger cost | $1.2219946 |

Every recorded gateway request settled on `glm-5.3`; no fallback model was
used. The configured price is still a provisional GLM-5.2-derived budget
rate, so the reported cost is not a reconciled provider bill.

The first 17 predictions came from a three-hour primary run. Three instances
entered repeated-action loops (258, 384, and 418 repetitions respectively),
which a cost ceiling alone did not stop because each response was short. Only
those three instances were retried with `per_instance_call_limit=59`. This
SWE-agent version checks that setting after it receives a response, which
means it bounds an instance to at most 60 completed calls. The retry produced
the three remaining patches and completed successfully; the 17 primary and
3 retry predictions were validated for unique instance IDs before grading.

This pilot establishes the end-to-end execution path, not a model-quality
ranking against historical SWE reports: their instance coverage and agent
settings differ. The detailed, reproducible record is
`docs/GLM53_SWE_PILOT_20_REPORT_20260820.md`; generated trajectories,
predictions, ledgers, and grader output remain under
`/home/yangjia/fusion/workload/`.

## Reproducibility and artifacts

Run directory:

```text
/home/yangjia/fusion/workload/glm53_standard_1063_20260819
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
  glm-5.3 1063 5 \
  /home/yangjia/fusion/workload/glm53_standard_1063_20260819 8
```

The benchmark registry entry is `glm-5.3` in `evaluator/validate.py`, using
the existing GLM Anthropic-compatible endpoint and `GLM_API_KEY`.

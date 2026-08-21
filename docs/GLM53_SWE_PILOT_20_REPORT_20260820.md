# GLM-5.3 SWE-bench-Live 20-instance Pilot

**Status:** complete · **Run dates:** 2026-08-19 to 2026-08-20<br>
**Mode:** single-model SWE-agent through Fusion Gateway<br>
**Model route:** Fusion Gateway `glm-5.3` → GLM Anthropic-compatible upstream<br>
**Dataset:** `SWE-bench-Live/verified` · **Pilot coverage:** 20 of 500 instances

## Result

The GLM-5.3 single-model SWE-agent path produced 20 non-empty patches and was
graded with the official SWE-bench harness. Twelve submitted patches resolved
their instances. `Aider-AI__aider-3806` exceeded the harness's
1,800-second test timeout, leaving 19 completed evaluations.

| item | value |
| --- | ---: |
| official split size | 500 |
| pilot predictions submitted | 20 |
| completed by official grader | 19 |
| resolved | 12 |
| unresolved | 7 |
| grader errors | 1 |
| resolution rate, submitted pilot | 12 / 20 = 60.0% |
| resolution rate, completed instances | 12 / 19 = 63.2% |
| empty patches | 0 |
| unstopped containers after grading | 0 |

This is a 20-instance pilot chosen from locally cached evaluation images. It
is not a full 500-instance SWE-bench-Live result and must not be compared as a
controlled ranking against historical reports with different prediction
coverage, images, agent versions, or stopping policies.

## Execution path

SWE-agent used `openai/glm-5.3` against the local Fusion Gateway, which
routed to the GLM Anthropic-compatible upstream. The runner explicitly set:

```text
--agent.model.max_input_tokens 128000
```

SWE-agent directly looks up the configured model name without LiteLLM's
provider-prefix normalisation. The local `sitecustomize.py` adapter registers
the bare `glm-5.3` key for LiteLLM pricing, function-call capability, and
context metadata; the explicit runner setting covers SWE-agent's direct
lookup.

The primary run began at 2026-08-19 22:31:46 JST. It generated 17 predictions
before a three-hour global timeout. Three remaining instances repeatedly
issued identical actions:

| instance | primary-run model calls | repeated exact action count |
| --- | ---: | ---: |
| `aws-cloudformation__cfn-lint-4009` | 390 | 258 |
| `delgan__loguru-1236` | 421 | 384 |
| `pypsa__pypsa-1012` | 481 | 418 |

The per-instance cost limit did not prevent these loops because the individual
responses were short and inexpensive. A follow-up run retried only these three
instances with:

```text
--agent.model.per_instance_call_limit 59
```

In this SWE-agent release, that limit is tested after each response, so a
setting of 59 permits no more than 60 completed model calls. The bounded retry
finished in about ten minutes, produced all three remaining non-empty patches,
and exited with return code zero. The 17 primary and 3 retry predictions were
merged only after checking for exactly 20 unique instance IDs.

## Gateway request accounting

The Fusion Gateway ledger is the source of request usage. GLM-5.3 prices in
the gateway were provisional copies of GLM-5.2's budget guard
($0.60/M input and $2.20/M output), so cost is an operational estimate rather
than a bill-reconciled provider charge.

| scope | calls | input tokens | output tokens | ledger cost |
| --- | ---: | ---: | ---: | ---: |
| primary 17-prediction run | 2,097 | 983,913 | 232,011 | $1.1007720 |
| bounded 3-instance retry | 130 | 125,639 | 20,836 | $0.1212226 |
| **total inference** | **2,227** | **1,109,552** | **252,847** | **$1.2219946** |

All gateway requests settled on `glm-5.3`; no fallback model was used.

## Official grading outcome

Resolved:

- `Azure__azure-sdk-for-python-40487`
- `BerriAI__litellm-10284`
- `Kozea__Radicale-1766`
- `Textualize__textual-5743`
- `ag2ai__faststream-1796`
- `ansible__ansible-lint-4566`
- `apify__crawlee-python-1155`
- `arviz-devs__arviz-2404`
- `beancount__beancount-931`
- `beeware__briefcase-2035`
- `flexget__flexget-4306`
- `kozea__weasyprint-2416`

Unresolved:

- `Pyomo__pyomo-3588`
- `aiogram__aiogram-1670`
- `amoffat__sh-750`
- `aws-cloudformation__cfn-lint-4009`
- `beetbox__beets-5437`
- `delgan__loguru-1236`
- `pypsa__pypsa-1012`

The single grading error was `Aider-AI__aider-3806`; its test execution
exceeded the official 1,800-second harness timeout. It is neither a resolved
nor an unresolved outcome.

## Artifacts and reproduction

All generated experiment data is intentionally retained outside the Git
repository:

```text
/home/yangjia/fusion/workload/
├── swe_pilot_glm53_20260819/
├── swe_pilot_glm53_retry_3_20260820/
├── glm53_swe_pilot_20_merged_20260820.json
├── swe_pilot_glm53_20260819.glm53_pilot20_grade_20260820.json
├── glm53_pilot_20_20260819_gateway.sqlite
├── glm53_pilot_retry3_20260820_gateway.sqlite
└── glm53_pilot20_grade_20260820.full.log
```

The retained helpers are:

```bash
bash /home/yangjia/fusion/workload/start_glm53_pilot_gateway_8812.sh
bash /home/yangjia/fusion/workload/run_glm53_swe_pilot_20_20260819.sh
bash /home/yangjia/fusion/workload/start_glm53_retry_gateway_8813.sh
bash /home/yangjia/fusion/workload/run_glm53_swe_pilot_retry_3_20260820.sh
bash /home/yangjia/fusion/workload/grade_glm53_swe_pilot_20_20260820.sh
```

The grading command uses the official `swebench.harness.run_evaluation` module
against `SWE-bench-Live/SWE-bench-Live`, split `verified`, with the merged
prediction file and a 10,800-second per-instance evaluation timeout.

## Implications

The experiment validates the full operational path: gateway routing,
GLM-5.3 agent tool use, SWE-agent patch generation, bounded retry, prediction
merge, Docker teardown, and official grading.

For any broader run, retain the per-instance call bound or introduce a
repeated-action guard. A budget ceiling alone is insufficient to terminate
low-token action loops promptly. The next comparison should hold the task
set, Docker images, agent configuration, prompts, and stopping policy fixed
while changing only the model or the `fusion-single` reviewer/fuser role.

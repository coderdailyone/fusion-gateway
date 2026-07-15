# M3a Routing-Signal Pilot — Result

**The pool determines everything.** A cheap-only pool has no routable signal
(one model dominates); a **cheap + strong** pool has strong signal. Concretely:

- **NO_GO** — DeepSeek + GLM-5.2 (both "cheap-ish"): DeepSeek Pareto-dominates.
- **GO** — DeepSeek + Claude Sonnet 5 (cheap + strong): route the cheap model by
  default, escalate to the strong one only when needed → **oracle quality at
  86.5% lower cost** than always using the strong model.

**Recommendation: proceed to M3b** (full sampling + router training) with a
cheap+strong pool. The routing thesis is validated: real value exists when the
pool has a genuine cost–quality spread.

Suite: `configs/suite.manifest.json` (1063 locked); stratified subset n=150
(85 mmlu_pro + 42 math + 23 humaneval), seed 1234. Models via LiteLLM.

## Run A — cheap-only pool (NO_GO)

| model | accuracy | cost (150) |
|---|---|---|
| **deepseek-chat** | **0.860** | **$0.0143** |
| glm-5.2 | 0.853 | $0.1597 |

DeepSeek is both more accurate AND ~11× cheaper → best_static = DeepSeek.
quality_headroom 0.040 (<0.05), **cost_savings −0.439** (matching oracle quality
costs *more*, since DeepSeek is already cheapest). **verdict NO_GO.** When one
model is best-and-cheapest, no routing beats "always use it."

(Kimi K2 was dropped: first run truncated at max_tokens=2048 — a reasoning model
spent the cap on hidden reasoning and returned 86 empty answers; the re-run hit
the account's billing-cycle quota. Two real bugs it exposed were fixed: reasoning
models now use max_tokens=8192, and the math scorer handles `\frac13` shorthand.)

## Run B — cheap + strong pool (GO)

| model | accuracy | cost (150) |
|---|---|---|
| claude-sonnet-5 | **0.880** | $0.3169 |
| deepseek-chat | 0.860 | **$0.0144** |

- **best_static:** claude-sonnet-5 (0.880) — but **22× more expensive**.
- **oracle_accuracy 0.927** → routing the right model per task beats *even Sonnet
  alone* by 4.7pt. quality_headroom 0.047.
- **iso_quality_cost $0.0427** vs best_static $0.3169 → **cost_savings 0.865**.
- **verdict GO** (cost signal, 0.865 ≫ 0.15).

**Why the signal is real (not a scorer artifact):** Sonnet is healthy on every
source (mmlu 75/85, math 35/42, code 22/23; only 2 empty outputs). The two models
are **complementary** — 7 tasks DeepSeek gets that Sonnet misses, 10 the reverse
(DeepSeek even edges Sonnet on math, 37 vs 35). Neither dominates, so a router
that picks the right one per task reaches 0.927 — above both — and, since DeepSeek
covers most tasks cheaply, does so at ~1/7 the always-Sonnet cost. This is the
RouteLLM value proposition, empirically present in our own data.

## Interpretation

Routing/fusion earns its keep only when the pool has a real cost–quality
tradeoff. Run A shows the failure mode (a dominant cheap model → nothing to
route to). Run B shows the win (a strong-but-expensive model that wins a
meaningful, distinct task slice → "cheap by default, strong when needed" gives
near-oracle quality far cheaper than the strong model alone).

The oracle / iso-quality numbers are **offline upper bounds** — a learned router
won't be perfect. But an 86.5% cost headroom is large enough that even a modest
router captures real value. Learning that router is exactly M3b.

## M3b pool recommendation

- **cheap leg:** DeepSeek (deepseek-chat) — strong and ~$0.01/150-tasks.
- **strong leg:** Claude Sonnet 5 (and optionally Opus 4.8 / gpt-5.5 for a richer
  frontier). Registered in `configs/pricing.toml` + `evaluator.validate.MODELS`.
- **cost caveat:** the strong models are 20–40× DeepSeek per token; full 1063-task
  sampling on a strong model is the main M3b spend — budget for it explicitly.

## Cost

Pilot total ≈ **$0.7** (Run A ~$0.30; Run B Sonnet $0.317; gpt-5.5 aborted ~$0.05;
probes ~$0.02). It answered a project-defining question — is routing viable? — for
under a dollar. Frozen outputs remain re-scorable offline.

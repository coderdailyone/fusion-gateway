# M3a Routing-Signal Pilot — Result

**Verdict: NO_GO for the current pool.** A ~150-task × 3-model pilot (~$0.30)
shows that DeepSeek **Pareto-dominates** the pool — it is simultaneously the most
accurate and by far the cheapest model, so no routing/fusion can beat it. Do not
spend on the full M3b sampling with this pool.

## Run

- Suite: `configs/suite.manifest.json` (1063 locked); stratified subset n=150
  (85 mmlu_pro + 42 math + 23 humaneval), seed 1234.
- Models: `deepseek-chat`, `glm-5.2`, `kimi-k2` (via LiteLLM).
- run_dir: `evaluator/runs/pilot_20260715T035622Z`.

## Trustworthy result (DeepSeek + GLM-5.2)

| model | accuracy | cost (150 tasks) |
|---|---|---|
| **deepseek-chat** | **0.860** | **$0.0143** |
| glm-5.2 | 0.853 | $0.1597 |

- **best_static:** deepseek-chat (higher accuracy AND ~11× cheaper).
- **oracle_accuracy:** 0.900 → **quality_headroom 0.040** (below the 0.05 bar).
- **disagreement_rate:** 0.087 (the two strong models mostly agree).
- **iso_quality_cost:** $0.0206 → **cost_savings −0.439** (matching oracle quality
  costs *more* than always using DeepSeek, because DeepSeek is the cheapest and its
  misses can only be covered by the ~11× pricier GLM).
- **verdict: NO_GO** — neither signal clears its threshold.

Per-model × source accuracy: DeepSeek 0.87/0.83/0.83, GLM-5.2 0.84/0.83/0.91
(mmlu/math/humaneval).

## Kimi is excluded (data unusable this cycle)

Kimi K2 could not be scored honestly:
1. **First run:** as a reasoning model it produced **86 empty answers at
   max_tokens=2048** — the cap was spent on hidden reasoning (mmlu 5/85). A
   measurement artifact, not weakness (it scored humaneval 22/23, math 31/42).
2. **Re-run at max_tokens=8192:** all 150 calls returned *"You've reached your
   usage limit for this billing cycle"* — the Kimi account is out of quota.

Two real bugs the pilot caught were fixed regardless (they helped DeepSeek/GLM):
- `make_completion_fn` now takes `max_tokens`; reasoning models use 8192.
- the math scorer now handles `\frac13` shorthand (no braces).

## Why NO_GO, and what would change it

Routing/fusion earns its keep only when the pool has a genuine **cost–quality
tradeoff** — a model that is *better on some tasks but more expensive*, so routing
to it *only when needed* beats every static choice. This pool doesn't have that:
DeepSeek is the best AND the cheapest, so the rational static policy is "always
DeepSeek," and no router can improve on it. (Even a fixed Kimi would need to beat
DeepSeek on a meaningful, affordable task slice to create signal.)

**To get routable signal, change the pool, not the method:** pair a cheap model
(DeepSeek) with a **stronger, more expensive frontier model** (e.g. a GPT-5.x or
Claude tier) that actually wins the tasks DeepSeek misses. That is the classic
weak+strong routing setup (RouteLLM) where "cheap by default, strong when needed"
can Pareto-beat both. That is the decision to make before re-running a pilot.

## Cost

Total pilot spend ≈ **$0.30** (DeepSeek $0.014 + GLM $0.16 + Kimi partial/errors
~$0.13 + smoke ~$0.01). The pilot cost ~1/20th of a full run and prevented an
uninformative $5–10 sampling pass. Frozen outputs remain re-scorable offline.

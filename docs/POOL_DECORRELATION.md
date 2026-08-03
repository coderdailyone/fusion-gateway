# Pool decorrelation — how much headroom is a different model worth?

**Date:** 2026-07-31 · **Cost:** $0.98 total ($0.11 re-sampling flash, $0.87 sampling v4-pro; the Claude/GPT comparisons were $0 from M2c's frozen outputs)

## Why

M5 and M6 each measured a **+0.7 pt ceiling** over the best pool member while the
panel oracle sat 4–6 pt higher, and the M6 report named the cause: the pool's
models fail *together*, so no aggregation rule can recover an answer nobody
produced. It also named the only lever that moves that ceiling — **add a model
with a different pretraining lineage**, not another Chinese chat model tuned on
similar data.

That lever had never been pulled. This is the measurement.

Everything is scored with the same official scorers and the same `oracle()` the
published M5 numbers came from, so the figures are directly comparable.
Reproduce with `PYTHONPATH=. .venv/bin/python scripts/pool_oracle.py`.

## The headline

On the 1063-task locked suite, over the 985 tasks every compared model answered:

| pool | oracle | vs domestic-only |
|---|---:|---:|
| domestic (deepseek-v4-flash + glm-5.2) | 0.9036 | — |
| + deepseek-v4-pro | 0.9289 | **+2.5 pt** |
| + claude-opus-4-8 | 0.9442 | **+4.1 pt** |
| + both | 0.9523 | +4.9 pt |

The domestic pair is **wrong together on 95 tasks**. Of those, `v4-pro` gets
**26.3%** right and `claude-opus-4-8` gets **42.1%**.

Three things follow:

1. **Decorrelation is real and measurable.** A different-lineage model recovers
   roughly 42% of the domestic pool's blind spot. The M6 report's diagnosis was
   correct.
2. **A same-vendor, different-recipe model is worth something too** — 26% is not
   nothing — but cross-vendor is worth ~1.6× more.
3. **Their blind spots do not overlap.** Adding both beats adding either, so the
   two are complementary rather than substitutes.

**This is a ceiling, not a gain.** The oracle is "at least one member is right".
M5 and M6 both showed an aggregator captures only a fraction of it on prose.
The right reading is: *the ceiling worth chasing moved from 0.9036 to 0.9442* —
whether anything can reach it is a separate question, and this project's only
evidence of an aggregator reaching its ceiling is the code cascade at 0.994,
which worked because **objective verification existed**.

## The DeepSeek update, and where our ruler disagrees with the vendor

DeepSeek shipped a v4-flash update on 2026-07-31 announcing *"Agent capability
greatly enhanced, benchmarks far exceeding V4-Pro-Preview"*. We re-sampled
flash and sampled v4-pro on the same suite.

| model | accuracy | | |
|---|---:|---|---|
| deepseek-v4-flash (frozen 2026-07-18) | 0.8609 | | |
| deepseek-v4-flash (2026-07-31, post-update) | 0.8548 | vs old: net −6, **p = 0.60** | no change |
| **deepseek-v4-pro** (V4-Pro-Preview) | **0.8751** | vs new flash: net +20, **p = 0.065** | pro is *better* |

By source, the gap is almost entirely code:

| source | flash (new) | v4-pro | delta |
|---|---:|---:|---:|
| humaneval | 0.8701 | **0.9481** | **+7.8 pt** |
| math | 0.9306 | 0.9271 | −0.4 |
| mmlu_pro | 0.8103 | 0.8269 | +1.7 |

**This is not evidence the vendor is wrong.** They claim *agent* capability;
this suite measures single-turn code, maths and knowledge. A model can be better
in an agent loop and worse on one-shot HumanEval. What it does mean is:

- **The update did not move our numbers** (p = 0.60). Whatever improved is not
  something this suite can see.
- **For single-turn code quality, `v4-pro` is materially better**, and the
  vendor's framing would lead you to the opposite choice.
- **We cannot evaluate the thing they improved.** We have no agent-capability
  grader — the same gap that leaves M9's tool-call fusion unmeasured.

## What this changes

- The gateway's panel is unchanged. Nothing here is a config recommendation yet:
  a raised oracle is permission to try, not a result.
- The highest-value next step is still **a grader for tool calls / agent steps**.
  It is the one thing that would let us evaluate both the vendor's claim and our
  own M9 work, and it is the condition under which this project has previously
  seen an aggregator actually reach its ceiling.
- If the pool is changed, the evidence favours **one cross-vendor member** over
  a second same-vendor one — but note that abandons the "domestic-only"
  constraint the research line was originally framed around.

## Caveats

- `deepseek-v4-pro` pricing in `configs/pricing.toml` is **assumed** (2× flash)
  and unverified against DeepSeek's published table.
- 78 of 1063 tasks are excluded from the comparison because at least one model
  has no `ok` row (mostly transient `SSL: UNEXPECTED_EOF` and 503 "Service is
  too busy" during the public-beta launch). A failed row is never retried by the
  sampler's resume logic, so those stay missing.
- kimi-k3 is absent throughout: its quota is exhausted, and the 492 tasks it did
  complete are **not** a representative subset — every model scores 4–6 pt
  higher on them, so oracles computed there are inflated and near-ceiling.
- `v4-pro` costs ~8× flash per suite run ($0.87 vs $0.11) and is ~3× slower.

## Tooling added

- `scripts/pool_oracle.py` — oracle over arbitrary pool compositions from frozen
  samples, $0. Includes a per-task-set difficulty control, which is what caught
  the kimi-subset inflation above.
- `scripts/sample_one.py` — sample one model over the locked suite, resumable
  into an existing run dir.
- `scripts/sample_one_par.py` — the same, concurrent. The serial sampler needed
  ~7 hours for `v4-pro`; this took ~33 minutes. Calls fan out to a thread pool;
  **writes and spend accounting stay serial under a lock** (`append_frozen`
  opens per row, so concurrent writers interleave), and the ceiling is enforced
  at dispatch so in-flight calls are recorded rather than paid for and discarded.

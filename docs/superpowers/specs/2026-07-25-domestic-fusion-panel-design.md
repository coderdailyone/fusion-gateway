# M5 — Domestic Fusion Panel (cheap models fused to frontier level)

**Status:** design approved 2026-07-25
**Milestone:** M5 (fusion family; the "fusion" the project is named for)

## Why

Every tier so far picks **one** model per task — the learned router (M3) and the
escalation cascades (M3 code cascade, M4 agentic). That caps quality at the best
single model in the pool. The project's namesake move is the one never tested:
**run a panel of cheap models on the same task and fuse their answers into one**.

The thesis: **three cheap domestic models, cross-reviewed and fused by a fourth
domestic model, reach frontier single-model quality** — without ever calling a
frontier model. If true, it is a far stronger version of "Useful Intelligence per
Dollar" than routing: not *choosing* cheap when it suffices, but *manufacturing*
frontier-grade answers out of cheap parts.

**Standing evidence to beat (M2c, 1063 tasks, official scoring):**
claude-opus-4-8 **0.913** · gpt-5.5 0.905 · claude-sonnet-5 0.898 ·
**gpt-5.6-sol 0.894** · deepseek-chat 0.859 · glm-5.2 0.842.

**Prior warning from this project (docs/DESIGN.md):** the earlier online-routing
research line found dynamic routing saved cost but **never beat the best static
in-pool model on quality**, and that **LLM judging was too noisy to train on**
(repeat-scoring agreement only 0.63–0.74 on fixed outputs). M5 must confront both:
it must beat the best in-pool model, and its cross-review must be treated as
noisy evidence, not truth.

## Positioning (locked with the user)

- **Mechanism:** candidates → **cross-review** → fuser **synthesizes a new answer**
  (not merely picks the best candidate).
- **Fuser is domestic.** Using a frontier model anywhere (fuser or judge) would
  make "reaches frontier level" circular. Non-negotiable.
- **Both tiers:** standard (1063) first to establish the claim, then hard (657)
  to probe the ceiling.
- **Fuse everything first, gate later.** Measure the true quality ceiling before
  optimizing cost; because outputs are frozen, gating policies are simulated
  afterwards at **$0**.
- Same disciplines as M2c/M2d/M4: official scoring, `evaluator/` isolated from
  `gateway.*`, frozen re-scorable outputs, preflight budget gate, secrets only in
  `runs/secrets/.env`. Standard and hard manifests stay byte-unchanged.

## Panel and fuser

| role | model | notes |
|---|---|---|
| candidate | `deepseek-chat` | strongest domestic (0.859); frozen 1063 exists |
| candidate | `glm-5.2` | 0.842; frozen 1063 exists |
| candidate | `kimi-k3` | quota refreshed by the user 2026-07-25; **candidates must be sampled** (M2c only has 58–67 rows per shard) |
| **fuser** | `glm-5.2` | deliberately *not* the strongest candidate, so fusion cannot degenerate into "always echo deepseek". If results disappoint, re-run fusion with `deepseek-chat` as fuser over the frozen candidates (cheap, no re-sampling). |

## Free gate before paid work: the oracle ceiling

Before spending on review/fusion, compute — from **existing frozen outputs** —
the **oracle** of the domestic panel: fraction of tasks where *at least one*
candidate is correct. This is fusion's theoretical ceiling.

- If **oracle < 0.894**, no fusion strategy can reach gpt-5.6-sol on this pool;
  the correct response is to change the pool, not to tune prompts. Report and stop.
- If **oracle ≥ 0.894**, the headroom is real and paid work is justified.

This costs $0 and is the milestone's first acceptance gate. (It requires kimi
candidates, so it runs after kimi sampling.)

## Architecture — `evaluator/fusion/`

Three units, each independently testable; none imports `gateway.*`.

```
evaluator/fusion/
  panel.py    # assemble N candidates per task from frozen runs -> PanelCase
  review.py   # cross-review: each model reviews the OTHERS' candidates -> structured verdicts
  fuse.py     # fuser reads (task + candidates + reviews) -> final answer
```

**Data flow per task:**
1. `panel.assemble(task_id)` → `PanelCase(task, candidates: {model: text})`, read
   from frozen M2c/M2d runs (**$0**).
2. `review.cross_review(case, completion_fn)` → for each reviewer model, a
   structured verdict on each *other* candidate:
   `{candidate_model: {verdict: "correct"|"wrong"|"unsure", reason: str}}`.
   Self-review is excluded (avoids self-preference bias). Structured, not prose,
   so the fuser gets evidence and we can measure reviewer agreement directly.
3. `fuse.fuse(case, reviews, completion_fn)` → the final answer text.
4. Scored by the **existing official scorers** (`evaluator/scorers/*` →
   `evaluator/official/*`) and frozen — identical grading to single models, so
   numbers are directly comparable.

## Prompts (the crux)

Both review and fusion prompts are built in `evaluator/fusion/prompts.py`,
reusing the per-source answer-format instructions from
`evaluator/official/prompts.py` so a fused answer obeys the same output contract
the graders expect (MMLU/GPQA: ends with `The answer is (X).`; math: `\boxed{}`;
code: code only). A correct answer in the wrong format scores wrong.

**Fusion prompt rules (explicit):**
- If candidates agree and reviews raise no objection → **adopt that answer**
  (do not rewrite; avoids breaking correct answers).
- If candidates disagree → adjudicate **using the reviews' specific objections**,
  and synthesize a corrected answer (may combine partial work from several
  candidates).
- Always emit the source's official answer format.

## Diagnostics that decide the verdict

The report must separate, against the best single domestic candidate:

- **fix rate** — tasks the best candidate got wrong that fusion got right.
- **break rate** — tasks the best candidate got right that fusion got wrong.
  (Fusion talking itself out of a correct majority is the known failure mode.)
- **net gain = fix − break.** If net gain ≤ 0, fusion adds nothing and is cut.
- **reviewer agreement** — how often reviewers agree on the same candidate;
  quantifies the 0.63–0.74 judge-noise problem on this pool.
- **format failure rate** of fused answers.

## Cost-gating, simulated after the fact ($0)

With all fused outputs frozen, simulate gate policies without new spend:
- fuse only when candidates **disagree** (agreement → adopt free);
- fuse only when cross-review flags a problem.
Report the resulting **cost–quality curve** and pick the operating point. This is
how M5 rejoins the project's Pareto framing without paying for it up front.

## Sampling scope and budget

1. **kimi-k3 candidates**: sample the full standard tier (1063) — required for
   the panel; user refreshed quota on 2026-07-25.
2. **Cross-review + fusion** on the standard tier: 3 reviews + 1 fusion per task
   ≈ 4 domestic calls/task. Domestic pricing is ~$0.0001–0.001/call, so the whole
   1063-task run is expected in the **$3–10** range.
3. **Hard tier (657)** repeats the same pipeline after the standard tier verdict.

**Budget gate:** reuse `scripts/resample_official.py::run_budgeted`'s discipline —
preflight estimate, hard ceiling, resumable, frozen. Ceiling **$25** for the
standard tier; hard tier gated on the standard-tier result. A **paid smoke on 20
tasks** precedes each full run, reporting measured per-task cost and a first
fix/break signal.

## Report — `docs/M5_FUSION_REPORT.md`

- Oracle ceiling of the domestic panel vs the frontier bar (0.894 / 0.913).
- Fusion accuracy with **Wilson 95% CI**, versus: best single domestic model,
  gpt-5.6-sol, and claude-opus-4-8; **McNemar** significance vs the best domestic
  single and vs sol.
- fix / break / net gain, reviewer agreement, format failure rate.
- The simulated cost–quality gate curve.
- Verdict: does a domestic panel reach frontier single-model quality, and at what
  cost per correct answer.

## Error handling

- Missing/error candidate for a task → panel degrades to the remaining
  candidates, recorded; never a crash. Tasks with fewer than 2 candidates are
  excluded from the headline comparison and counted.
- Malformed review output → that verdict is dropped (counted); fusion proceeds on
  the remaining evidence.
- Fused answer unparseable by the official scorer → scored incorrect (same rule
  as single models) and counted in format-failure diagnostics.
- Transient mirror 5xx/timeout → prune-and-retry (M2c pattern), resumable.
- Kimi quota exhaustion mid-run → stop cleanly, report how many tasks have full
  panels; the oracle gate and comparisons run on the complete-panel subset.

## Testing

- Unit: `panel.assemble` (including degraded panels), `review` parsing (well-formed
  and malformed), `fuse` prompt construction (agreement path vs disagreement path),
  all with stubbed completion functions — no network.
- Format test: fused answers for each source type carry the official answer
  format, verified against the same extractors the graders use.
- Isolation test: `evaluator/fusion/*` imports no `gateway.*`.
- Determinism: re-scoring frozen fused outputs twice is identical.

## Acceptance criteria

1. kimi-k3 candidates sampled for the standard tier under the budget gate;
   panel completeness reported.
2. **Oracle gate** computed at $0 and reported (proceed only if ≥ 0.894, else
   stop and report the pool is insufficient).
3. Cross-review + fusion run over the standard tier, frozen, resumable, under the
   $25 ceiling, preceded by a 20-task paid smoke.
4. `docs/M5_FUSION_REPORT.md` published with Wilson CIs, McNemar vs the best
   domestic single and vs gpt-5.6-sol, fix/break/net gain, reviewer agreement,
   and the simulated gate curve.
5. Hard tier repeated (or explicitly deferred with the standard-tier rationale).
6. No frontier model used in any panel/review/fusion role; `evaluator/` isolated;
   manifests byte-unchanged; scoring deterministic.

## Non-goals

- Not replacing M3 routing or M4 agentic results — M5 is an additional strategy
  and a measurement deliverable.
- Not wiring fusion into the live gateway (a later milestone, gated on M5's
  result).
- No LLM judge in **scoring** — cross-review informs fusion only; grading stays
  objective (extraction / math equivalence / sandboxed execution).
- No new frontier-model spend; frontier numbers come from existing M2c results.

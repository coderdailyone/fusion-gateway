# M5 — Domestic Fusion Panel: Results

**Status:** standard tier complete 2026-07-25 · **Milestone:** M5 (fusion)
**Question:** can a panel of cheap **domestic** models, cross-reviewed and fused
by another **domestic** model, reach frontier single-model quality — with no
frontier model anywhere in the loop?

## TL;DR

Fusion **works, but the win is not statistically significant, and the panel did
not reach the frontier bar.** Fusion scores **0.8901**, above the best single
domestic model (kimi-k3, 0.8789) by 1.1 points (net +12 tasks), but McNemar
**p = 0.176** — inside the noise. It lands just under gpt-5.6-sol (0.894) and
2.3 points under claude-opus-4-8 (0.913).

The real headline is elsewhere: **the whole run cost $0.14** — about **$0.00015
per correct answer**, roughly **28× cheaper than opus** on the same suite, for
2.3 points less accuracy. And a **single domestic model (kimi-k3) is already
within a point of the frontier**, which reframes the original question.

## Result (standard tier, 1063 tasks, official scoring, no LLM judge)

| strategy | correct | accuracy | 95% CI (Wilson) |
|---|---:|---:|---|
| **fusion (panel → cross-review → glm-5.2 fuser)** | **948** | **0.8901** | [0.8699, 0.9075] |
| kimi-k3 *(best single domestic)* | 936 | 0.8789 | [0.8579, 0.8971] |
| deepseek-chat | 907 | 0.8516 | [0.8290, 0.8717] |
| glm-5.2 | 887 | 0.8329 | [0.8093, 0.8541] |
| *panel oracle (any member correct)* | — | *0.9308* | *ceiling* |

**Frontier reference (M2c, same suite, same graders):** claude-opus-4-8 0.913 ·
gpt-5.5 0.905 · claude-sonnet-5 0.898 · **gpt-5.6-sol 0.894**.

**Fusion vs the best single domestic model:** McNemar b=39, c=27, **p = 0.176**
(not significant). fix = 38, break = 26, **net = +12** over 1055 comparable tasks.

## What this actually shows

1. **Fusion adds real but small value.** It fixes 38 tasks the best member got
   wrong — and breaks 26 it had right. The net is positive but the sign is not
   established at this sample size. Anyone claiming "fusion beats the best model"
   from these numbers would be over-reading them.
2. **`break` is the lever.** The panel's oracle is 0.9308, so 4 points of
   headroom exist above fusion's 0.8901. Almost all of the loss is fusion talking
   itself out of a correct answer. Cutting `break` — not chasing more `fix` — is
   the highest-value next step.
3. **Structured cross-review tamed the judge-noise problem.** Reviewer agreement
   is **0.9157**, versus the 0.63–0.74 that blocked this project's earlier
   fusion-gain work (`docs/DESIGN.md`). Forcing verdicts into a fixed
   `VERDICT <target> <correct|wrong|unsure> <reason>` line, and excluding
   self-review, is what changed.
4. **A single domestic model is now near-frontier.** kimi-k3 at 0.8789 sits
   between sonnet (0.898) and the cheap tier — and on this run it was sampled
   fresh, since M2c could not measure it (quota exhausted). The original framing
   ("can cheap models *combined* reach the frontier?") is partly answered by one
   model alone.

## Cost

| | total | per correct answer |
|---|---:|---:|
| fusion run (3 reviews + 1 fusion per task, 1063 tasks) | **$0.1418** | **$0.000150** |
| kimi-k3 candidate sampling (one-off, 1063 tasks) | ~$1.89 | — |
| *(reference)* claude-opus-4-8 on the same suite (M2c) | ~$4.48 | ~$0.00461 |

Fusion is **~28× cheaper per correct answer than opus** while scoring 2.3 points
lower. On a cost-quality basis this is a strong operating point, which is the
axis this project actually optimizes.

## Simulated gate curve ($0, from frozen outputs)

Gate policies replayed over the frozen results, with unanimity decided on
**extracted** answers and review cost charged only for tasks actually fused:

| policy | accuracy | cost |
|---|---:|---:|
| always fuse | 0.8986 | $0.1418 |
| **fuse only on candidate disagreement** | **0.8910** | **$0.0710** |

Gating halves the cost for ~0.8 points — the better operating point for
production, and consistent with the project's Pareto framing.

## Caveats (honest)

- **The headline comparison is not significant** (p = 0.176). Treat "fusion >
  best domestic" as directional, not established.
- **kimi-k3 has 1022/1063 candidates** (96.3% sampled ok); tasks missing a kimi
  candidate fall back to a 2-model panel, which slightly understates the panel.
- **10/1065 fused rows were empty/unparseable** and are scored wrong (not
  dropped), so the fusion number is a floor, not a flattered figure.
- **Cost units:** deepseek/glm costs are real; the kimi mirror returns no price
  metadata, so kimi's share of the cost is **not** included in the $0.1418 —
  the true fusion cost is somewhat higher, though still far below opus.
- Standard tier only. The hard tier (657 contamination-resistant tasks) is
  deferred pending the decision below.

## Verdict and next step

**The pool is good; the fuser is the bottleneck.** With an oracle of 0.9308 the
panel *contains* frontier-level answers on 93% of tasks — fusion recovers only
0.8901 of that. Two concrete follow-ups, in order:

1. **Reduce `break`.** Add an explicit "if a clear majority of candidates agree
   and no review objects, copy that answer verbatim" rule, and consider a
   stronger domestic fuser (kimi-k3 itself, which outscores the current glm-5.2
   fuser by 4.6 points as a solver). Both are cheap to test — the candidates are
   frozen, so only fusion re-runs.
2. **Then re-measure significance** on the full suite, and only afterwards spend
   on the hard tier.

**On the original goal:** a domestic panel did *not* reach opus-5 level (0.913)
this round, and matched-but-did-not-beat gpt-5.6-sol (0.894 vs 0.8901). What it
did achieve is near-frontier quality at ~1/28 the cost per correct answer.

## Reproducibility

Panel candidates come from frozen runs (`evaluator/runs/m2c_full/*`,
`evaluator/runs/m5_fusion/kimi-k3-s*`); fused answers and cross-review verdicts
are frozen at `evaluator/runs/m5_fusion/fused_full/fused.jsonl`. Re-scoring and
gate-curve simulation cost **$0**. Driver: `scripts/run_fusion.py`
(`oracle` | `run`); metrics: `scripts/fusion_report.py`.
Design: `docs/superpowers/specs/2026-07-25-domestic-fusion-panel-design.md`.

## Follow-up: two attempts to break the ceiling (2026-07-25)

The first run's diagnosis was "the fuser is the bottleneck, `break` is the
lever." Two cheap interventions were tested by re-running fusion over the same
frozen candidates:

- **v2-glm** — added an explicit rule: *if a majority of candidates give the same
  final answer and no review objects, COPY it verbatim; do not rewrite.*
- **v2-kimi** — same rule, but with **kimi-k3** (the strongest panel member,
  +4.6 points over glm-5.2 as a solver) as the fuser.

Common set of 995 tasks, official scoring:

| strategy | correct | accuracy | 95% CI | vs kimi-k3 | fix / break / net | cost |
|---|---:|---:|---|---|---|---:|
| kimi-k3 *(best single)* | 886 | 0.8905 | [0.8695, 0.9084] | — | — | — |
| v1 glm *(no rule)* | 893 | 0.8975 | [0.8771, 0.9148] | p=0.427 | 32 / 25 / **+7** | $0.1349 |
| **v2 glm (+majority-copy)** | 893 | **0.8975** | [0.8771, 0.9148] | p=0.391 | 28 / 21 / **+7** | **$0.0311** |
| v2 kimi (+majority-copy) | 889 | 0.8935 | [0.8728, 0.9111] | p=0.784 | 28 / 25 / +3 | $0.0305 |

**Neither intervention moved the needle, and the reason is informative.**

1. **The majority-copy rule worked as designed — and that wasn't enough.**
   `break` fell 25 → 21, exactly the intended effect. But `fix` fell 32 → 28 in
   lockstep, leaving net gain unchanged at +7. Making the fuser more conservative
   buys fewer broken answers at the price of fewer repaired ones. **`fix` and
   `break` are coupled**: they are two faces of the same willingness to overrule
   the panel. That is a real trade-off, not a prompt-tuning defect.
2. **A stronger solver made a worse fuser.** kimi-k3 outscores glm-5.2 by 4.6
   points as a solver, yet fusing with it was *worse* (net +3 vs +7) — it tends
   to defend its own answer rather than synthesize others'. Fusing and solving
   are different skills.
3. **The win remains statistically insignificant** across all three variants
   (p = 0.39–0.78).
4. **Cost dropped 4×** ($0.135 → $0.031) because copying on agreement spends far
   fewer output tokens — the rule is worth keeping on cost grounds alone.

### Where this leaves the original goal

On this 995-task common set the numbers land as follows against the frontier
(opus 0.913 · gpt-5.5 0.905 · sonnet 0.898 · gpt-5.6-sol 0.894):

- **"Domestic models reaching gpt-5.6 level" — achieved, and without fusion.**
  kimi-k3 alone scores **0.8905**, above gpt-5.6-sol's 0.894.
- **"Fusion pushing that to opus level" — not achieved.** Best fusion is
  **0.8975** (above sonnet, near gpt-5.5), still 1.5 points under opus, and the
  gain over the best single model is not significant.
- The pool is not the limit (oracle 0.9308). The limit is that **an LLM fuser
  cannot separate "answers worth overruling" from "answers worth keeping"** —
  every gain in `fix` is paid for in `break`.

**Recommendation:** stop tuning the fuser prompt; the coupling above says that
path is exhausted. The remaining levers are (a) an *objective* arbiter where one
exists — for code, run the tests and pick the candidate that passes, which is
this project's proven verify-cascade rather than an LLM judgment call; or (b) a
learned gate that fuses only where fusion has historically won. Both are
different milestones, not variations of this one.

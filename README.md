# Fusion Gateway

An OpenRouter-style, industrial-grade **LLM fusion gateway**. It receives
OpenAI-compatible requests and decides — from public task features alone —
whether to answer with a single model, a cheap→strong **cascade**, or a
multi-model **panel** whose outputs are **fused** only when doing so is worth
the extra cost. The goal is a **cost–quality Pareto SOTA**: a dynamic policy
whose cost/quality curve envelopes every static single-model baseline.

## Why

Dynamic routing reliably saves cost and latency, but naive fusion often fails
to beat the best static policy on quality. This project treats that as the core
problem: fuse only when a learned disagreement gate says it pays off, and hold
every fusion/cascade point to the bar of **expanding the Pareto frontier or
being cut**.

## Design at a glance

- **OpenAI-compatible API** (`/v1/chat/completions`, streaming supported).
- **SQLite is the only execution truth** — append-only, replayable event
  traces; a cost **ledger** with preflight→settle on every real call.
- **Budget guardrails** — per-milestone caps, alert at 80%, trip (kill switch)
  at 100%, cleared only by explicit admin action.
- **Static routing first**, then a learned cost-aware router
  (`utility = quality − λ·cost`) trained on a public benchmark suite.
- **Strict evaluation isolation** — judges and reference answers never enter
  routing inputs; validation is group-by-task; a judge must clear a
  repeat-scoring sign-agreement floor before its labels are trusted.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design and
[`docs/DISCIPLINES.md`](docs/DISCIPLINES.md) for the engineering rules the code
is held to. Decision records live in [`docs/adr/`](docs/adr/).

## Status

Early. M0 (governance, disciplines) is in place; M1 (minimal production
gateway) is under active development.

## License

TBD.

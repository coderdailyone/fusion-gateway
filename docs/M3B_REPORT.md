# M3b — Cost-Aware Router Report

## Learnability gate (free — on M3a pilot data)

Before any paid full sampling, we test the router's core premise: can public
features (problem TF-IDF + source + length) predict each model's per-task
correctness? Run on the 150-task pilot (DeepSeek + Claude Sonnet 5), group-by-task
CV, per-fold featurizer refit.

| model | per-model CV AUC |
|---|---|
| **deepseek-chat** | **0.66** |
| claude-sonnet-5 | 0.50 |

**Verdict: GO** (threshold 0.55; DeepSeek passes).

**Reading:** DeepSeek's *failures* are predictable from the problem text (AUC 0.66)
— the router can learn "escalate to a strong model when DeepSeek is likely wrong."
Sonnet's correctness is near-unpredictable (0.50), which is expected for a strong,
uniform model; routing only needs to predict when the *cheap* model fails, and it
can. The signal is real but **modest** — the learned router will capture *part* of
the oracle headroom (oracle was 0.927 @ 86.5% cost saving), not all of it. The
full run (below) quantifies how much.

_(Full 4-model sampling + trained-router Pareto results are appended after the
gated sampling run.)_

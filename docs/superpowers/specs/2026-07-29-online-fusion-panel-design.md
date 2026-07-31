# M8 — Online Fusion Panel

**Status:** design approved 2026-07-29
**Milestone:** M8 (gateway feature; makes the gateway actually fuse)

## Why

The gateway went live serving **one model per request** with a failover chain.
The fusion machinery built in M5 lives in `evaluator/` and was never wired in —
`gateway/` imports neither `evaluator` nor `router`. The user's question, in
full: *"我们这个不是 fusion 吗 怎么变成调用单一模型了"*. It is a fair
question, and this milestone answers it.

## What we know, and what we do not

M5 measured this exact panel on 1063 benchmark tasks under official scoring:

| strategy | accuracy |
|---|---:|
| **fusion (panel → cross-review → glm-5.2 fuser)** | **0.8901** |
| kimi-k3 *(best single member)* | 0.8789 |
| deepseek-chat | 0.8516 |
| glm-5.2 | 0.8329 |
| *panel oracle (any member correct)* | *0.9308* |

McNemar **p = 0.176** — the +1.1 point win is **not statistically significant**.
fix = 38, break = 26, net = +12 over 1055 tasks.

Three things follow, and the spec is honest about all of them:

1. **The gain is real but unproven.** Anyone reading "fusion beats the best
   model" out of these numbers is over-reading them. The user has seen these
   figures and chose to ship it anyway; that is a decision, not an oversight.
2. **It was measured on benchmark tasks, not chat.** Candidates were generated
   from official 0-shot prompts with grader-specific format instructions. Online
   traffic is arbitrary multi-turn conversation. **Whether the fusion gain
   transfers is unknown, and there is no online grader to measure it.**
3. **`break` is the failure mode.** Fusion talks itself out of a correct answer
   in ~2.5% of tasks. The majority-copy rule in the fusion prompt is the
   countermeasure, and it is carried over verbatim.

## Positioning (locked with the user)

- Panel is **deepseek-chat + glm-5.2 + kimi-k3**, fuser is **glm-5.2** — the
  configuration M5 measured, unchanged. (kimi-k3's quota is exhausted; the user
  is topping it up. The design degrades safely until then.)
- **Fusion is opt-in, standalone, and exposed on the public base URL.**
  `policy.default_model` stays `deepseek-chat`; a request that names no model
  (or `"auto"`/`""`) takes the ordinary single-model path, unchanged. A client
  reaches fusion only by naming it explicitly (`model: "fusion"`), and
  `/v1/models` lists it alongside the real models so it is discoverable —
  that listing is the only thing that makes "select it by name" actually
  reachable for a client that enumerates models first. This was reversed from
  an earlier decision to make fusion the default; existing default traffic is
  untouched by this feature.
- Cost and compute are **not** constraints ("目标是效果好而非省钱", "不惜算力").
  Latency is treated as a correctness concern, not a cost one.
- `gateway/` **must not import `evaluator/`**. The evaluator pulls litellm,
  datasets and scikit-learn; the production venv is 36 MB precisely because
  those were never installed. The prompt logic is **ported**, not imported.

## The latency problem, and the exact fix

Naively, fusion is three serial stages, each waiting on its slowest member.
kimi-k3 is a reasoning model measured at ~34 s per call, so it is the bottleneck
in both the candidate and the review stage: **34 + 34 + 5 ≈ 73 s**, against
0.8 s for a single deepseek call. For a client that opts into fusion that is
not merely a cost problem, it is an availability problem — many clients time
out at 60 s.

The fix comes out of M5's own fusion prompt, which says:

> If a majority of the candidates give the SAME final answer and no review
> calls it wrong, COPY that answer verbatim.

With three candidates a majority is two. **So when deepseek-chat and glm-5.2
agree and neither review objects, the fuser is required to copy that answer —
kimi-k3 cannot change the outcome.** That is lossless when the two quorum
answers agree in substance. But the review verdict alone doesn't guarantee
that: deepseek-chat and glm-5.2 can give *different* answers and each judge
the other's "correct" without the text matching, and the majority-copy rule
only fires on identical text — so that case instead takes the fuser's
"resolve the specific objections raised" branch, where kimi-k3's discarded
answer genuinely could have changed the result. Cancelling kimi-k3 on the
quorum-consensus signal is therefore a deliberate quality-for-latency trade in
the differing-answer case, not provably wasted time in general. M5 measured
79% unanimity on mmlu_pro, so most requests take this path.

```
t=0     deepseek-chat ∥ glm-5.2  (quorum candidates)
        kimi-k3 candidate launched CONCURRENTLY, not later
t≈5s    deepseek-chat ⇄ glm-5.2 mutual review
t≈10s   both verdicts "correct"  → fuse the 2, cancel kimi   → return ≈15 s
        otherwise                → kimi already 10 s in flight,
                                   wait it out, review it, fuse all 3 → ≈44 s
```

Two independent savings: kimi runs from t=0 rather than after the decision, and
**kimi is not a reviewer** (it remains a candidate — its answer still reaches
the fuser). Expected latency ≈ 0.79·15 + 0.21·44 ≈ **21 s**, against 73 s.

**The mutual review doubles as the agreement detector.** Open-ended chat has no
answer extractor, so string comparison is not available; "deepseek says glm is
correct AND glm says deepseek is correct" is the agreement signal, and it is
already being computed.

## Architecture

```
gateway/fusion_prompts.py   pure, no IO — prompt building + verdict parsing
gateway/fusion.py           the orchestrator (async, uses existing adapters)
gateway/config.py           + [fusion] section
gateway/app.py              + a branch to the fusion path
```

`gateway/providers.py`, `gateway/providers_anthropic.py`, `gateway/ledger.py`
and `gateway/db.py` are **unchanged**.

### Ported from `evaluator/fusion/`, with the benchmark scaffolding removed

`prompts.py` carries machinery that exists only to satisfy official graders:
`_FORMAT` (per-benchmark answer-format sentences), `_MCQ_SOURCES`, and
`_MCQ_NO_EARLY_ANSWER_IS_RULE` (which forbids the phrase "answer is" outside the
final line because the MCQ extractor takes the first occurrence). **None of that
applies to a chat gateway** — there is no extractor and no benchmark source.
All of it is dropped.

**Kept verbatim,** because it is the part that measurably worked:

- the majority-copy rule and "only depart when a review identifies a concrete
  error" — the `break` countermeasure;
- the `VERDICT <target> <correct|wrong|unsure> <reason>` line format, which took
  reviewer agreement from 0.63–0.74 to **0.9157**;
- no self-review: a reviewer never sees its own candidate.

### Conversation rendering

M5's prompts took `task.problem`, a single string. A chat request is
`messages[]`. Candidate calls receive the client's `messages` **verbatim**, so
multi-turn and system prompts work unchanged. **Tool definitions do NOT** —
final whole-branch review, finding 1 (CRITICAL) found that a request carrying
`tools` or `tool_choice` never reaches a fuser at all: the standard OpenAI
function-call shape (`message.content: null`, `tool_calls: [...]`) makes
`_extract_text` return `""` for every candidate, so the panel ends up with
zero usable candidates while still being billed in full. `app.py` therefore
checks for `tools`/`tool_choice` **before** deciding to fuse and routes such a
request to the single-model chain instead, starting from the panel's
preferred member — fusing a tool call is not meaningful anyway, since the
fuser writes prose and there is no sound way to merge divergent tool
invocations. The review and fusion prompts need the problem as text, so
`render_conversation(messages)` renders the conversation into a transcript
block. The final user turn is what the candidates are answering; earlier
turns are context.

### Fuser parameters

Candidates get the client's body verbatim, so `response_format`, `stop`,
`n` and every other OpenAI field already reach them unchanged. The fuser
does not: `fuser_body` builds a fresh single-message body and passes through
only an explicit allowlist (`_FUSER_PASSTHROUGH`). Final whole-branch review,
finding 6 found `response_format` and `stop` were missing from it — a client
in JSON mode silently got prose back on the fusion path, and a client-supplied
`stop` sequence was ignored by the answer the client actually received. Both are now passed through: the fuser is an ordinary completions
call to the same kind of upstream the candidates use, so "the fuser writes
the final answer" should mean the final answer honours the same
`response_format`/`stop` contract the candidates' answers did. `n` is
deliberately **not** passed through — the fuser must return exactly one
answer, and `_extract_text` only ever reads `choices[0]`, so forwarding `n`
would pay for extra choices that are silently discarded. `max_tokens`,
`temperature` and `top_p` were already passed through and are unchanged.

## Config

```toml
[fusion]
model = "fusion"                              # the pseudo-model clients request
panel = ["deepseek-chat", "glm-5.2", "kimi-k3"]
quorum = ["deepseek-chat", "glm-5.2"]         # agreement here short-circuits
reviewers = ["deepseek-chat", "glm-5.2"]      # kimi excluded: latency only
fuser = "glm-5.2"
review_max_tokens = 512                       # verdict lines are short
stage_timeout_s = 120
```

`load_config` validates at load time, raising `ConfigError` for: a `panel`,
`quorum`, `reviewers` or `fuser` entry not in `[models]`; a `quorum` that is not
a subset of `panel`; a panel smaller than 2; a duplicate entry in `panel`
(final whole-branch review, finding 8 — a duplicate collapses to one task in
`gather_panel`'s task dict, permanently pinning `degraded` true for that
panel); or a `model` name that collides with a real `[models]` key. `[fusion]`
may be omitted entirely — then no fusion pseudo-model is served and
`default_model` must name a real model.

## Request flow

`plan_route` is unchanged. `app.py` resolves the requested model first: if it
equals `cfg.fusion.model` (including via `"auto"`/`""` → `default_model`, were
`default_model` ever set to the fusion pseudo-model), the fusion path runs;
otherwise the existing single-model chain runs untouched. In the shipped
config `default_model` is `deepseek-chat`, so `"auto"`/`""` take the
single-model path — a client only reaches fusion by naming `"fusion"`
explicitly.

**Non-streaming** returns a normal OpenAI response with `model: "fusion"`, plus
a non-standard `"fusion"` object (`path`, `panel`, `fuser`, `degraded`) that
conformant clients ignore and dogfooding can read. `x-fusion-trace-id` is
returned on this path too, not only on streams.

**Streaming** streams the **fuser's** output. Stages 1–2 produce nothing
visible, so the client would see a silent 10–40 s. During that window the
gateway emits SSE comment lines (`: fusion <stage>\n\n`) — the SSE spec's
keepalive, skipped by conformant parsers including the OpenAI SDKs — so idle
timeouts do not fire. If the fuser fails after the first byte the stream ends
as it does today; before the first byte, the degradation ladder applies.

## Degradation ladder — fusion must never 500

Each rung is an event with a reason, and the response says `degraded: true`.

| condition | behaviour |
|---|---|
| the request carries `tools` or `tool_choice` | never fuse — route to the single-model chain from the panel's preferred member (finding 1a) |
| a panel member fails or times out | continue with the rest |
| **fewer than 2 candidates** succeed | return the single survivor verbatim, no fusion |
| **zero usable candidates** succeed (non-streaming) | fall through to the single-model chain, starting from the panel's preferred member (including *its own* fallback chain); `502 upstream_exhausted` only if that also exhausts (finding 1b — was an unconditional 502 before this fix) |
| all reviews fail | fuse anyway — the prompt already renders `(no reviews available)` |
| the fuser fails | return the best candidate verbatim: the quorum-agreed answer if there was one, else the first surviving member in `panel` order |
| `BudgetTripped` at any preflight | abort remaining stages, `503 budget_exhausted` |
| a stage exceeds `stage_timeout_s` | treated as failure of the outstanding calls |

"No usable text" includes an upstream error, a 2xx with `content: null` (a
refusal, or a model that puts text only in `reasoning_content`), and —
before finding 1a — the standard tool-call shape. The single-model fallback
for the zero-candidates rung is currently wired for the **non-streaming**
path only; the streaming path still calls the fuser regardless of candidate
count. That streaming gap is a known, separate limitation left for a
follow-up — it never 5xxs (the existing `_fuser_gave_nothing` fallback still
applies if the fuser itself then fails), it is just not as resilient as the
non-streaming path against a panel that produced nothing usable.

The panel deliberately keeps the M5 rule that **two candidates are enough** to
fuse. With kimi-k3's quota exhausted the panel is effectively two today, and
the gateway must serve traffic regardless.

## Billing and tracing

**The ledger needs no change.** It already writes one row per upstream call
keyed by `request_id`. Final whole-branch review, finding 5: the row count is
**never 5 or 7** — with the configured 3-model panel it is **6 rows** (3
candidates + 2 reviews + 1 fuser) when the quorum reaches consensus, or when
the full path's slow leg adds no new candidate to review (kimi-k3 403ing is
exactly this case in production today — see finding 3 below), and **8 rows**
(3 candidates + 2 reviews × 2 rounds + 1 fuser) only when the full path's slow
leg succeeds and a genuine second review round runs. `consumed_usd` sums
whichever rows exist correctly. Every call goes through the existing
`preflight` → `settle`/`fail` path with the model's own prices.

**Cancelled slow-leg calls are settled, not failed.** When the quorum
short-circuits, kimi-k3's in-flight request is aborted — but the upstream has
already done work and may bill for it. Recording `failed` would post $0 and
under-count real spend, so the row is settled with `usage_source="estimated"`
using the preflighted estimate. This over-counts when the upstream bills less;
that is the safe direction for a budget, and it is the honest one for a ledger.
The row must never be left in `preflight` — that is a consuming state recovered
only at startup.

New events, all under the request's id: `fusion.started` (panel, quorum),
`fusion.candidate` (model, ok/failed), `fusion.consensus` (agreed, verdicts),
`fusion.review` (reviewer, parsed verdict count), `fusion.fused` (fuser, path),
`fusion.degraded` (rung, reason).

## Testing

- **Pure** (`fusion_prompts.py`, no IO): conversation rendering incl. multi-turn
  and system messages; `VERDICT` parsing incl. malformed lines dropped and
  unknown targets ignored; the majority-copy rule present in the fusion prompt;
  a reviewer never sees its own candidate; and a **guard test asserting the
  benchmark scaffolding is gone** — the strings `answer is (X)`, `\boxed{}` and
  `mmlu_pro` must not appear in any generated prompt.
- **Orchestrator** with fake adapters: quorum agreement short-circuits and the
  slow leg is cancelled; disagreement takes the full path; a "wrong" verdict
  forces the full path even when the two answers match; every rung of the
  degradation ladder; `BudgetTripped` mid-fusion; streaming emits keepalives
  then the fuser's stream.
- **Billing**: a fused request writes one ledger row per call under one
  `request_id`; a cancelled slow leg is `settled`/`estimated`, never
  `preflight`; the sum matches `/admin/status`.
- **Isolation**: a guard test asserting no module under `gateway/` imports
  `evaluator` or `router`.
- **Regression**: the existing 334 tests still pass, and a request naming a real
  model still takes the single-model path byte-identically.
- **Live smoke (gated, real keys):** one fused non-streaming and one fused
  streaming request; assert 6 or 8 ledger rows under one request id (never 5
  or 7 — see Billing above), a non-zero delta, and a sane answer. Records the
  measured latency of both paths.

## Acceptance criteria

1. `model: "fusion"` returns a fused answer; naming a real model, or naming no
   model at all (`"auto"`/`""`, which resolves to `default_model`), still
   takes the single-model path with no behaviour change — fusion is reached
   only by naming it explicitly.
2. The quorum short-circuit fires when the two fast members agree, cancels the
   slow leg, and its output equals what full fusion would have produced under
   the majority-copy rule.
3. Every upstream call in a fused request appears in the ledger under one
   `request_id`, and no row is ever left in `preflight`.
4. Every rung of the degradation ladder is exercised by a test, and fusion never
   returns a 5xx the gateway itself produced.
5. `gateway/` imports nothing from `evaluator/` or `router/`, enforced by a test.
6. Streaming fusion is consumed unmodified by an OpenAI SDK.

## Non-goals

- No learned router, no verify-cascade — those are separate milestones, and the
  cascade is the higher-value one on measured evidence.
- No self-consistency resampling (M6 measured it as no better than this).
- No online quality measurement: there is no grader for chat traffic, so the
  fusion gain cannot be verified in production. Claims about quality rest on
  M5's benchmark numbers and their stated limits.
- No caching of candidates across requests.
- No change to auth, the public API surface, the ledger schema, or the event
  schema beyond additive event kinds.

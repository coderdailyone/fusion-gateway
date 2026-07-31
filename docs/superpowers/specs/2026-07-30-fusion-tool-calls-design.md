# M9 — Tool Calls Through the Fusion Panel

**Status:** design approved 2026-07-30
**Milestone:** M9 (gateway feature; makes `model: "fusion"` usable from an agent)

## Why

M8 shipped fusion as an opt-in model, and its final review found a CRITICAL
defect: a request carrying `tools` was billed for a full panel and then handed
back a 502. The fix was to **bypass fusion entirely** for such requests, routing
them to `panel[0]` (deepseek-chat) with a `fusion.bypassed` event. The reasoning
recorded at the time: *"fusing a tool call is not meaningful — the fuser writes
prose, and there is no sound way to merge divergent tool invocations."*

That made `model: "fusion"` useless from the one client the owner actually uses.
**Pi** is a coding agent whose seven built-in tools (`read`, `ls`, `grep`,
`find`, `edit`, `write`, `bash`) appear on nearly every turn, so selecting
`fusion` in Pi was equivalent to selecting `deepseek-chat` with an extra hop.

## The observation that changes the conclusion

M5 and M6 both measured a **+0.7 pt ceiling** over the best pool member, and the
M6 report named the cause: on prose there is no answer extractor, so agreement
can only be *judged* by an LLM, and judgement is noisy. The same report noted
the one place aggregation genuinely worked — the deterministic code cascade at
**0.994** — and why: *objective verification existed there.*

**A tool call carries its own extractor.** `name` plus JSON `arguments` can be
compared structurally, exactly, for free. So the half of the problem that
defeated M5 and M6 — *deciding whether the models agree* — is decidable here
without an LLM in the loop.

The final review's conclusion was half right: merging *divergent* tool calls is
genuinely hard. But detecting *agreement* is trivial, and in an agent loop the
overwhelming majority of steps are mechanical (`read` this file, `grep` that
symbol) where three models produce byte-identical calls. That is where the cost
was going and where none of it was needed.

## What we know, and what we do not

**We have no evidence that fusion improves tool calls.** M5's 0.8901 and M6's
voting curves were measured on prose answers to benchmark questions. Nothing in
this project has ever measured whether a panel picks better tool calls than one
model, and — as with M8 — there is no grader on production traffic, so it cannot
be measured online either.

What this milestone buys is structural, not measured: a second and third opinion
on each action, and an objective agreement test that costs nothing. Whether that
converts into better agent behaviour is an open question. The design is built so
that the common case is cheap enough that being wrong about the benefit costs
little.

**This makes tool-call requests more expensive than today, not less.** Today's
bypass is one upstream call. The fast path here is two. Read-only steps go from
~0.8 s to ~2 s and from 1 call to 2. That is the price of the second opinion.

## Positioning (locked with the owner)

- **Escalate only on disagreement.** Full fusion on every agent step was
  explicitly rejected: a 50-step task at ~15 s/step is over 12 minutes, and loop
  latency compounds in a way single-question latency does not.
- **Disagreement is arbitrated by a third vote first, the fuser second.** Two of
  three structurally-identical calls wins outright — this is M6's objective
  plurality, finally applicable because the ballots are comparable. The LLM
  fuser is only the backstop for a three-way split.
- **Write-class tools keep the review even when the models agree.** Structural
  agreement is a stronger *agreement* signal than mutual `correct` verdicts, but
  it carries no independent *correctness* check — two models making the same
  mistake sail through. For irreversible actions the owner chose to keep the
  check.

  **Correction (2026-07-30), after the final review:** that check is **not
  independent**, and cannot be in this topology. `agree_review` is entered only
  when every candidate's calls are byte-identical, and the reviewers are drawn
  from the candidates — so each reviewer is handed its own output under another
  model's name. The no-self-review rule holds by name and is defeated by
  content. A single surviving verdict also suffices, so one model's endorsement
  of its own call can release a write. What the branch actually buys is
  **self-critique** (evaluating is easier than generating, so some errors are
  caught) rather than an independent review, and M5's 0.9157 reviewer-agreement
  figure does not transfer — it was measured on *differing* candidates. The only
  non-proposer in the deployed panel is kimi-k3, excluded from `reviewers` for
  its ~34 s latency. The owner reviewed this and chose to keep the current
  behaviour for now; this paragraph exists so the claim above is not read as
  stronger than it is.
- **Classification defaults to reviewing.** `readonly_tools` is a positive list
  of known-safe names; anything absent is treated as write-class. A new tool, or
  a different client's tool, is reviewed rather than silently waved through.
- The single-model path and the prose fusion path keep their current behaviour.

## Flow

```
t=0    deepseek-chat ∥ glm-5.2 candidates   (the client's `tools` forwarded verbatim)
       kimi-k3 candidate launched CONCURRENTLY, as today

t≈2s   compare the two quorum candidates structurally
       ├─ tool calls identical
       │   ├─ every call in readonly_tools → EMIT, cancel kimi     2 calls, ~2 s
       │   └─ any call outside it          → cross-review          4 calls, ~7 s
       │        ├─ no reviewer says wrong  → EMIT that call
       │        └─ a reviewer says wrong   → treat as disagreement
       ├─ both are plain text             → the existing prose pipeline, unchanged
       └─ tool calls differ, or one is text and one is a call → disagreement
              ↓
          await kimi (in flight since t=0)
          ├─ 2 of 3 identical → EMIT that call        objective plurality, no LLM
          └─ all 3 differ     → cross-review + fuser, with `tools` forwarded
```

Two rules are carried over verbatim from the prose path rather than reinvented:
a reviewer objection cancels a copy (the prose rule is "majority agree **and** no
review calls it wrong"), and a **missing** review is not agreement — if the
review stage produces no verdicts for a write-class call, that escalates too.

### Correction (2026-07-30): the tree above composes into a hole

As first written, the tree said "a reviewer says wrong → treat as disagreement"
and, two lines later, "2 of 3 identical → EMIT that call". Task 5's review proved
those two rules compose into a defect: entering the objection case *requires* the
two quorum members to agree, so the plurality vote re-elects exactly the call the
reviewers rejected. Measured across all four slow-leg outcomes, the objected-to
write call was served with `reviews={}`, no fuser, and one wasted billed call.
The copy was not cancelled — it was relabelled. The same hole swallowed the
degradation row for a wholly-failed review stage.

The tree is therefore governed by one invariant, which overrides any reading of
the steps above:

> **No write-class call is ever emitted without either a clean cross-review or
> the fuser's decision.**

Two consequences. The `all_readonly` gate applies to **any** call about to be
emitted, not only to a quorum agreement — so a read-only plurality may emit
directly but a write-class plurality may not. And an objection, or a wholly
failed review stage, on a write-class agreement goes to the **fuser**, carrying
whatever reviews exist so the fuser can act on the objection; it must not be
routed through the plurality vote, which cannot do anything but re-elect the
rejected call.

### Why agreement skips the review at all

On the prose path the mutual review *is* the agreement signal; there is nothing
else. Here there is something better, and paying 2 calls plus ~5 s to add a
correctness check that M5's own numbers say cannot catch the dominant failure
(79% unanimity on mmlu_pro; when the models are wrong they are wrong together)
is not a good trade — for reversible actions. For irreversible ones the owner
overrode that, which is the `readonly_tools` split.

## Comparison rules

A call is canonicalised to `(name, json.dumps(json.loads(arguments),
sort_keys=True, separators=(",", ":")))`. Consequences, all deliberate:

- Key order and whitespace differences vanish. `{"path":"a.py"}` equals
  `{"path": "a.py"}`.
- **Equality is exact, not semantic.** `"a.py"` and `"./a.py"` are a
  disagreement. Semantic equivalence would need an LLM, which is the cost this
  design exists to avoid, and being strict errs toward more scrutiny.
- **Unparseable `arguments` make a call unusable**, which counts as
  disagreement — never as a match.
- A response may carry **several** calls (parallel tool calls). Candidates are
  compared as a **sorted multiset** of canonical calls, so ordering differences
  do not count as disagreement but a duplicated call does. When emitting, the
  winning candidate's own order is preserved.
- A candidate with tool calls and a candidate with only text never match.
- **Only the calls are compared; accompanying text is ignored.** A model may
  return both a call and prose explaining it. Two candidates proposing the same
  action with different explanations agree, because the action is what gets
  executed. The winning candidate's own text is what ships with it.

## Declared tools (security fix, 2026-07-30)

The gateway does not execute the calls it emits — the client's agent loop
does. Structural agreement and the fuser's judgement are both signals about
whether the *panel* trusts a call; neither says anything about whether the
*client* asked for it. A review of this milestone through the real app found
two ways that gap was exploitable: a fast-path request that declared only
`bash` in `tools` got back `read {"path": "/etc/shadow"}`, emitted with
`path: "tool_fast"` and `degraded: false`, because `read` happens to sit in
the server's `readonly_tools` list — classification never looked at what the
client actually declared. And on a genuine three-way split, a fuser given
`tools` forwarded and free rein over the answer proposed
`exfiltrate {"url": "...", "data": "/etc/shadow"}`, a tool nobody declared at
all, emitted with `answered_by: "fuser"` and `degraded: false`. The
pre-existing `all_readonly` gate governs read-only-vs-write-class review, a
question about the *server's* policy; it says nothing about whether the
client asked for the tool at all, and it never applied to the fuser's own
output regardless.

**The rule:** a call may only be emitted if it names a tool the client
declared, read from `body["tools"][*]["function"]["name"]` and the
deprecated `body["functions"][*]["name"]`, unioned. This applies at every
point a call is about to be served: the structural fast paths
(`tool_fast`/`tool_reviewed`/`tool_plurality`), a plurality winner, the
`best_candidate` fallback used when the fuser fails or only one candidate
survives, and the fuser's own output.

**Exemption:** if the client declared no tools at all in this request
(`tools` and `functions` both absent or empty), the check does not apply. A
provider with server-side tools may legitimately return a call the client
never listed here, and blocking that would be a regression — the rule is
"did you ask for *this* tool", not "did you ask for tools at all".

**Treatment:** an undeclared call is unusable, the same way unparseable
`arguments` are unusable (`tool_vote`'s own "None never matches anything") —
it escalates rather than being emitted, reusing the existing degradation
ladder rather than a parallel path. The fuser's own undeclared call is
treated exactly like the pre-existing "fuser returned no tool calls" rung: a
fuser failure that falls back to `best_candidate`, which itself now only
serves a candidate whose calls are all declared. Nothing is synthesised or
rewritten — an undeclared call is discarded whole, never stripped down to
its declared calls, per this spec's own non-goal against hand-editing a call
no model proposed. A distinct `fusion.degraded {"rung":
"undeclared_tool_call"}` event fires wherever this changes what gets served,
separate from the generic degradation rungs, so an operator can tell it
apart from an ordinary disagreement.

## Config

```toml
[fusion]
# ... existing keys unchanged ...
# Tools whose calls may be emitted on structural agreement alone, with no
# cross-review. Everything NOT listed is treated as write-class and keeps the
# review even when the models agree -- so a new or unknown tool is reviewed
# rather than waved through. Defaults to Pi's four read-only tools.
readonly_tools = ["read", "ls", "grep", "find"]
```

`load_config` accepts an omitted `readonly_tools` (defaulting to the four
above), requires a list of non-empty strings, and rejects duplicates with
`ConfigError`, matching the validation style already in that file. Tool names
are compared case-sensitively and exactly — no pattern matching, because a
prefix rule like `read*` would silently admit a future `readwrite` tool.

## Data model change

`PanelResult.candidates` is `dict[str, str]` today — text only, which is the
root of the original CRITICAL. It becomes `dict[str, Candidate]`:

```python
@dataclass(frozen=True)
class Candidate:
    text: str
    tool_calls: tuple[dict, ...] = ()      # raw OpenAI tool_call dicts
```

This ripples through `is_consensus`, `build_review_prompt`,
`build_fusion_prompt`, `best_candidate` and `openai_response`, all of which
currently assume a string. The prose path's behaviour must be unchanged: a
`Candidate` with empty `tool_calls` behaves exactly as the string did.

## Architecture

```
gateway/tool_vote.py        NEW — pure: canonicalise, compare, plurality, classify
gateway/fusion.py           Candidate type; tool-aware collect/consensus/fuser body
gateway/fusion_prompts.py   render tool calls in review and fusion prompts
gateway/config.py           + readonly_tools
gateway/app.py              remove the wants_tools bypass; emit tool-call responses
```

`gateway/tool_vote.py` is pure with no IO, the same shape as
`fusion_prompts.py`, so every comparison rule above is unit-testable without a
network. `gateway/providers.py`, `providers_anthropic.py`, `anthropic_translate.py`,
`ledger.py` and `db.py` are **unchanged** — glm-5.2's Anthropic-wire tool
translation already works and is reused as-is.

## Prompts

Review and fusion prompts must render a candidate that is a tool call. A call is
shown as its name and canonical arguments, so a reviewer judges the action
rather than prose. The `VERDICT <target> <correct|wrong|unsure> <reason>` format
is unchanged — "is this the right tool with the right arguments" fits it
directly.

The fusion prompt gains a rule for the three-way-split case: choose one of the
proposed calls, or propose a corrected one; do not answer in prose when the
conversation calls for an action. The fuser call therefore carries the client's
`tools` and `tool_choice`, which `fuser_body` currently strips.

## Response shape

`openai_response` must emit either shape. For a tool call: `message.content` is
`null` (or the candidate's text when it has both), `message.tool_calls` carries
the winning calls, and `finish_reason` is `"tool_calls"` rather than `"stop"`.

**Streaming** synthesises a chunk stream from the winning candidate, as
`_as_chunks` already does for text — the candidates are non-streaming calls, so
there is nothing to relay. A tool-call stream is a role chunk, one chunk
carrying the complete `tool_calls` array (no need to fragment `arguments`, since
the whole call is already in hand), a `finish_reason: "tool_calls"` chunk, then
`data: [DONE]`. The keepalive comments stay as they are.

## Degradation ladder

Every rung keeps M8's guarantee that fusion never returns a gateway-produced
5xx, and extends it to tool calls.

| condition | behaviour |
|---|---|
| a panel member fails | continue with the rest |
| fewer than 2 candidates | return the survivor verbatim — **including its tool calls** |
| zero candidates | `502 upstream_exhausted`, as today |
| the review stage fails entirely on a write-class agreement | escalate (a missing review is not agreement) |
| the fuser fails | return the best candidate, tool calls included |
| `BudgetTripped` | `503 budget_exhausted` |
| a stage timeout | treated as failure of the outstanding calls |

## Billing

Unchanged: one ledger row per upstream call, all sharing one `request_id`, never
left in `preflight`, cancelled legs settled with `usage_source="estimated"`.

Row counts follow the branch taken and are **not** a fixed set, because a
write-class agreement that a reviewer objects to escalates and keeps the review
rows it already paid for. The floor is **2** (read-only agreement: two
candidates, no review, no fuser) and the ceiling is the existing full path.
Tests assert the count for the specific branch they exercise rather than a
global range — the M8 spec previously claimed "5–7" and was wrong, and its
"6 or 8" correction describes the prose path only.

### Correction (2026-07-30), second: the floor is 3 on a 3-member panel, not 2

The "floor is 2" claim above, and acceptance criterion 2 below as first
written, both describe a 2-member panel (quorum only, no slow leg). The
milestone's own panel is 3 members (2 quorum + 1 slow leg), and on THAT shape
the read-only fast path is **3** upstream calls and **3** ledger rows, not 2:
the two quorum candidates settle normally, and the slow leg — launched at t=0
alongside them, per the Flow section above — is cancelled once the quorum
agrees, but a cancelled call still bills. The money invariant stated in
Billing above is exactly why: a cancelled leg is never left in `preflight`
and never `fail`ed (that would post \$0 for work the upstream may already be
billing) — it is settled with `usage_source="estimated"`. That settlement
*is* the third row. The plan and the tests were corrected to this before this
spec was: `test_identical_readonly_calls_emit_with_no_review_and_no_slow_leg`
and the app-level `test_a_tool_calling_request_now_reaches_the_fusion_panel`
both assert `len(rows) == 3` on the read-only fast path, with an explicit
comment tracing the discrepancy back to an earlier draft of this plan that
said 2.

This is not a rounding error in the cost story. Measured on the read-only fast
path with a live (not 403ing) slow leg:

```
a  settled  reported   $0.000007
b  settled  reported   $0.000007
s  settled  ESTIMATED  $0.001025   <- cancelled leg, preflight estimate
rows=3  preflight_left=0  upstream dispatched=3
```

The cancelled leg alone is **98.7%** of that request's total cost — the two
settled candidates are three orders of magnitude cheaper. And that measured
figure is not the worst case: `estimate_tokens` defaults `est_out` to 1024
when the client omits `max_tokens` (the shape of a typical agent tool call,
and unlike the request above), and the cancelled leg's ledger row is always
an *estimate*, never a measurement. At kimi-k3's \$2.50/mtok out, a cancelled
leg priced off that default alone is ≈1024 × \$2.50 / 1e6 ≈ **\$0.00256** —
about **4.4×** the entire three-request M9 live smoke's \$0.000579 total (see
`docs/M1_ACCEPTANCE.md`). Both numbers are invisible in that recorded smoke
only because kimi-k3 was 403ing there: a provider error bills as `fail`
(\$0), not a cancellation, so the smoke never exercised this leg at all. A
client that reliably supplies a tight `max_tokens` lowers this; one that
doesn't should expect the cancelled leg to dominate the recorded cost of the
cheapest, most common path this milestone has.

## Testing

- **Pure** (`tool_vote.py`): canonicalisation across key order and whitespace;
  exact-not-semantic (`"a.py"` vs `"./a.py"` differ); unparseable arguments count
  as disagreement; multiset comparison ignores order but not duplication; a
  tool-call candidate never matches a text candidate; plurality returns the
  2-of-3 winner and `None` on a three-way split; the read-only classifier is
  exact and default-deny, including a call to an unlisted tool and a mixed batch
  where one call is read-only and one is not.
- **Prompts**: a tool call renders in both the review and fusion prompts; the
  fusion prompt's action rule is present; the prose path's prompts are
  byte-identical to today for text-only candidates.
- **Orchestrator**: the read-only fast path cancels the slow leg but still
  bills it (3 ledger rows on the milestone's 3-member panel — see the second
  Billing correction above); a write-class agreement runs the review and
  emits on no objection; a reviewer objection escalates; a 2-of-3 plurality
  emits without the fuser; a three-way split reaches the fuser with `tools`
  forwarded; every degradation rung; no row ever left in `preflight`.
- **App**: a `tools` request no longer bypasses fusion and no longer 502s; the
  response carries `tool_calls` with `finish_reason: "tool_calls"`; the
  synthesised stream is consumed by the real `openai` SDK's SSE decoder; a
  request naming a real model still takes the single-model path unchanged.
- **Regression**: the existing 422 tests still pass, and the prose fusion path
  behaves identically.
- **Live smoke (paid):** one read-only tool call and one write-class tool call
  through `model: "fusion"`, recording the measured latency and ledger rows of
  each path. kimi-k3's quota is still exhausted, so the plurality branch cannot
  be exercised live until it is topped up — record that gap rather than claiming
  coverage.

## Acceptance criteria

1. A request carrying `tools` reaches the fusion panel instead of bypassing it,
   and returns a valid OpenAI tool call.
2. Two identical read-only calls are emitted with no review and no fuser; on
   the milestone's 3-member panel (2 quorum + 1 slow leg) this is exactly 3
   upstream calls and 3 ledger rows — the two settled quorum candidates plus
   the cancelled slow leg, which must still settle at its preflight estimate
   rather than vanish (see the billing correction above).
3. Two identical write-class calls are reviewed; an objection escalates.
4. Differing calls are arbitrated by the third member's vote before the fuser is
   consulted; a three-way split reaches the fuser.
5. Comparison is exact after JSON canonicalisation, order-insensitive across
   parallel calls, and treats unparseable arguments as disagreement.
6. An unlisted tool is treated as write-class.
7. Every upstream call appears in the ledger under one `request_id`; none is
   left in `preflight`.
8. Fusion never returns a 5xx the gateway itself produced.
9. Streaming tool calls are consumed unmodified by an OpenAI SDK.

## Non-goals

- **No semantic argument equivalence.** Exact after canonicalisation only.
- **No hand-merging of divergent calls.** The plurality vote or the fuser picks
  one; the gateway never synthesises a call no model proposed.
- No per-tool policy beyond the read-only / write-class split — no per-tool
  reviewer counts, no risk scores.
- No streaming of candidate calls; candidates stay non-streaming and the client
  stream is synthesised.
- No change to auth, the ledger schema, the single-model path, or the prose
  fusion path's behaviour.
- No attempt to measure whether this improves agent outcomes. There is no
  grader for production traffic, and this spec does not pretend otherwise.

"""Online fusion panel: candidates -> cross-review -> fuse.

Every panel member's candidate call is launched at t=0. Only the QUORUM is
awaited. If the quorum members review each other as correct, that is
consensus, and the slow leg is cancelled. This is lossless when the two
quorum answers also agree in substance: M5's fusion prompt requires the fuser
to COPY that shared answer verbatim, so the slow leg provably cannot change
it. But "each other correct" does not require identical text -- two
DIFFERING answers can each be judged correct by the other, in which case the
majority-copy rule never fires and the fuser instead resolves the specific
objections raised, a branch the slow leg genuinely could have swung.
`is_consensus` reads only the review verdicts, never the candidate texts, so
it cannot tell the two cases apart; cancelling on it is a deliberate
quality-for-latency trade in the differing-answer case, not a proven no-op.
Otherwise (no consensus) the slow leg (running since t=0) is awaited and
folded in.

This module talks to upstreams only through the adapters `app.py` already
built, and bills through the existing ledger: one row per upstream call, all
sharing one request_id. It imports nothing from `evaluator/` or `router/`.

Invariant: `gather_panel` never returns or raises with a panel task still
running. `asyncio.wait_for(asyncio.shield(task))` -- used so a slow quorum
member keeps running past its own collection timeout -- only cancels the
*wait*, not the shielded task, so a naive "cancel the slow leg on the
consensus path only" leaves a hole: a quorum member that itself timed out is
skipped with `continue` but its task keeps running past the end of the
request, holding an upstream connection open. `gather_panel` closes that hole
with a `try/finally` around the whole body: on every exit path -- quorum
consensus, full path, a timed-out member, or any exception including
BudgetTripped -- every not-yet-done panel task is cancelled and awaited
before the function returns or re-raises. `call_model`'s CancelledError
handler (below) is what makes that safe: a cancelled leg still SETTLES its
ledger row with the preflight estimate rather than being left in 'preflight'
(a CONSUMING_STATE only a restart clears) or `fail`ed (which would post $0
for work the upstream may already be billing).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from gateway.fusion_prompts import (
    Verdict, build_fusion_prompt, build_review_prompt, parse_review,
    render_conversation,
)
from gateway.ledger import BudgetTripped, estimate_tokens
from gateway.providers import ProviderError
from gateway.tool_vote import all_readonly, canonical_calls, plurality

logger = logging.getLogger(__name__)

# Passed through to the fuser; everything else the client sent is dropped.
# `response_format` and `stop` are here so a client in JSON mode gets JSON
# back and a client-supplied stop sequence is honoured by the answer the
# client actually receives (final review, finding 6) -- both the candidates
# and the fuser are ordinary completions calls to the same kind of upstream,
# so passing them through is exactly what "the fuser writes the final answer"
# means. `n` is deliberately NOT here: the fuser must return exactly one
# answer, and `_extract_text` only ever reads `choices[0]`, so passing `n`
# through would ask (and pay) for extra choices that are silently discarded.
_FUSER_PASSTHROUGH = ("max_tokens", "temperature", "top_p", "response_format", "stop")


@dataclass(frozen=True)
class Candidate:
    """One panel member's answer: prose, tool calls, or both.

    Candidates used to be bare strings, which is why a tool call made
    `_extract_text` return "" and got the candidate dropped -- a fully-billed
    panel then returned 502 (M8 final review, finding 1a). A Candidate with
    empty `tool_calls` must behave exactly as the string did.
    """
    text: str
    tool_calls: tuple[dict, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text or self.tool_calls)


@dataclass(frozen=True)
class PanelResult:
    conversation: str
    candidates: dict[str, Candidate]
    reviews: dict[str, dict[str, Verdict]]
    path: str            # "quorum" | "full" | "tool_fast" | "tool_reviewed" | "tool_plurality"
    degraded: bool


# The three paths where gather_panel deliberately returns a single-candidate
# panel because it structurally DECIDED, not because it lost candidates.
# app.py's _finish_fusion uses this to skip the fuser without misreporting a
# healthy fast path as a degradation (Task 5 fix round 1, finding 4): every
# one of these paths trips `len(panel.candidates) < 2`, which is otherwise
# the signal for "we lost candidates and are falling back."
TOOL_DECIDED_PATHS = frozenset({"tool_fast", "tool_reviewed", "tool_plurality"})


def _extract_text(resp) -> str:
    """Pull the assistant text out of an upstream response, defensively.

    The response is upstream-controlled; a raw index or key access here would
    turn a malformed 200 into a gateway 500.
    """
    try:
        content = resp["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):        # content parts
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _extract_message(resp) -> Candidate:
    """Pull text AND tool calls out of an upstream response, defensively.

    The response is upstream-controlled, so every access is guarded: a raw
    index would turn a malformed 200 into a gateway 500. Kept alongside
    `_extract_text` on purpose (not a duplicate to "clean up"): the review
    and fuser calls return prose and have no use for tool calls, so only the
    candidate path needs this.
    """
    text = _extract_text(resp)
    calls: tuple[dict, ...] = ()
    try:
        raw = resp["choices"][0]["message"].get("tool_calls")
    except (TypeError, KeyError, IndexError, AttributeError):
        raw = None
    if isinstance(raw, (list, tuple)):
        calls = tuple(c for c in raw if isinstance(c, dict))
    return Candidate(text, calls)


async def call_model(*, model_name, body, cfg, adapters, ledger, events, clock,
                     request_id, kind):
    """One billed upstream call. Returns its answer, or None on failure.

    `kind` labels the event ("candidate" | "review" | "fuser"). Raises only
    BudgetTripped and asyncio.CancelledError. The candidate AND fuser paths
    can both carry tool calls -- once `fuser_body` forwards `tools`, the
    fuser is free to answer with `content: null` + `tool_calls` exactly like
    a candidate, and `_extract_text` would silently return "" for that shape
    (M8 finding 1a, reproduced in the fuser leg -- Task 5 step 5). So both
    return a `Candidate` via `_extract_message`. Only review calls stay
    text-only (`_extract_text`) -- a reviewer's job is to judge, never to act.
    """
    model_cfg = cfg.models[model_name]
    adapter = adapters[model_cfg.provider]
    est_in, est_out = estimate_tokens(body.get("messages") or [],
                                      body.get("max_tokens"))
    entry_id = ledger.preflight(
        request_id, model_cfg.provider, model_name, est_in, est_out,
        model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
    )
    events.append(request_id, "call.attempt", {"model": model_name, "stage": kind})
    start = clock.now()

    def _latency():
        return int((clock.now() - start).total_seconds() * 1000)

    try:
        resp = await adapter.chat(model_cfg.upstream_model, body)
    except asyncio.CancelledError:
        # The quorum agreed and this leg is no longer needed (or the request
        # is being torn down for some other reason). The upstream has already
        # done work and may bill for it, so settle with the preflight
        # estimate: fail() would post $0 and under-count real spend, and
        # leaving the row in 'preflight' would hold a CONSUMING_STATE that
        # only a restart clears.
        ledger.settle(entry_id, est_in, est_out, "estimated", _latency(),
                      model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
        events.append(request_id, "fusion.candidate",
                      {"model": model_name, "status": "cancelled"})
        raise
    except ProviderError as exc:
        ledger.fail(entry_id)
        events.append(request_id, "call.failed",
                      {"model": model_name, "stage": kind,
                       "kind": exc.kind, "status": exc.status})
        return None
    except Exception:
        logger.exception("fusion %s call failed model=%s request_id=%s",
                         kind, model_name, request_id)
        ledger.fail(entry_id)
        events.append(request_id, "call.failed",
                      {"model": model_name, "stage": kind, "kind": "unknown"})
        return None

    usage = (resp or {}).get("usage") or {}
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        in_tok, out_tok, source = usage["prompt_tokens"], usage["completion_tokens"], "reported"
    else:
        in_tok, out_tok, source = est_in, est_out, "estimated"
    ledger.settle(entry_id, in_tok, out_tok, source, _latency(),
                  model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
    events.append(request_id, "call.succeeded", {"model": model_name, "stage": kind})
    if kind in ("candidate", "fuser"):
        return _extract_message(resp)
    return _extract_text(resp)


def _merge_reviews(base: dict[str, dict[str, Verdict]],
                   extra: dict[str, dict[str, Verdict]]
                   ) -> dict[str, dict[str, Verdict]]:
    """Merge a newer review round (`extra`) into an older one (`base`).

    A reviewer present in `extra` supersedes its own verdicts from `base`
    (it re-judged with more, or different, evidence); a reviewer ABSENT
    from `extra` keeps its `base` verdicts rather than losing them. This is
    the exact merge rule the prose path's own "round 2" fix already
    established (a bare `reviews = round2` used to discard already-paid
    round-1 verdicts whenever round 2 failed wholly or partly) -- extracted
    here so the tool decision tree's own fall-through into the full path
    (Task 5 fix round 1, finding 1) can reuse it instead of a second,
    subtly different inline copy.
    """
    merged = {r: dict(v) for r, v in base.items()}
    for reviewer, verdicts in extra.items():
        merged.setdefault(reviewer, {}).update(verdicts)
    return merged


def is_consensus(candidates: dict[str, Candidate],
                 reviews: dict[str, dict[str, Verdict]]) -> bool:
    """True only when every candidate was reviewed `correct` by every other.

    Deliberately conservative: a missing review is NOT agreement. Absence of
    evidence sends the request down the slow path, which is the safe direction.

    Only ever reads `candidates`' keys (never a value), so it is unaffected
    by candidates carrying tool calls -- the annotation just tracks reality.
    """
    names = sorted(candidates)
    if len(names) < 2:
        return False
    for reviewer in names:
        verdicts = reviews.get(reviewer)
        if not verdicts:
            return False
        for target in names:
            if target == reviewer:
                continue
            got = verdicts.get(target)
            if got is None or got.verdict != "correct":
                return False
    return True


async def _cross_review(*, candidates, fcfg, cfg, adapters, ledger, events,
                        clock, request_id, conversation):
    """Each configured reviewer that produced a candidate judges the others.

    Bounded by one `stage_timeout_s` deadline for the WHOLE review stage
    (fix round 1, finding 2 -- reviews previously had no bound at all beyond
    httpx's 120s default, paid on top of the same unbounded candidate stage).
    A reviewer that doesn't answer within the stage's remaining budget is
    dropped, same treatment a slow candidate gets in `collect()`.

    Each reviewer runs as an explicit task, not a bare `asyncio.gather(...)`
    coroutine (fix round 1, finding 1 -- CRITICAL). `asyncio.gather` with its
    default `return_exceptions=False` propagates the first sibling exception
    immediately, without cancelling or awaiting the others. Since `ledger.
    preflight` can raise `BudgetTripped` for ANY reviewer independently, that
    left an orphaned running task and a ledger row stuck in `preflight` --
    exactly at budget exhaustion, when the accounting matters most. The
    `try/finally` below gives this stage the same "never return or raise
    with a task still running" invariant `gather_panel` already has.
    """
    reviewers = [r for r in fcfg.reviewers if r in candidates]
    if len(candidates) < 2 or not reviewers:
        return {}

    async def one(reviewer):
        targets = {m for m in candidates if m != reviewer}
        if not targets:
            return {}
        body = {"messages": [{"role": "user", "content":
                              build_review_prompt(conversation, candidates, reviewer)}],
                "max_tokens": fcfg.review_max_tokens}
        text = await call_model(model_name=reviewer, body=body, cfg=cfg,
                                adapters=adapters, ledger=ledger, events=events,
                                clock=clock, request_id=request_id, kind="review")
        parsed = parse_review(text, targets)
        events.append(request_id, "fusion.review",
                      {"reviewer": reviewer, "verdicts": len(parsed)})
        return parsed

    deadline = time.monotonic() + fcfg.stage_timeout_s
    tasks = {r: asyncio.create_task(one(r)) for r in reviewers}
    try:
        remaining = max(0.0, deadline - time.monotonic())
        # FIRST_EXCEPTION, not the ALL_COMPLETED default: the moment any
        # reviewer's call raises (BudgetTripped from ledger.preflight, most
        # importantly), stop waiting on the others immediately rather than
        # riding out their full delay before reacting -- they get cancelled
        # below instead of awaited out. When nothing raises this behaves
        # exactly like ALL_COMPLETED.
        await asyncio.wait(tasks.values(), timeout=remaining,
                          return_when=asyncio.FIRST_EXCEPTION)

        results: dict[str, dict] = {}
        pending: list[str] = []
        first_exc: BaseException | None = None
        for reviewer, task in tasks.items():
            if not task.done():
                pending.append(reviewer)
                continue
            # `Task.exception()` RAISES CancelledError for a task in the
            # cancelled state -- it does not return it. Nothing in this loop
            # cancels these tasks before we get here (that only happens in
            # the `finally` below, after this loop has already run), so this
            # guard is currently unreachable dead code on any live path --
            # but it must stay: one refactor that starts cancelling a
            # reviewer task earlier (e.g. a per-reviewer timeout) would turn
            # `task.exception()` into a crash here otherwise. Treat a
            # cancelled task the same as "not done" -- there is no verdict to
            # collect from it either way.
            if task.cancelled():
                pending.append(reviewer)
                continue
            # `.exception()` retrieves it even when we don't act on it below,
            # so asyncio never logs "Task exception was never retrieved" for
            # a sibling we're about to discard in favour of the first one.
            exc = task.exception()
            if exc is not None:
                if first_exc is None:
                    first_exc = exc
                continue
            verdicts = task.result()
            if verdicts:
                results[reviewer] = verdicts

        if first_exc is not None:
            # `pending` here means "abandoned because a sibling blew up", not
            # "timed out" -- don't mislabel it as the latter. The finally
            # below cancels and awaits every one of them before this
            # propagates, so gather_panel never sees a leftover task.
            raise first_exc

        for reviewer in pending:
            events.append(request_id, "fusion.degraded",
                          {"rung": "review_timeout", "model": reviewer})
        return results
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        for task in tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                # Swallow only a CancelledError this cleanup itself caused
                # (task.cancelled() is True). A CancelledError here whose
                # task is NOT actually cancelled means something else
                # cancelled the coroutine we are running in (this function's
                # own caller), and that must propagate, not be absorbed.
                if not task.cancelled():
                    raise
            except Exception:
                pass


def decide_tools(candidates: dict[str, Candidate], readonly: frozenset[str]):
    """Classify a set of candidates by their tool calls.

    Returns one of:
      ("agree_readonly", winner)  every candidate proposed the same calls, all
                                  read-only -- emit with no review
      ("agree_review", winner)    same calls, but at least one is write-class --
                                  the review still runs
      ("disagree", None)          the calls differ, or one is prose and one a
                                  call, or none are comparable
      ("prose", None)             no candidate proposed a call at all

    Pure and read-only: never touches the ledger or events. `winner` is the
    alphabetically-first model among the (structurally identical) agreeing
    set -- deterministic, so it does not depend on dict iteration order.

    "prose" means no candidate proposed a call AT ALL -- `tool_calls ==
    ()`. A candidate that proposed a call with unparseable arguments is
    NOT prose (`tool_calls` is non-empty; `canonical_calls` just can't
    canonicalise it) -- `tool_vote`'s own docstring is explicit that an
    unparseable call is unusable, not absent, so it must fall through to
    "disagree" like any other non-comparable pair. Using `by_model`'s
    None-ness for this (as an earlier version of this function did) would
    conflate the two: an all-unparseable panel would read as "prose" and
    skip the tool decision tree entirely instead of correctly disagreeing.
    """
    has_any_call = any(getattr(c, "tool_calls", ()) for c in candidates.values())
    if not has_any_call:
        return "prose", None
    by_model = {m: canonical_calls(c.tool_calls) for m, c in candidates.items()}
    if len(candidates) < 2:
        return "disagree", None
    values = list(by_model.values())
    if any(v is None for v in values) or len(set(values)) != 1:
        return "disagree", None
    winner = sorted(candidates)[0]
    kind = "agree_readonly" if all_readonly(values[0], readonly) else "agree_review"
    return kind, winner


async def gather_panel(*, fcfg, cfg, adapters, ledger, events, clock,
                       request_id, body) -> PanelResult:
    """Stages 1-2. Raises BudgetTripped; every other call failure degrades.

    Invariant: never returns or raises with a panel task still running (see
    module docstring) -- enforced by the try/finally below, which is
    unconditional over every exit path.
    """
    conversation = render_conversation(body.get("messages") or [])
    slow = [m for m in fcfg.panel if m not in fcfg.quorum]
    events.append(request_id, "fusion.started",
                  {"panel": list(fcfg.panel), "quorum": list(fcfg.quorum)})

    async def candidate(model_name):
        text = await call_model(model_name=model_name, body=body, cfg=cfg,
                                adapters=adapters, ledger=ledger, events=events,
                                clock=clock, request_id=request_id,
                                kind="candidate")
        events.append(request_id, "fusion.candidate",
                      {"model": model_name, "status": "ok" if text else "failed"})
        return text

    # t=0: launch EVERY panel member, quorum and slow alike. Launching the slow
    # leg later would make the full path cost quorum + slow instead of
    # max(quorum, slow) -- the entire latency argument rests on this line.
    tasks = {m: asyncio.create_task(candidate(m)) for m in fcfg.panel}

    async def collect(names):
        """Await `names` against ONE deadline for this whole stage call.

        Fix round 1, finding 3: giving each member its own fresh
        `stage_timeout_s` makes the deadline accumulate -- three hanging
        members under one `collect()` call cost ~3x the configured budget
        instead of ~1x. `deadline` is computed once, up front, and every
        member's `wait_for` gets whatever of it remains.
        """
        deadline = time.monotonic() + fcfg.stage_timeout_s
        got = {}
        for m in names:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                text = await asyncio.wait_for(asyncio.shield(tasks[m]),
                                              timeout=remaining)
            except asyncio.TimeoutError:
                # `shield` means only this wait is abandoned here -- tasks[m]
                # itself keeps running. That is fine: `cancel_remaining` in
                # the finally below sweeps up anything still not done before
                # gather_panel returns, whether it eventually finishes,
                # itself times out, or never gets awaited again.
                events.append(request_id, "fusion.degraded",
                              {"rung": "candidate_timeout", "model": m})
                continue
            if text:
                got[m] = text
        return got

    async def cancel_remaining():
        """Cancel and await every panel task still running, unconditionally.

        Runs in `finally` on every exit path (quorum consensus, full path,
        exception, BudgetTripped) so gather_panel never leaves a task behind
        for the caller to discover later -- an abandoned task would settle
        its ledger row after the response is already gone, or hold an
        upstream connection open indefinitely.
        """
        for task in tasks.values():
            if not task.done():
                task.cancel()
        for task in tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                # Fix round 1, finding 5: swallow only a CancelledError this
                # cleanup itself caused (task.cancelled() is True). If it is
                # NOT actually cancelled, this CancelledError was injected by
                # something cancelling gather_panel's own coroutine -- e.g.
                # the request handler task -- while we happened to be
                # awaiting an already-finished child task here. That must
                # propagate so gather_panel itself ends up cancelled, not
                # swallowed into a normal return.
                if not task.cancelled():
                    raise
            except Exception:
                pass

    async def cancel(names):
        """Cancel and await a SPECIFIC subset of panel tasks.

        Every current call site returns immediately afterward with no other
        `await` in between, which makes this PROVABLY REDUNDANT with
        `cancel_remaining`'s unconditional `finally`-block cleanup below --
        confirmed in Task 5's fix round 1 review by deleting these calls
        entirely (both call sites, and even turning the whole helper into a
        no-op): the full 469-test suite stayed green, and a direct
        ledger/timing check showed identical settlement ("estimated" vs
        "reported") with and without them across several slow-leg delays.
        `cancel_remaining` fires at the exact same point in the coroutine's
        execution either way, because nothing here does real async work
        between the decision and the `return`.

        Kept anyway, deliberately, for two reasons: it documents AT THE
        DECISION POINT that the slow leg is no longer needed, which a
        reader would otherwise only discover by tracing all the way to the
        bottom of `gather_panel`; and it is a hedge against a future change
        to either call site that inserts real work between the decision and
        its `return` -- which WOULD reopen a real gap this doesn't
        currently have. Reuses the exact same pattern `cancel_remaining`
        does (task.cancel(), then await, re-raising only a CancelledError
        that ISN'T this cleanup's own doing) rather than settling the
        ledger by hand, so if it ever does start mattering again, it stays
        correct for the same reason `cancel_remaining` is: `call_model`'s
        CancelledError handler is the ONLY place that settles a cancelled
        leg, at its preflight estimate, rather than leaving the row in
        'preflight' or `fail`ing it.
        """
        for m in names:
            if not tasks[m].done():
                tasks[m].cancel()
        for m in names:
            try:
                await tasks[m]
            except asyncio.CancelledError:
                if not tasks[m].cancelled():
                    raise
            except Exception:
                pass

    try:
        candidates = await collect(fcfg.quorum)
        reviews: dict[str, dict[str, Verdict]] = {}
        # Set once the tool decision tree below has already awaited the
        # slow leg, so the pre-existing full-path code at the bottom of
        # this function does not do it AGAIN (fix round 1, finding 3): the
        # tasks are already done by then, so a second `collect(slow)` just
        # re-awaits them and `new_from_slow` comes back non-empty, which
        # used to trigger a full, byte-identical SECOND review round over
        # the same three candidates -- doubling both the bill and, on a
        # hanging slow leg, the latency budget.
        slow_collected = False

        verdict, winner = decide_tools(candidates, fcfg.readonly_tools)
        if verdict != "prose":
            # Gated: this fires on every request once tool support ships,
            # including the (common) prose ones, where the verdict never
            # influences anything downstream -- no reason to double the
            # event volume for a verdict nobody will ever read back.
            events.append(request_id, "fusion.tool_verdict",
                          {"verdict": verdict, "winner": winner})

        if verdict == "agree_readonly":
            # Structural agreement is an exact agreement signal, unlike mutual
            # "correct" verdicts -- so no review is needed. It carries no
            # independent correctness check, which is why write-class calls
            # below keep the review even when the models agree.
            await cancel(slow)
            return PanelResult(conversation, {winner: candidates[winner]}, {},
                               "tool_fast",
                               degraded=len(candidates) < len(fcfg.quorum))

        if verdict == "agree_review":
            reviews = await _cross_review(
                candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
                ledger=ledger, events=events, clock=clock,
                request_id=request_id, conversation=conversation)
            objected = any(v.verdict == "wrong"
                           for verds in reviews.values() for v in verds.values())
            # A MISSING review is not agreement either -- the same conservative
            # rule the prose path's is_consensus uses.
            if reviews and not objected:
                await cancel(slow)
                return PanelResult(conversation, {winner: candidates[winner]},
                                   reviews, "tool_reviewed",
                                   degraded=len(candidates) < len(fcfg.quorum))
            # Spec correction, fix round 1 finding 1 (CRITICAL): an
            # objection, or a wholly-failed review stage (`reviews == {}`,
            # caught by the `reviews and` guard above), on a write-class
            # agreement must not silently re-serve the rejected call. `a`
            # and `b` (the quorum) already match by construction of
            # reaching `agree_review` at all, so `plurality`, run against
            # {a, b, <slow>}, ALWAYS finds {a, b} as its first >=2 group --
            # regardless of what the slow leg proposes -- and the original
            # bug emitted that pair directly with reviews={} and no fuser:
            # not a cancelled copy, a RELABELLED one, silently re-serving
            # the exact call a reviewer just rejected (or serving it on no
            # evidence at all, if review failed outright).
            #
            # This routes to its own verdict, `disagree_no_plurality`,
            # rather than falling into the `disagree` branch below and
            # relying solely on that branch's own `all_readonly` gate
            # (finding 2) to block emission -- even though, checked directly
            # (see task-5-report.md, Fix round 1): given that `agree_review`
            # is ONLY ever reached for a write-class call in the first
            # place (agree_readonly already siphons off every read-only
            # agreement), `disagree`'s plurality winner in THIS scenario is
            # PROVABLY always that same write-class {a, b} call, which
            # `all_readonly` always correctly rejects anyway -- reverting
            # this verdict assignment to plain "disagree" leaves the whole
            # 479-test suite green. Kept as its own explicit verdict anyway,
            # for the same reason `cancel`'s redundant-but-kept helper is
            # (see its docstring): it documents, AT THE DECISION POINT, that
            # this case must not be resolved by majority vote, rather than
            # depending on a reader to trace into `disagree` and rediscover
            # why its gate happens to always fire here. It is also a hedge
            # against `disagree`'s gate ever changing (e.g. if `plurality`
            # or `all_readonly`'s semantics shift) in a way that would
            # silently reopen this exact hole.
            verdict = "disagree_no_plurality"

        if verdict == "disagree":
            candidates.update(await collect(slow))
            slow_collected = True
            plur = plurality({m: canonical_calls(c.tool_calls)
                              for m, c in candidates.items()})
            if plur is not None:
                winner_canon = canonical_calls(candidates[plur].tool_calls)
                # Spec correction, fix round 1 finding 2: the all_readonly
                # gate applies to ANY call about to be emitted without a
                # review, not only a quorum agreement -- a read-only
                # plurality is exact for the same reason agree_readonly is
                # (structural agreement, no judgement involved), but a
                # write-class plurality carries no more of a correctness
                # check than a write-class STRUCTURAL agreement does, which
                # is the milestone's own stated reason agree_review keeps
                # the review in the first place.
                if all_readonly(winner_canon, fcfg.readonly_tools):
                    return PanelResult(conversation, {plur: candidates[plur]},
                                       {}, "tool_plurality",
                                       degraded=len(candidates) < len(fcfg.panel))
                # Write-class plurality: do not emit. Fall through to the
                # full path below, which cross-reviews everyone (including
                # the slow leg, already collected) and lets the fuser
                # decide instead.
            # Three-way split, or a write-class plurality that must not be
            # emitted directly: fall through to the full path.

        if verdict == "disagree_no_plurality":
            candidates.update(await collect(slow))
            slow_collected = True
            # Deliberately does NOT attempt `plurality` -- see the comment
            # in the `agree_review` branch above for why that would just
            # re-elect the rejected call.

        reviews = _merge_reviews(reviews, await _cross_review(
            candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
            ledger=ledger, events=events, clock=clock, request_id=request_id,
            conversation=conversation))
        agreed = is_consensus(candidates, reviews)
        events.append(request_id, "fusion.consensus",
                      {"agreed": agreed, "candidates": sorted(candidates)})

        if agreed:
            return PanelResult(conversation, candidates, reviews, "quorum",
                               degraded=len(candidates) < len(fcfg.quorum))

        if slow_collected:
            # Fix round 1, finding 3: already awaited (and already reviewed,
            # just above) by the tool decision tree -- collecting it again
            # here would just re-await already-done tasks and (since they
            # come back non-empty) trigger the SECOND, byte-identical review
            # round this exact guard exists to prevent.
            new_from_slow = {}
        else:
            new_from_slow = await collect(slow)
            candidates.update(new_from_slow)
        # Fix (finding 3): only re-review when the slow leg actually added a
        # candidate. Without this gate, a slow leg that failed or timed out
        # (kimi-k3 is 403ing in production right now) still triggered a
        # SECOND, byte-identical review round -- same reviewers, same
        # candidates, same prompts -- charged twice for evidence that could
        # not possibly change.
        if new_from_slow and len(candidates) >= 2:
            round2 = await _cross_review(
                candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
                ledger=ledger, events=events, clock=clock,
                request_id=request_id, conversation=conversation)
            # Fix (finding 4): merge round 2 into round 1 instead of
            # replacing it -- see `_merge_reviews`'s docstring.
            reviews = _merge_reviews(reviews, round2)
        return PanelResult(conversation, candidates, reviews, "full",
                           degraded=len(candidates) < len(fcfg.panel))
    finally:
        await cancel_remaining()


def fuser_body(fcfg, panel: PanelResult, body: dict) -> dict:
    """The OpenAI body for the fuser call: one user message.

    `tools`/`tool_choice` are forwarded only when the panel actually holds tool
    calls -- the fuser has to be able to answer with a call, and on the prose
    path forwarding them would invite a call nobody asked for.
    """
    out = {k: body[k] for k in _FUSER_PASSTHROUGH if k in body}
    # `getattr(c, "tool_calls", ())`, not `c.tool_calls`: several existing
    # tests build a PanelResult directly with bare strings for candidates
    # (predating Task 3's Candidate refactor -- see _candidate_block's
    # docstring in fusion_prompts.py for the same convention), and this must
    # tolerate that exactly like render_candidate already does, rather than
    # AttributeError on the prose path's own pre-existing tests.
    if any(getattr(c, "tool_calls", ()) for c in panel.candidates.values()):
        for k in ("tools", "tool_choice"):
            if k in body:
                out[k] = body[k]
    out["messages"] = [{"role": "user", "content": build_fusion_prompt(
        panel.conversation, panel.candidates, panel.reviews)}]
    return out


def best_candidate(fcfg, panel: PanelResult):
    """The answer to fall back on when the fuser fails: the first surviving
    member in configured panel order. Returns (model, Candidate) or None."""
    for m in fcfg.panel:
        c = panel.candidates.get(m)
        if c:
            return m, c
    for m in sorted(panel.candidates):
        c = panel.candidates[m]
        if c:
            return m, c
    return None


def openai_response(candidate: Candidate, model: str, meta: dict) -> dict:
    """The client-facing JSON response for a fused (or fallback) answer.

    `candidate` must be a `Candidate` -- both sources that reach here now
    agree on that shape: `best_candidate`'s fallback always was one, and as
    of Task 5 step 5 the fuser's own call is too (`call_model(kind="fuser")`
    returns `_extract_message`, not `_extract_text`, precisely so a fuser
    that answers with a tool call isn't silently coerced into empty prose).
    A bare `str` used to reach here from the fuser leg, so this function
    briefly coerced one into a `Candidate(text)` -- that coercion is gone on
    purpose. If `call_model`'s fuser branch ever regresses back to
    `_extract_text` (which drops any tool call the upstream returned), a
    lingering coercion here would silently re-wrap the resulting bare string
    into prose and serve it as a normal 200 -- a wrong answer with nothing to
    signal it. Without the coercion, `candidate.tool_calls` on a plain `str`
    raises `AttributeError` immediately: a loud, obvious failure instead of a
    quiet, wrong one.
    """
    message: dict = {"role": "assistant"}
    if candidate.tool_calls:
        # OpenAI sends content: null alongside tool_calls unless the model
        # also produced prose; finish_reason must be "tool_calls" or clients
        # will not execute the call.
        message["content"] = candidate.text or None
        message["tool_calls"] = list(candidate.tool_calls)
        finish = "tool_calls"
    else:
        message["content"] = candidate.text
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "fusion": meta,
    }

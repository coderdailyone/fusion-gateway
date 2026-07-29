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
class PanelResult:
    conversation: str
    candidates: dict[str, str]
    reviews: dict[str, dict[str, Verdict]]
    path: str            # "quorum" | "full"
    degraded: bool


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


async def call_model(*, model_name, body, cfg, adapters, ledger, events, clock,
                     request_id, kind):
    """One billed upstream call. Returns its text, or None on failure.

    `kind` labels the event ("candidate" | "review" | "fuser"). Raises only
    BudgetTripped and asyncio.CancelledError.
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
    return _extract_text(resp)


def is_consensus(candidates: dict[str, str],
                 reviews: dict[str, dict[str, Verdict]]) -> bool:
    """True only when every candidate was reviewed `correct` by every other.

    Deliberately conservative: a missing review is NOT agreement. Absence of
    evidence sends the request down the slow path, which is the safe direction.
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

    try:
        candidates = await collect(fcfg.quorum)
        reviews = await _cross_review(
            candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
            ledger=ledger, events=events, clock=clock, request_id=request_id,
            conversation=conversation)
        agreed = is_consensus(candidates, reviews)
        events.append(request_id, "fusion.consensus",
                      {"agreed": agreed, "candidates": sorted(candidates)})

        if agreed:
            return PanelResult(conversation, candidates, reviews, "quorum",
                               degraded=len(candidates) < len(fcfg.quorum))

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
            # replacing it. A bare `reviews = round2` discarded already-paid
            # round-1 verdicts whenever round 2 failed wholly or partly (a
            # reviewer timing out or erroring on the second call), leaving
            # the fuser prompt saying "(no reviews available)" even though
            # round 1 succeeded. A reviewer present in round 2 supersedes its
            # own round-1 verdicts (it re-judged with the full candidate set,
            # which is strictly more evidence); a reviewer ABSENT from round
            # 2 keeps its round-1 verdicts rather than losing them.
            merged = {r: dict(v) for r, v in reviews.items()}
            for reviewer, verdicts in round2.items():
                merged.setdefault(reviewer, {}).update(verdicts)
            reviews = merged
        return PanelResult(conversation, candidates, reviews, "full",
                           degraded=len(candidates) < len(fcfg.panel))
    finally:
        await cancel_remaining()


def fuser_body(fcfg, panel: PanelResult, body: dict) -> dict:
    """The OpenAI body for the fuser call: one user message, no client tools."""
    out = {k: body[k] for k in _FUSER_PASSTHROUGH if k in body}
    out["messages"] = [{"role": "user", "content": build_fusion_prompt(
        panel.conversation, panel.candidates, panel.reviews)}]
    return out


def best_candidate(fcfg, panel: PanelResult):
    """The answer to fall back on when the fuser itself fails: the first
    surviving member in configured panel order."""
    for m in fcfg.panel:
        if panel.candidates.get(m):
            return m, panel.candidates[m]
    for m in sorted(panel.candidates):
        if panel.candidates[m]:
            return m, panel.candidates[m]
    return None


def openai_response(text: str, model: str, meta: dict) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "fusion": meta,
    }

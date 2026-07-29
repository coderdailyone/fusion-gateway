from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.clock import Clock, SystemClock
from gateway.config import load_config
from gateway.db import Store, connect
from gateway.events import EventLog
from gateway.fusion import (
    best_candidate, call_model, fuser_body, gather_panel, openai_response,
)
from gateway.ledger import BudgetTripped, Ledger, estimate_tokens
from gateway.policy import UnknownModel, plan_route
from gateway.providers import ProviderAdapter, ProviderError, make_adapter, parse_stream_usage

logger = logging.getLogger("gateway.app")

ORPHAN_AFTER = timedelta(hours=1)


def _parse_tokens() -> dict[str, str]:
    """GATEWAY_TOKENS="prism:tokA,admin:tokB" -> {"tokA": "prism", "tokB": "admin"}."""
    raw = os.environ.get("GATEWAY_TOKENS", "")
    tokens: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        principal, _, token = pair.partition(":")
        if token:
            tokens[token] = principal
    return tokens


def _insert_request(store: Store, request_id: str, client: str,
                     requested_model: str | None, clock: Clock) -> None:
    with store.lock:
        store.conn.execute(
            "INSERT INTO requests (id, created_at, client, requested_model, status, finished_at) "
            "VALUES (?, ?, ?, ?, 'open', NULL)",
            (request_id, clock.now().isoformat(), client, requested_model),
        )
        store.conn.commit()


def _finish_request(store: Store, request_id: str, status: str, clock: Clock) -> None:
    with store.lock:
        store.conn.execute(
            "UPDATE requests SET status=?, finished_at=? WHERE id=?",
            (status, clock.now().isoformat(), request_id),
        )
        store.conn.commit()


def _ensure_admin_sentinel(store: Store, clock: Clock) -> None:
    with store.lock:
        row = store.conn.execute(
            "SELECT 1 FROM requests WHERE id = 'admin'"
        ).fetchone()
        if row is None:
            store.conn.execute(
                "INSERT INTO requests (id, created_at, client, requested_model, status, finished_at) "
                "VALUES ('admin', ?, 'system', NULL, 'open', NULL)",
                (clock.now().isoformat(),),
            )
            store.conn.commit()


def _recover_orphans(store: Store, clock: Clock) -> None:
    """Best-effort: mark stale 'open' requests (and their preflight ledger
    rows) as 'orphaned'. Never touches the 'admin' sentinel row."""
    threshold = (clock.now() - ORPHAN_AFTER).isoformat()
    with store.lock:
        rows = store.conn.execute(
            "SELECT id FROM requests WHERE status = 'open' AND created_at < ? AND id != 'admin'",
            (threshold,),
        ).fetchall()
        orphan_ids = [row["id"] for row in rows]
        for rid in orphan_ids:
            store.conn.execute(
                "UPDATE requests SET status='orphaned' WHERE id = ?", (rid,)
            )
            store.conn.execute(
                "UPDATE ledger SET state='orphaned' WHERE request_id = ? AND state = 'preflight'",
                (rid,),
            )
        store.conn.commit()


def _as_chunks(text: str, model: str) -> list[bytes]:
    """Render a plain answer as a minimal OpenAI chunk stream.

    Used when the fuser dies before its first byte and a candidate's answer is
    served instead: the client already opened a stream, so it must receive one.
    """
    chunk = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": {"content": text},
                          "finish_reason": None}]}
    done = dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    return [f"data: {json.dumps(chunk)}\n\n".encode(),
            f"data: {json.dumps(done)}\n\n".encode(),
            b"data: [DONE]\n\n"]


def create_app_from_env() -> FastAPI:
    """uvicorn --factory entrypoint: gateway.app:create_app_from_env.

    Reads GATEWAY_CONFIG (config toml path, default configs/gateway.toml)
    and GATEWAY_DB (sqlite path, default data/gateway.sqlite).
    """
    config_path = os.environ.get("GATEWAY_CONFIG", "configs/gateway.toml")
    db_path = os.environ.get("GATEWAY_DB", "data/gateway.sqlite")
    return create_app(config_path, db_path)


def create_app(
    config_path: str | Path,
    db_path: str | Path,
    clock: Clock | None = None,
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
) -> FastAPI:
    clock = clock or SystemClock()
    cfg = load_config(config_path)
    conn = connect(db_path)
    store = Store(conn)
    events = EventLog(store, clock)
    tokens = _parse_tokens()

    def _alert_cb(consumed: float, cap: float) -> None:
        logger.warning("budget.alert consumed=%s cap=%s", consumed, cap)

    ledger = Ledger(
        store,
        clock,
        cap_usd=cfg.budget_caps[cfg.active_budget],
        budget_name=cfg.active_budget,
        alert_cb=_alert_cb,
    )

    adapters: dict[str, ProviderAdapter] = {}
    for name, provider_cfg in cfg.providers.items():
        transport = (transports or {}).get(name)
        adapters[name] = make_adapter(provider_cfg, transport=transport)

    _ensure_admin_sentinel(store, clock)
    _recover_orphans(store, clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Adapter lifecycle: the providers task left no public close method,
        # so closing the internal httpx client here is acceptable to avoid
        # leaking connections.
        for adapter in adapters.values():
            await adapter._client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.events = events
    app.state.ledger = ledger
    app.state.adapters = adapters
    app.state.clock = clock
    app.state.tokens = tokens

    async def get_principal(request: Request) -> str:
        auth = request.headers.get("authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = auth[len("Bearer "):]
        principal = tokens.get(token)
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid token")
        return principal

    async def require_admin(principal: str = Depends(get_principal)) -> str:
        if principal != "admin":
            raise HTTPException(status_code=403, detail="admin only")
        return principal

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def list_models(principal: str = Depends(get_principal)):
        return {"data": [{"id": name} for name in cfg.models]}

    async def _fusion_request(*, request_id, body, streaming, fcfg):
        common = dict(fcfg=fcfg, cfg=cfg, adapters=adapters, ledger=ledger,
                      events=events, clock=clock, request_id=request_id)

        if not streaming:
            try:
                panel = await gather_panel(body=body, **common)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.model})
                _finish_request(store, request_id, "failed", clock)
                return JSONResponse(status_code=503,
                                    content={"error": {"type": "budget_exhausted"}})

            try:
                text, source = await _finish_fusion(panel, body, fcfg, common,
                                                    request_id)
            except BudgetTripped:
                # The fuser's own preflight (inside call_model) tripped the
                # budget after the panel already spent -- e.g. a killswitch
                # trip mid-panel, or a concurrent request crossing the cap in
                # the window between gather_panel and this preflight. Without
                # this catch it propagated out of the handler as a gateway
                # 500 with the `requests` row left 'open'.
                events.append(request_id, "budget.tripped", {"model": fcfg.fuser})
                _finish_request(store, request_id, "failed", clock)
                return JSONResponse(status_code=503,
                                    content={"error": {"type": "budget_exhausted"}})
            if text is None:
                # Final review, finding 1b: paying for a panel and then
                # hard-failing is the worst outcome available. Fall through
                # to the single-model chain, starting from the panel's
                # preferred member -- the same thing the client would have
                # gotten by naming it explicitly, including ITS OWN fallback
                # chain (which can reach a model outside the panel that may
                # still be up when every panel member is down).
                events.append(request_id, "fusion.degraded",
                              {"rung": "zero_candidates_chain_fallback",
                               "model": fcfg.panel[0]})
                chain_plan = plan_route(cfg, fcfg.panel[0])
                return await _run_chain_once(request_id, body, chain_plan.chain)
            _finish_request(store, request_id, "succeeded", clock)
            meta = {"path": panel.path, "panel": sorted(panel.candidates),
                    "fuser": fcfg.fuser, "degraded": panel.degraded or source != "fuser",
                    "answered_by": source}
            events.append(request_id, "fusion.fused",
                          {"fuser": fcfg.fuser, "path": panel.path, "source": source})
            return JSONResponse(content=openai_response(text, fcfg.model, meta),
                                headers={"x-fusion-trace-id": request_id})

        async def gen():
            # Stages 1-2 produce nothing visible and can take tens of seconds.
            # SSE comment lines are the spec's keepalive and are skipped by
            # conformant parsers (including the OpenAI SDKs), so an idle client
            # timeout does not fire while the panel works.
            yield b": fusion panel\n\n"
            try:
                panel = await gather_panel(body=body, **common)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.model})
                _finish_request(store, request_id, "failed", clock)
                # Defensive hardening alongside finding 2's fix below: this
                # site is provably safe already (the preceding keepalive
                # comment always ends its own line cleanly), but a leading
                # blank line is a no-op for a conformant parser, so there is
                # no reason for this generator to have any error yield that
                # ISN'T self-contained regardless of what precedes it.
                yield b'\n\ndata: {"error": {"type": "budget_exhausted"}}\n\n'
                return
            yield b": fusion fusing\n\n"

            fbody = dict(fuser_body(fcfg, panel, body), stream=True)
            model_cfg = cfg.models[fcfg.fuser]
            adapter = adapters[model_cfg.provider]
            est_in, est_out = estimate_tokens(fbody["messages"],
                                              fbody.get("max_tokens"))
            try:
                entry_id = ledger.preflight(
                    request_id, model_cfg.provider, fcfg.fuser, est_in, est_out,
                    model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.fuser})
                _finish_request(store, request_id, "failed", clock)
                yield b'\n\ndata: {"error": {"type": "budget_exhausted"}}\n\n'
                return

            events.append(request_id, "call.attempt",
                          {"model": fcfg.fuser, "stage": "fuser"})
            start = clock.now()
            deadline = time.monotonic() + fcfg.stage_timeout_s
            accumulated = bytearray()
            first_byte = False
            resolved = False   # True once this row has been settle()d or fail()d.

            async def _fuser_gave_nothing(kind):
                """The fuser produced zero bytes -- a pre-first-byte error, a
                pre-first-byte timeout, or a 2xx with an empty body all land
                here. Fail its ledger row and fall back to the best
                candidate, rendered through _as_chunks since the client
                already has an open stream and must still get one."""
                nonlocal resolved
                resolved = True
                ledger.fail(entry_id)
                events.append(request_id, "call.failed",
                              {"model": fcfg.fuser, "stage": "fuser",
                               "kind": kind})
                fallback = best_candidate(fcfg, panel)
                if fallback is None:
                    _finish_request(store, request_id, "failed", clock)
                    yield b'\n\ndata: {"error": {"type": "upstream_exhausted"}}\n\n'
                    return
                events.append(request_id, "fusion.degraded",
                              {"rung": "fuser_failed", "model": fallback[0]})
                _finish_request(store, request_id, "succeeded", clock)
                for piece in _as_chunks(fallback[1], fcfg.model):
                    yield piece

            try:
                try:
                    stream_iter = adapter.chat_stream(
                        model_cfg.upstream_model, fbody).__aiter__()
                    while True:
                        remaining = max(0.0, deadline - time.monotonic())
                        try:
                            # stage_timeout_s bounds stages 1-2 already; bound
                            # the fuser's own stream the same way -- per
                            # chunk, not for the whole call, so a fuser that
                            # is genuinely still producing output at the
                            # deadline isn't cut off mid-token, only one that
                            # goes quiet is. That requires `deadline` itself
                            # to be reset after every chunk (below) -- a
                            # deadline computed once before this loop and
                            # never touched again is a WALL-CLOCK bound on
                            # the whole call, the exact thing this comment
                            # says is not what happens here (final review,
                            # finding 7).
                            chunk = await asyncio.wait_for(
                                stream_iter.__anext__(), timeout=remaining)
                        except StopAsyncIteration:
                            break
                        first_byte = True
                        accumulated.extend(chunk)
                        yield chunk
                        deadline = time.monotonic() + fcfg.stage_timeout_s
                except Exception:
                    if not first_byte:
                        async for piece in _fuser_gave_nothing("unknown"):
                            yield piece
                        return
                    logger.exception("fusion stream failed request_id=%s", request_id)
                    resolved = True
                    ledger.settle(entry_id, est_in, max(len(accumulated) // 4, 0),
                                  "estimated",
                                  int((clock.now() - start).total_seconds() * 1000),
                                  model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
                    _finish_request(store, request_id, "failed", clock)
                    # Final review, finding 2 (CRITICAL): `accumulated` holds
                    # raw upstream bytes forwarded via bare `yield chunk`
                    # above, cut at an arbitrary socket boundary -- there is
                    # no guarantee the last byte sent was a clean SSE line
                    # terminator. Concatenating the error envelope directly
                    # onto that produced e.g. `..."content":"wordata:
                    # {"error"...`, which the real openai SDK's SSEDecoder
                    # cannot parse (JSONDecodeError, not a clean recognized
                    # API error). A leading blank line is a no-op for a
                    # conformant parser and closes out whatever line was left
                    # open.
                    yield b'\n\ndata: {"error": {"type": "stream_failed"}}\n\n'
                    return

                if not first_byte:
                    # A 2xx that yielded no bytes: nothing reached the client,
                    # so it's still safe to fall back (mirrors the
                    # single-model streaming loop's `empty_stream` handling).
                    async for piece in _fuser_gave_nothing("empty_stream"):
                        yield piece
                    return

                latency_ms = int((clock.now() - start).total_seconds() * 1000)
                raw = bytes(accumulated)
                usage = parse_stream_usage(raw)
                if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
                    in_tok, out_tok, src = (usage["prompt_tokens"],
                                            usage["completion_tokens"], "reported")
                else:
                    in_tok, out_tok, src = est_in, max(len(raw) // 4, 0), "estimated"
                resolved = True
                ledger.settle(entry_id, in_tok, out_tok, src, latency_ms,
                              model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
                events.append(request_id, "fusion.fused",
                              {"fuser": fcfg.fuser, "path": panel.path, "source": "fuser"})
                _finish_request(store, request_id, "succeeded", clock)
            finally:
                if not resolved:
                    # Reached only via a client disconnect / server shutdown:
                    # asyncio.CancelledError and GeneratorExit are
                    # BaseException, not Exception, so they skip every
                    # `except Exception` above and land here instead. The
                    # upstream call may already be billing, so settle at the
                    # preflight estimate -- same treatment call_model gives a
                    # cancelled leg in fusion.py -- rather than leaving the
                    # row in 'preflight' (a CONSUMING_STATE only a restart
                    # clears) or fail()ing it (which would post $0 for work
                    # that may already be billed). `resolved` guards every
                    # other exit path above, so this can fire at most once.
                    ledger.settle(entry_id, est_in, max(len(accumulated) // 4, 0),
                                  "estimated",
                                  int((clock.now() - start).total_seconds() * 1000),
                                  model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"x-fusion-trace-id": request_id})

    async def _finish_fusion(panel, body, fcfg, common, request_id):
        """Run the fuser. Returns (text, source) with source in
        {"fuser", "candidate"}, or (None, "none") when nothing survived."""
        if len(panel.candidates) < 2:
            fallback = best_candidate(fcfg, panel)
            if fallback is None:
                events.append(request_id, "fusion.degraded",
                              {"rung": "no_candidates"})
                return None, "none"
            events.append(request_id, "fusion.degraded",
                          {"rung": "single_candidate", "model": fallback[0]})
            return fallback[1], "candidate"

        try:
            # stage_timeout_s bounds stages 1-2 (gather_panel) already;
            # without a bound here the fuser call falls back to the
            # adapter's 120s httpx default regardless of what the config
            # says. call_model's own CancelledError handling (settle at the
            # preflight estimate, then re-raise) is exactly what
            # asyncio.wait_for's cancel-on-timeout triggers, so the ledger
            # row is never left in 'preflight' by this timeout.
            text = await asyncio.wait_for(
                call_model(model_name=fcfg.fuser,
                          body=fuser_body(fcfg, panel, body),
                          kind="fuser",
                          **{k: v for k, v in common.items() if k != "fcfg"}),
                timeout=fcfg.stage_timeout_s)
        except asyncio.TimeoutError:
            events.append(request_id, "call.failed",
                          {"model": fcfg.fuser, "stage": "fuser", "kind": "timeout"})
            text = None
        if text:
            return text, "fuser"
        fallback = best_candidate(fcfg, panel)
        if fallback is None:
            return None, "none"
        events.append(request_id, "fusion.degraded",
                      {"rung": "fuser_failed", "model": fallback[0]})
        return fallback[1], "candidate"

    async def _run_chain_once(request_id, body, chain):
        """Try each model in `chain` in order (non-streaming); always
        returns a JSONResponse. This is the plain single-model path's own
        loop, factored out so the fusion path's zero-usable-candidates
        fallback (finding 1b, above) can reuse it exactly rather than
        reimplementing the same preflight/settle/fallback bookkeeping."""
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens")

        for model_name in chain:
            model_cfg = cfg.models[model_name]
            adapter = adapters[model_cfg.provider]
            est_in, est_out = estimate_tokens(messages, max_tokens)

            try:
                entry_id = ledger.preflight(
                    request_id, model_cfg.provider, model_name,
                    est_in, est_out, model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
                )
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": model_name})
                _finish_request(store, request_id, "failed", clock)
                return JSONResponse(
                    status_code=503,
                    content={"error": {"type": "budget_exhausted"}},
                )

            events.append(request_id, "call.attempt", {"model": model_name})
            start = clock.now()
            try:
                upstream_resp = await adapter.chat(model_cfg.upstream_model, body)
            except ProviderError as exc:
                ledger.fail(entry_id)
                events.append(
                    request_id, "call.failed",
                    {"model": model_name, "kind": exc.kind, "status": exc.status},
                )
                continue
            except Exception:
                # Mirrors the streaming loop's net. Nothing has reached the
                # client on this path, so any failure is as safe to fall back
                # from as a ProviderError — and without this the ledger row
                # stays in 'preflight', a CONSUMING_STATE that only
                # _recover_orphans clears, and that runs at startup only. The
                # translator raises on client-controlled JSON (`stop: 5`,
                # `tools: "abc"`), and httpx.DecodingError / TooManyRedirects
                # are RequestError but not TransportError, so neither adapter's
                # except clauses catch them either.
                logger.exception("call.failed model=%s request_id=%s",
                                 model_name, request_id)
                ledger.fail(entry_id)
                events.append(
                    request_id, "call.failed",
                    {"model": model_name, "kind": "unknown"},
                )
                continue

            latency_ms = int((clock.now() - start).total_seconds() * 1000)
            usage = upstream_resp.get("usage") or {}
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                in_tokens = usage["prompt_tokens"]
                out_tokens = usage["completion_tokens"]
                usage_source = "reported"
            else:
                in_tokens, out_tokens = est_in, est_out
                usage_source = "estimated"

            ledger.settle(
                entry_id, in_tokens, out_tokens, usage_source, latency_ms,
                model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
            )
            events.append(request_id, "call.succeeded", {"model": model_name})
            _finish_request(store, request_id, "succeeded", clock)

            upstream_resp["model"] = model_name
            return JSONResponse(
                content=upstream_resp,
                headers={"x-fusion-trace-id": request_id},
            )

        _finish_request(store, request_id, "failed", clock)
        return JSONResponse(status_code=502, content={"error": {"type": "upstream_exhausted"}})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, principal: str = Depends(get_principal)):
        body = await request.json()
        streaming = bool(body.get("stream"))

        request_id = uuid.uuid4().hex
        requested_model = body.get("model") or ""
        _insert_request(store, request_id, principal, requested_model, clock)
        events.append(request_id, "request.received",
                       {"model": requested_model, "client": principal})

        fcfg = cfg.fusion
        resolved = (cfg.default_model
                    if requested_model in ("", "auto") else requested_model)
        if fcfg is not None and resolved == fcfg.model:
            # Re-review residual 1: `functions`/`function_call` is the
            # deprecated OpenAI shape `tools`/`tool_choice` replaced, but
            # real clients still send it. It carries the exact same
            # semantics-loss `tools`/`tool_choice` does (a fuser writing
            # prose over what should be a function call), so it gets the
            # same over-conservative-in-the-safe-direction treatment.
            wants_tools = (bool(body.get("tools")) or body.get("tool_choice") is not None
                          or bool(body.get("functions")) or body.get("function_call") is not None)
            if not wants_tools:
                return await _fusion_request(
                    request_id=request_id, body=body, streaming=streaming,
                    fcfg=fcfg,
                )
            # Final review, finding 1a (CRITICAL): fusing a tool call is not
            # meaningful -- the fuser writes prose, and there is no sound way
            # to merge divergent tool invocations. The standard OpenAI
            # function-call shape (`message.content: null`, `tool_calls:
            # [...]`) makes `_extract_text` return "" for every candidate, so
            # the panel ended up with zero usable candidates, was billed in
            # full, and then handed back a 502 -- while naming the same
            # model explicitly worked fine (see the corrected "Conversation
            # rendering" section of the M8 spec). Route to the single-model
            # chain instead, starting from the panel's preferred member, the
            # same as if the client had named it explicitly.
            events.append(request_id, "fusion.bypassed",
                          {"reason": "tools", "model": fcfg.panel[0]})
            requested_model = fcfg.panel[0]

        try:
            plan = plan_route(cfg, requested_model)
        except UnknownModel:
            _finish_request(store, request_id, "failed", clock)
            events.append(request_id, "route.failed", {"reason": "unknown_model"})
            return JSONResponse(status_code=400, content={"error": {"type": "unknown_model"}})

        with store.lock:
            store.conn.execute(
                "INSERT INTO decisions (request_id, policy_version, action, features, degraded) "
                "VALUES (?, ?, 'route', ?, 0)",
                (request_id, plan.policy_version, json.dumps({"chain": list(plan.chain)})),
            )
            store.conn.commit()
        events.append(request_id, "route.planned",
                       {"chain": list(plan.chain), "policy_version": plan.policy_version})

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens")

        if streaming:
            async def gen():
                for model_name in plan.chain:
                    model_cfg = cfg.models[model_name]
                    adapter = adapters[model_cfg.provider]
                    est_in, est_out = estimate_tokens(messages, max_tokens)

                    try:
                        entry_id = ledger.preflight(
                            request_id, model_cfg.provider, model_name,
                            est_in, est_out, model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
                        )
                    except BudgetTripped:
                        events.append(request_id, "budget.tripped", {"model": model_name})
                        _finish_request(store, request_id, "failed", clock)
                        yield b'\n\ndata: {"error": {"type": "budget_exhausted"}}\n\n'
                        return

                    events.append(request_id, "call.attempt", {"model": model_name})
                    start = clock.now()
                    accumulated = bytearray()
                    first_byte = False
                    try:
                        async for chunk in adapter.chat_stream(model_cfg.upstream_model, body):
                            first_byte = True
                            accumulated.extend(chunk)
                            yield chunk
                    except ProviderError as exc:
                        # Adapter contract: ProviderError is only raised before
                        # the first byte reaches the client, so it's always
                        # safe to fall back to the next model in the chain.
                        ledger.fail(entry_id)
                        events.append(
                            request_id, "call.failed",
                            {"model": model_name, "kind": exc.kind, "status": exc.status},
                        )
                        continue
                    except Exception:
                        if not first_byte:
                            # Defensive: treat any pre-first-byte failure like
                            # a ProviderError and fall back.
                            ledger.fail(entry_id)
                            events.append(
                                request_id, "call.failed",
                                {"model": model_name, "kind": "unknown"},
                            )
                            continue
                        latency_ms = int((clock.now() - start).total_seconds() * 1000)
                        in_tokens = est_in
                        out_tokens = max(len(accumulated) // 4, 0)
                        ledger.settle(
                            entry_id, in_tokens, out_tokens, "estimated", latency_ms,
                            model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
                        )
                        events.append(request_id, "call.failed",
                                       {"model": model_name, "kind": "stream_error"})
                        _finish_request(store, request_id, "failed", clock)
                        # Final review, finding 2 (CRITICAL) -- the ORIGINAL
                        # site (inherited by the fusion path above). Same
                        # reasoning: `accumulated` is raw, arbitrarily-cut
                        # upstream bytes, so a leading blank line before the
                        # error envelope is required, not decorative.
                        yield b'\n\ndata: {"error": {"type": "stream_failed"}}\n\n'
                        return

                    if not first_byte:
                        # Upstream returned a 2xx with an empty body: no bytes
                        # reached the client, so it's still safe to fall back.
                        ledger.fail(entry_id)
                        events.append(request_id, "call.failed",
                                       {"model": model_name, "kind": "empty_stream"})
                        continue

                    latency_ms = int((clock.now() - start).total_seconds() * 1000)
                    raw = bytes(accumulated)
                    usage = parse_stream_usage(raw)
                    if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
                        in_tokens = usage["prompt_tokens"]
                        out_tokens = usage["completion_tokens"]
                        usage_source = "reported"
                    else:
                        in_tokens = est_in
                        out_tokens = max(len(raw) // 4, 0)
                        usage_source = "estimated"

                    ledger.settle(
                        entry_id, in_tokens, out_tokens, usage_source, latency_ms,
                        model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
                    )
                    events.append(request_id, "call.succeeded", {"model": model_name})
                    _finish_request(store, request_id, "succeeded", clock)
                    return

                _finish_request(store, request_id, "failed", clock)
                yield b'\n\ndata: {"error": {"type": "upstream_exhausted"}}\n\n'

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-fusion-trace-id": request_id},
            )

        return await _run_chain_once(request_id, body, plan.chain)

    @app.get("/admin/status")
    async def admin_status(principal: str = Depends(require_admin)):
        with store.lock:
            rows = store.conn.execute(
                "SELECT status, COUNT(*) AS c FROM requests WHERE id != 'admin' GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["c"] for row in rows}
        return {"ledger": ledger.status(), "requests": counts}

    @app.post("/admin/killswitch/trip")
    async def killswitch_trip(principal: str = Depends(require_admin)):
        # With no cap configured the automatic trip never fires, so this is the
        # operator's only way to stop spending short of stopping the service.
        ledger.trip()
        events.append("admin", "killswitch.tripped", {"by": principal})
        return {"ok": True, "state": ledger.status()["state"]}

    @app.post("/admin/killswitch/release")
    async def killswitch_release(principal: str = Depends(require_admin)):
        ledger.release()
        events.append("admin", "killswitch.released", {"by": principal})
        return {"ok": True}

    return app

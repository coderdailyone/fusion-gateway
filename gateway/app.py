from __future__ import annotations

import json
import logging
import os
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
from gateway.ledger import BudgetTripped, Ledger, estimate_tokens
from gateway.policy import UnknownModel, plan_route
from gateway.providers import ProviderAdapter, ProviderError, parse_stream_usage

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
        adapters[name] = ProviderAdapter(provider_cfg, transport=transport)

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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, principal: str = Depends(get_principal)):
        body = await request.json()
        streaming = bool(body.get("stream"))

        request_id = uuid.uuid4().hex
        requested_model = body.get("model") or ""
        _insert_request(store, request_id, principal, requested_model, clock)
        events.append(request_id, "request.received",
                       {"model": requested_model, "client": principal})

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
                        yield b'data: {"error": {"type": "budget_exhausted"}}\n\n'
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
                        yield b'data: {"error": {"type": "stream_failed"}}\n\n'
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
                yield b'data: {"error": {"type": "upstream_exhausted"}}\n\n'

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-fusion-trace-id": request_id},
            )

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

    @app.get("/admin/status")
    async def admin_status(principal: str = Depends(require_admin)):
        with store.lock:
            rows = store.conn.execute(
                "SELECT status, COUNT(*) AS c FROM requests WHERE id != 'admin' GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["c"] for row in rows}
        return {"ledger": ledger.status(), "requests": counts}

    @app.post("/admin/killswitch/release")
    async def killswitch_release(principal: str = Depends(require_admin)):
        ledger.release()
        events.append("admin", "killswitch.released", {"by": principal})
        return {"ok": True}

    return app

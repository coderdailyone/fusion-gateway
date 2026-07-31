"""The Anthropic wire driven end to end through the real gateway app.

Nothing here is a fake gateway: create_app builds the real request lifecycle,
the real fallback chain and the real Ledger, with httpx.MockTransport standing
in only for the socket. That matters because every defect these tests pin is a
money defect -- the assertion is on what the LEDGER recorded, not on what the
adapter returned.
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from tests.helpers import FakeClock

# An anthropic-wire primary with an openai-wire fallback: exactly the topology
# of the upcoming live smoke (glm-5.2 -> deepseek).
CONFIG = """
[budget]
active = "T"
[budgets.T]
cap_usd = 5.0

[providers.glm_anthropic]
base_url = "https://example.test/anthropic"
api_key_env = "GLM_ANTHROPIC_KEY"
wire = "anthropic"

[providers.deepseek]
base_url = "https://example.test/v1"
api_key_env = "DEEPSEEK_API_KEY"

[models."glm-5.2"]
provider = "glm_anthropic"
upstream_model = "glm-5.2"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 2.0
fallback = ["deepseek-chat"]

[models."deepseek-chat"]
provider = "deepseek"
upstream_model = "deepseek-v4-flash"
in_usd_per_mtok = 0.14
out_usd_per_mtok = 0.28
fallback = []

[policy]
version = "test-v0"
default_model = "glm-5.2"
"""

OPENAI_SSE = (b'data: {"choices":[{"delta":{"content":"fallback"}}]}\n\n'
              b'data: {"choices":[],"usage":{"prompt_tokens":5,'
              b'"completion_tokens":2}}\n\n'
              b'data: [DONE]\n\n')


def openai_ok(request: httpx.Request) -> httpx.Response:
    """The openai-wire fallback, answering either mode."""
    if json.loads(request.content).get("stream"):
        return httpx.Response(200, content=OPENAI_SSE,
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "id": "cmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant",
                                             "content": "fallback"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}})


def make_client(tmp_path, monkeypatch, anthropic, deepseek=openai_ok):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("GLM_ANTHROPIC_KEY", "sk-a")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-d")
    cfg_path = tmp_path / "anthropic.toml"
    cfg_path.write_text(CONFIG)
    app = create_app(
        cfg_path, tmp_path / "g.sqlite", clock=FakeClock(),
        transports={"glm_anthropic": httpx.MockTransport(anthropic),
                    "deepseek": httpx.MockTransport(deepseek)})
    return TestClient(app)


def H(tok="tokA"):
    return {"Authorization": f"Bearer {tok}"}


BODY = {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}


def ledger_rows(client) -> list[dict]:
    store = client.app.state.store
    with store.lock:
        rows = store.conn.execute(
            "SELECT model, state, est_cost_usd, actual_cost_usd, in_tokens, "
            "out_tokens, usage_source FROM ledger ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def event_kinds(client) -> list[str]:
    store = client.app.state.store
    with store.lock:
        rows = store.conn.execute(
            "SELECT kind FROM events WHERE request_id != 'admin' ORDER BY seq"
        ).fetchall()
    return [r["kind"] for r in rows]


# --- F1: a 200 that never stated usage must not settle as an authoritative $0 --


def test_a_200_without_usage_is_billed_by_estimate_not_as_a_reported_zero(
        tmp_path, monkeypatch):
    """The upstream answered, with real content, and said nothing about tokens.

    app.py branches on key presence, so a manufactured usage dict makes that a
    'reported' settlement of in=0/out=0/$0.00: the client is served, the money
    is spent, and the budget guard is told nothing was consumed. Absent an
    upstream count the only honest answer is an estimate."""
    def handler(request):
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "a real, billable answer"}]})

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())

    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "a real, billable answer"
    assert r.json()["model"] == "glm-5.2"

    rows = ledger_rows(c)
    assert [(row["model"], row["state"]) for row in rows] == [("glm-5.2", "settled")]
    assert rows[0]["usage_source"] == "estimated"
    assert rows[0]["actual_cost_usd"] > 0, "a served answer settled at $0.00"
    status = c.get("/admin/status", headers=H("tokB")).json()["ledger"]
    assert status["consumed_usd"] > 0, "the budget guard was told nothing was spent"


def test_a_200_with_only_an_input_count_is_not_billed_as_zero_completion(
        tmp_path, monkeypatch):
    """The half-usage variant: a stated prompt count and no completion count.
    app.py consumes the usage dict wholesale, so half a report is not usable --
    completion_tokens: 0 would be recorded as authoritative."""
    def handler(request):
        return httpx.Response(200, json={
            "id": "msg_1", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "words the model was paid for"}],
            "usage": {"input_tokens": 900}})

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    assert c.post("/v1/chat/completions", json=BODY, headers=H()).status_code == 200

    row = ledger_rows(c)[0]
    assert row["usage_source"] == "estimated"
    assert row["out_tokens"] > 0 and row["actual_cost_usd"] > 0


def test_a_200_carrying_an_error_body_falls_back_instead_of_billing_zero(
        tmp_path, monkeypatch):
    """This endpoint returns balance errors as HTTP 200 (configs/gateway.toml).
    Accepting it hands the client content: null as a success, bills $0.00 as
    reported, and never tries deepseek."""
    def handler(request):
        return httpx.Response(200, json={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "1113 balance"}})

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())

    assert r.status_code == 200
    assert r.json()["model"] == "deepseek-chat"
    assert r.json()["choices"][0]["message"]["content"] == "fallback"
    assert [(row["model"], row["state"]) for row in ledger_rows(c)] == [
        ("glm-5.2", "failed"), ("deepseek-chat", "settled")]
    assert "call.failed" in event_kinds(c)


# --- F2: a translator crash must not strand a budget-consuming ledger row ------


BAD_BODIES = [
    {"stop": 5},                                    # TypeError: list(5)
    {"tools": "abc"},                               # AttributeError: str.get
    {"tool_choice": {"function": "nope"}},          # AttributeError: str.get
    {"messages": [{"role": "assistant", "tool_calls": "y"}]},   # AttributeError
]


def test_a_client_body_that_breaks_the_translator_frees_the_row_and_falls_back(
        tmp_path, monkeypatch):
    """to_anthropic_request is hardened against upstream JSON but not against
    CLIENT JSON, and /v1/chat/completions does no schema validation. The
    streaming loop nets these; the non-streaming loop did not, so the request
    500s with its ledger row stuck in 'preflight' -- a CONSUMING_STATE that only
    _recover_orphans clears, and that runs at startup only. With a large
    max_tokens a handful of such requests exhaust the cap until a restart."""
    def handler(request):   # pragma: no cover - must never be reached
        raise AssertionError("the translator should have failed first")

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    for bad in BAD_BODIES:
        r = c.post("/v1/chat/completions",
                   json={**BODY, "max_tokens": 100_000, **bad}, headers=H())
        assert r.status_code == 200, bad
        assert r.json()["model"] == "deepseek-chat", bad

    rows = ledger_rows(c)
    stranded = [row for row in rows if row["state"] == "preflight"]
    assert not stranded, f"{len(stranded)} rows still consuming budget"
    assert [row["state"] for row in rows if row["model"] == "glm-5.2"] \
        == ["failed"] * len(BAD_BODIES)

    status = c.get("/admin/status", headers=H("tokB")).json()["ledger"]
    assert status["state"] == "active"
    # Each stranded preflight would have held est_out * $2/Mtok = $0.20.
    assert status["consumed_usd"] < 0.01, status


def test_a_translator_crash_on_the_streaming_path_still_falls_back(
        tmp_path, monkeypatch):
    """The streaming loop's net already worked; pin it so the shared contract
    does not regress when the non-streaming one gains its own."""
    def handler(request):   # pragma: no cover - must never be reached
        raise AssertionError("the translator should have failed first")

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True, "stop": 5}, headers=H()) as r:
        raw = b"".join(r.iter_raw())

    assert b"fallback" in raw and b"[DONE]" in raw
    assert not [row for row in ledger_rows(c) if row["state"] == "preflight"]


# --- F4: a non-SSE or empty streaming body must not become an empty answer ----


@pytest.mark.parametrize("content,ctype", [
    (b"", "text/event-stream"),
    (b'{"error": {"message": "1113 balance"}}', "application/json"),
])
def test_a_streaming_body_with_no_events_falls_back_instead_of_answering_empty(
        tmp_path, monkeypatch, content, ctype):
    """Without the guard the client receives a well-formed, EMPTY completion
    with finish_reason 'stop', billed by estimate and logged as call.succeeded,
    and deepseek is never tried."""
    def handler(request):
        return httpx.Response(200, content=content,
                              headers={"content-type": ctype})

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        raw = b"".join(r.iter_raw())

    assert b"fallback" in raw, "served the fabricated empty answer"
    assert [(row["model"], row["state"]) for row in ledger_rows(c)] == [
        ("glm-5.2", "failed"), ("deepseek-chat", "settled")]
    assert event_kinds(c).count("call.succeeded") == 1


def test_a_well_formed_anthropic_stream_still_settles_from_reported_usage(
        tmp_path, monkeypatch):
    """The control case for all of the above: a real stream still bills real
    tokens through the real ledger, on the primary, with no fallback."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 900}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 400}},
    ]
    body = b"".join(b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
                    for e in events)

    def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    c = make_client(tmp_path, monkeypatch, anthropic=handler)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        raw = b"".join(r.iter_raw())

    assert b"[DONE]" in raw and b"fallback" not in raw
    row, = ledger_rows(c)
    assert (row["model"], row["state"], row["usage_source"]) == (
        "glm-5.2", "settled", "reported")
    assert (row["in_tokens"], row["out_tokens"]) == (900, 400)

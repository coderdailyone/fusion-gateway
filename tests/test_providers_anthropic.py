import json

import httpx
import pytest

from gateway.config import ProviderCfg
from gateway.providers import (ProviderAdapter, ProviderError, make_adapter,
                               parse_stream_usage)
from gateway.providers_anthropic import AnthropicAdapter

CFG = ProviderCfg("glm_anthropic", "https://example.test/anthropic", "X_KEY",
                  wire="anthropic")

# A complete, well-formed Anthropic SSE exchange: 8 prompt tokens in,
# 3 completion tokens out, one text delta.
EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 8}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "hi"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
     "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
]
BODY = b"".join(b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
                for e in EVENTS)
# WHATWG SSE permits CRLF; the same exchange, framed that way.
CRLF_BODY = b"".join(b"event: x\r\ndata: " + json.dumps(e).encode() + b"\r\n\r\n"
                     for e in EVENTS)
# Ends on message_delta -- the event carrying output_tokens -- so a body cut
# short here loses the only count that makes the ledger bill real tokens.
BODY_NO_TRAILING_BLANK = b"".join(
    b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
    for e in EVENTS[:4]).rstrip(b"\n")


def _adapter(handler, monkeypatch):
    monkeypatch.setenv("X_KEY", "secret-key")
    return AnthropicAdapter(CFG, transport=httpx.MockTransport(handler))


def _sliced(body: bytes, size: int):
    """An async byte stream that cuts `body` at fixed offsets, ignoring SSE
    framing — exactly what a real socket does."""
    async def gen():
        for i in range(0, len(body), size):
            yield body[i:i + size]
    return gen()


async def _collect(agen) -> bytes:
    out = bytearray()
    async for chunk in agen:
        out.extend(chunk)
    return bytes(out)


def _data_objects(collected: bytes) -> list[dict]:
    """Every JSON object the emitted stream carries on a `data:` line."""
    objs = []
    for line in collected.decode().splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        objs.append(json.loads(payload))
    return objs


def test_factory_picks_the_adapter_from_the_wire_field():
    openai_cfg = ProviderCfg("p", "https://example.test/v1", "X_KEY")
    anthropic_cfg = ProviderCfg("q", "https://example.test/anthropic", "X_KEY",
                                wire="anthropic")
    assert isinstance(make_adapter(openai_cfg), ProviderAdapter)
    assert isinstance(make_adapter(anthropic_cfg), AnthropicAdapter)


@pytest.mark.anyio
async def test_chat_posts_to_v1_messages_with_anthropic_headers(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}})

    out = await _adapter(handler, monkeypatch).chat(
        "glm-5.2", {"messages": [{"role": "user", "content": "hi"}]})

    assert seen["url"] == "https://example.test/anthropic/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert "authorization" not in {k.lower() for k in seen["headers"]}
    assert seen["body"]["model"] == "glm-5.2"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["usage"]["prompt_tokens"] == 5


@pytest.mark.anyio
async def test_chat_raises_provider_error_on_non_2xx(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ProviderError) as exc:
        await _adapter(handler, monkeypatch).chat(
            "m", {"messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.kind == "http" and exc.value.status == 429


@pytest.mark.anyio
async def test_chat_rejects_a_non_object_upstream_body(monkeypatch):
    """from_anthropic_response assumes a dict by design; a 200 carrying a JSON
    array must fail loudly, not become an empty completion the ledger bills."""
    def handler(request):
        return httpx.Response(200, json=["not", "an", "object"])

    with pytest.raises(ProviderError) as exc:
        await _adapter(handler, monkeypatch).chat(
            "m", {"messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.kind == "http" and exc.value.status == 200


@pytest.mark.anyio
async def test_chat_rejects_an_unparseable_upstream_body(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=b"<html>gateway timeout</html>",
                              headers={"content-type": "application/json"})

    with pytest.raises(ProviderError) as exc:
        await _adapter(handler, monkeypatch).chat(
            "m", {"messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.kind == "http" and exc.value.status == 200


@pytest.mark.anyio
async def test_chat_timeout_and_network_map_to_provider_error_kinds(monkeypatch):
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderError) as exc:
        await _adapter(timeout, monkeypatch).chat("m", {"messages": []})
    assert exc.value.kind == "timeout" and exc.value.status is None

    def network(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as exc:
        await _adapter(network, monkeypatch).chat("m", {"messages": []})
    assert exc.value.kind == "network" and exc.value.status is None


@pytest.mark.anyio
async def test_stream_translates_and_ends_with_parseable_usage(monkeypatch):
    def handler(request):
        assert json.loads(request.content).get("stream") is True
        return httpx.Response(200, content=BODY,
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}], "stream": True}))

    text = collected.decode()
    assert text.rstrip().endswith("data: [DONE]")
    assert '"chat.completion.chunk"' in text
    # The REAL ledger parser, not a copy: if this breaks, every stream bills 0.
    assert parse_stream_usage(collected) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}


@pytest.mark.anyio
async def test_stream_reassembles_data_lines_split_across_byte_chunks(monkeypatch):
    """aiter_bytes cuts at arbitrary offsets. Feeding whole SSE events to
    iter_sse_data is the adapter's job; a `data:` line split across two reads
    is otherwise dropped silently and the ledger bills zero output tokens."""
    def handler(request):
        return httpx.Response(200, content=_sliced(BODY, 7),
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    assert parse_stream_usage(collected) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert "".join(o["choices"][0]["delta"].get("content", "")
                   for o in _data_objects(collected) if o["choices"]) == "hi"


@pytest.mark.anyio
async def test_stream_survives_a_body_with_no_trailing_blank_line(monkeypatch):
    """The last event of a body that stops without its terminating blank line
    is still a real event. Here that event is message_delta, so dropping the
    residual buffer would lose output_tokens entirely."""
    def handler(request):
        return httpx.Response(200, content=_sliced(BODY_NO_TRAILING_BLANK, 7),
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    assert parse_stream_usage(collected) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}


@pytest.mark.anyio
async def test_stream_delivers_crlf_framed_events_before_the_body_ends(monkeypatch):
    """Splitting on b"\\n\\n" alone never matches a \\r\\n\\r\\n-framed stream, so
    the whole body accumulates in RAM and the client sees nothing until EOF --
    a hang, as far as an SSE client with an idle timeout is concerned."""
    slices = [CRLF_BODY[i:i + 7] for i in range(0, len(CRLF_BODY), 7)]
    produced = {"n": 0}

    async def gen():
        for part in slices:
            produced["n"] += 1
            yield part

    def handler(request):
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})

    collected = bytearray()
    first_chunk_after = None
    async for chunk in _adapter(handler, monkeypatch).chat_stream(
            "m", {"messages": [{"role": "user", "content": "hi"}]}):
        if first_chunk_after is None:
            first_chunk_after = produced["n"]
        collected.extend(chunk)

    assert first_chunk_after < len(slices), "buffered the whole body before emitting"
    assert parse_stream_usage(bytes(collected)) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert "".join(o["choices"][0]["delta"].get("content", "")
                   for o in _data_objects(bytes(collected)) if o["choices"]) == "hi"


@pytest.mark.anyio
async def test_stream_omits_usage_when_upstream_never_reported_output_tokens(monkeypatch):
    """A stream that ends cleanly before message_delta never reported an output
    count. Emitting {"completion_tokens": 0} would be taken by app.py as
    authoritative and bill zero for text the client actually received; with no
    usage chunk at all it degrades to an estimate, like the OpenAI path."""
    body = b"".join(b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
                    for e in EVENTS[:3] + [{"type": "message_stop"}])

    def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    assert parse_stream_usage(collected) is None
    objs = _data_objects(collected)
    assert not [o for o in objs if "usage" in o]
    # The stream is still well-formed: content, one finish_reason, then [DONE].
    assert "".join(o["choices"][0]["delta"].get("content", "")
                   for o in objs if o["choices"]) == "hi"
    assert len([o for o in objs
                if any(c.get("finish_reason") for c in o.get("choices", []))]) == 1
    assert collected.decode().rstrip().endswith("data: [DONE]")


@pytest.mark.anyio
async def test_stream_emits_exactly_one_finish_and_one_usage_chunk(monkeypatch):
    """StreamTranslator.finish() is not idempotent: calling it twice would emit
    two finish_reason chunks and two usage chunks. The usage chunk also has an
    EMPTY choices list, so it must not be filtered out as 'no content'."""
    def handler(request):
        return httpx.Response(200, content=BODY,
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    objs = _data_objects(collected)
    finished = [o for o in objs
                if any(c.get("finish_reason") for c in o.get("choices", []))]
    usages = [o for o in objs if o.get("usage")]
    assert len(finished) == 1 and finished[0]["choices"][0]["finish_reason"] == "stop"
    assert len(usages) == 1 and usages[0]["choices"] == []
    assert objs.index(usages[0]) == len(objs) - 1
    assert collected.decode().count("data: [DONE]") == 1


@pytest.mark.anyio
async def test_stream_raises_provider_error_before_the_first_byte(monkeypatch):
    def handler(request):
        return httpx.Response(503, json={"error": {"message": "down"}})

    agen = _adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ProviderError) as exc:
        async for _ in agen:
            pass
    assert exc.value.kind == "http" and exc.value.status == 503


@pytest.mark.anyio
async def test_stream_timeout_before_the_first_byte_is_a_provider_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectTimeout("no route", request=request)

    agen = _adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ProviderError) as exc:
        async for _ in agen:
            pass
    assert exc.value.kind == "timeout"


@pytest.mark.anyio
async def test_stream_failure_after_the_first_byte_is_not_a_provider_error(monkeypatch):
    """app.py treats ProviderError as 'safe to fall back to the next model'.
    Once bytes have reached the client, switching models mid-response would
    corrupt it, so a late failure must surface as an ordinary exception."""
    async def body():
        yield b"event: x\ndata: " + json.dumps(EVENTS[0]).encode() + b"\n\n"
        raise httpx.ReadTimeout("upstream died mid-stream")

    def handler(request):
        return httpx.Response(200, content=body(),
                              headers={"content-type": "text/event-stream"})

    seen = bytearray()
    with pytest.raises(httpx.ReadTimeout) as exc:
        async for chunk in _adapter(handler, monkeypatch).chat_stream(
                "m", {"messages": [{"role": "user", "content": "hi"}]}):
            seen.extend(chunk)

    assert seen, "the test must actually reach the post-first-byte branch"
    assert not isinstance(exc.value, ProviderError)


@pytest.mark.anyio
async def test_stream_error_event_before_the_first_byte_is_a_provider_error(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=(
            b'event: error\ndata: {"type":"error","error":{"message":"bad"}}\n\n'),
            headers={"content-type": "text/event-stream"})

    agen = _adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ProviderError) as exc:
        async for _ in agen:
            pass
    assert exc.value.kind == "http"


def _error_stream(prefix_events) -> bytes:
    return b"".join(b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
                    for e in prefix_events) + (
        b'event: error\ndata: {"type":"error",'
        b'"error":{"message":"overloaded"}}\n\n')


@pytest.mark.anyio
async def test_stream_error_after_the_first_byte_terminates_the_stream_cleanly(monkeypatch):
    """Returning bare on a mid-stream error gives the client an abrupt EOF with
    no finish_reason and no [DONE] -- indistinguishable from a dropped
    connection. Signal the failure, close the protocol, then stop."""
    def handler(request):
        return httpx.Response(200, content=_error_stream(EVENTS[:3]),
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    text = collected.decode()
    objs = _data_objects(collected)
    errors = [o for o in objs if o.get("error")]
    assert [o["error"]["type"] for o in errors] == ["upstream_error"]
    assert len([o for o in objs
                if any(c.get("finish_reason") for c in o.get("choices", []))]) == 1
    assert text.rstrip().endswith("data: [DONE]") and text.count("data: [DONE]") == 1
    # No output count ever arrived, so usage stays unknown rather than zero.
    assert parse_stream_usage(collected) is None


@pytest.mark.anyio
async def test_stream_error_after_a_complete_usage_report_keeps_the_real_counts(monkeypatch):
    """When the upstream did report both counts before failing, those banked
    numbers must survive into the terminating usage chunk -- the error object
    carries no `usage` key, so it cannot displace them in parse_stream_usage."""
    def handler(request):
        return httpx.Response(200, content=_error_stream(EVENTS[:4]),
                              headers={"content-type": "text/event-stream"})

    collected = await _collect(_adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]}))

    assert parse_stream_usage(collected) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert collected.decode().rstrip().endswith("data: [DONE]")

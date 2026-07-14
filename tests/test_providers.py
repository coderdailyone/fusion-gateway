import httpx, json, pytest
from gateway.config import ProviderCfg
from gateway.providers import ProviderAdapter, ProviderError, parse_stream_usage

CFG = ProviderCfg("deepseek", "https://api.deepseek.com", "TEST_KEY")

def make(transport):
    return ProviderAdapter(CFG, transport=transport)

@pytest.fixture(autouse=True)
def key(monkeypatch): monkeypatch.setenv("TEST_KEY", "sk-test")

@pytest.mark.anyio
async def test_chat_success_and_auth_header():
    seen = {}
    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
    out = await make(httpx.MockTransport(handler)).chat("deepseek-chat", {"messages": []})
    assert out["usage"]["completion_tokens"] == 1
    assert seen["auth"] == "Bearer sk-test" and seen["body"]["model"] == "deepseek-chat"

@pytest.mark.anyio
async def test_http_error_raises():
    t = httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(ProviderError) as e:
        await make(t).chat("m", {"messages": []})
    assert e.value.kind == "http" and e.value.status == 500

def test_parse_stream_usage():
    sse = (b'data: {"choices":[{"delta":{"content":"h"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
           b'data: [DONE]\n\n')
    assert parse_stream_usage(sse)["completion_tokens"] == 2

def test_parse_stream_usage_none_when_absent():
    sse = b'data: {"choices":[{"delta":{"content":"h"}}]}\n\ndata: [DONE]\n\n'
    assert parse_stream_usage(sse) is None

@pytest.mark.anyio
async def test_timeout_raises_timeout_kind():
    def handler(req):
        raise httpx.ReadTimeout("timed out", request=req)
    t = httpx.MockTransport(handler)
    with pytest.raises(ProviderError) as e:
        await make(t).chat("m", {"messages": []})
    assert e.value.kind == "timeout" and e.value.provider == "deepseek" and e.value.status is None

@pytest.mark.anyio
async def test_network_error_raises_network_kind():
    def handler(req):
        raise httpx.ConnectError("connection refused", request=req)
    t = httpx.MockTransport(handler)
    with pytest.raises(ProviderError) as e:
        await make(t).chat("m", {"messages": []})
    assert e.value.kind == "network" and e.value.status is None

@pytest.mark.anyio
async def test_chat_stream_sets_stream_options_and_yields_raw_bytes():
    seen = {}
    sse_body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse_body,
                               headers={"content-type": "text/event-stream"})
    chunks = []
    async for chunk in make(httpx.MockTransport(handler)).chat_stream("m", {"messages": []}):
        chunks.append(chunk)
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert b"".join(chunks) == sse_body

@pytest.mark.anyio
async def test_chat_stream_http_error_before_first_byte_raises():
    t = httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "boom"}))
    async def drain():
        async for _ in make(t).chat_stream("m", {"messages": []}):
            pass
    with pytest.raises(ProviderError) as e:
        await drain()
    assert e.value.kind == "http" and e.value.status == 500

@pytest.mark.anyio
async def test_chat_stream_timeout_before_first_byte_raises():
    def handler(req):
        raise httpx.ReadTimeout("timed out", request=req)
    t = httpx.MockTransport(handler)
    async def drain():
        async for _ in make(t).chat_stream("m", {"messages": []}):
            pass
    with pytest.raises(ProviderError) as e:
        await drain()
    assert e.value.kind == "timeout"

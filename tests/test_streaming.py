import httpx, pytest
from tests.test_app import make_client, H

SSE = (b'data: {"choices":[{"delta":{"content":"h"}}]}\n\n'
       b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
       b'data: [DONE]\n\n')

def stream_handler(req):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)

BODY = {"model": "auto", "stream": True, "messages": [{"role":"user","content":"hi"}]}

def test_stream_passthrough_and_settle(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=stream_handler)
    with c.stream("POST", "/v1/chat/completions", json=BODY, headers=H()) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_raw())
    assert b"[DONE]" in raw
    st = c.get("/admin/status", headers=H("tokB")).json()
    assert st["ledger"]["consumed_usd"] > 0

def test_prefirstbyte_failure_falls_back(tmp_path, monkeypatch):
    def boom(req): return httpx.Response(500, json={"e": 1})
    c = make_client(tmp_path, monkeypatch, deepseek=boom, glm=stream_handler)
    with c.stream("POST", "/v1/chat/completions", json=BODY, headers=H()) as r:
        raw = b"".join(r.iter_raw())
    assert b"[DONE]" in raw

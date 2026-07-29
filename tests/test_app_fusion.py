import httpx, json, pytest
from fastapi.testclient import TestClient
from gateway.app import create_app
from tests.helpers import FakeClock

CFG = """
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://example.invalid"
api_key_env = "P_KEY"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[models."b"]
provider = "p"
upstream_model = "b"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 128
stage_timeout_s = 5
[policy]
version = "static-v0"
default_model = "fusion"
"""


def handler(req):
    body = json.loads(req.content)
    prompt = body["messages"][-1]["content"]
    if "VERDICT" in prompt:
        text = "VERDICT a correct ok\nVERDICT b correct ok"
    elif "Produce the single best final answer" in prompt:
        text = "FUSED ANSWER"
    else:
        text = "candidate answer"
    if body.get("stream"):
        chunk = {"choices": [{"index": 0, "delta": {"content": text},
                              "finish_reason": None}]}
        payload = (f"data: {json.dumps(chunk)}\n\n"
                   'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'
                   "data: [DONE]\n\n")
        return httpx.Response(200, content=payload.encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}})


def make_client(tmp_path, monkeypatch, h=handler):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    p = tmp_path / "g.toml"
    p.write_text(CFG)
    app = create_app(p, tmp_path / "g.sqlite", clock=FakeClock(),
                     transports={"p": httpx.MockTransport(h)})
    return TestClient(app)


def H(tok="tokA"):
    return {"Authorization": f"Bearer {tok}"}


BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


def test_auto_takes_the_fusion_path(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fusion"
    assert body["choices"][0]["message"]["content"] == "FUSED ANSWER"
    assert body["fusion"]["path"] == "quorum"
    assert "x-fusion-trace-id" in r.headers


def test_naming_the_pseudo_model_explicitly_also_fuses(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "model": "fusion"}, headers=H())
    assert r.json()["fusion"]["path"] == "quorum"


def test_naming_a_real_model_takes_the_single_model_path(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "model": "a"}, headers=H())
    assert r.status_code == 200 and r.json()["model"] == "a"
    assert "fusion" not in r.json()


def test_fusion_writes_one_ledger_row_per_call_under_one_request_id(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    rid = r.headers["x-fusion-trace-id"]
    st = c.get("/admin/status", headers=H("tokB")).json()
    assert st["ledger"]["consumed_usd"] > 0
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger WHERE request_id=?",
                        (rid,)).fetchall()
    # 2 candidates + 2 reviews + 1 fuser
    assert len(rows) == 5
    assert not any(state == "preflight" for _, state in rows)


def test_all_upstreams_down_returns_502_not_500(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, h=lambda req: httpx.Response(500, json={}))
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_exhausted"


def test_a_lone_survivor_is_returned_verbatim(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        if body["model"] == "a":
            return httpx.Response(500, json={})
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})     # fuser also down
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "only b"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "only b"
    assert r.json()["fusion"]["degraded"] is True


def test_a_dead_fuser_falls_back_to_the_best_candidate(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"answer from {body['model']}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "answer from a"
    assert r.json()["fusion"]["degraded"] is True


def test_streaming_fusion_emits_keepalives_then_the_fuser_stream(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion" in raw                    # SSE comment keepalive
    assert "FUSED ANSWER" in raw
    assert raw.rstrip().endswith("data: [DONE]")
    # Every keepalive must be an SSE COMMENT, not a data line -- a data line
    # the client cannot parse as a chunk would break a conformant SDK.
    for line in raw.splitlines():
        if line.startswith(": "):
            continue
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            json.loads(line[6:])       # must be valid JSON, or this raises

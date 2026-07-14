import httpx, json, pytest
from fastapi.testclient import TestClient
from gateway.app import create_app
from tests.helpers import FakeClock

def ok_handler(req):
    return httpx.Response(200, json={"choices":[{"message":{"content":"hi"}}],
                                     "usage":{"prompt_tokens":5,"completion_tokens":2}})

def boom_handler(req):
    return httpx.Response(500, json={"error":"boom"})

def make_client(tmp_path, monkeypatch, deepseek=ok_handler, glm=ok_handler):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-d"); monkeypatch.setenv("GLM_API_KEY", "sk-g")
    app = create_app("configs/gateway.toml", tmp_path / "g.sqlite", clock=FakeClock(),
                     transports={"deepseek": httpx.MockTransport(deepseek),
                                 "glm": httpx.MockTransport(glm)})
    return TestClient(app)

def H(tok="tokA"): return {"Authorization": f"Bearer {tok}"}
BODY = {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}

def test_auth_required(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401

def test_happy_path_settles_ledger(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200 and r.json()["model"] == "deepseek-chat"
    assert "x-fusion-trace-id" in r.headers
    st = c.get("/admin/status", headers=H("tokB")).json()
    assert st["ledger"]["consumed_usd"] > 0 and st["requests"]["succeeded"] == 1

def test_fallback_chain_on_provider_error(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=boom_handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200 and r.json()["model"] == "glm-4.7"

def test_all_providers_down_502(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=boom_handler, glm=boom_handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_exhausted"

def test_admin_endpoints_gated(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.get("/admin/status", headers=H("tokA")).status_code == 403
    assert c.post("/admin/killswitch/release", headers=H("tokB")).status_code == 200

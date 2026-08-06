import httpx, json, pytest
from fastapi.testclient import TestClient
import gateway.app as app_module
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
BODY = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hello"}]}

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

def test_auto_now_takes_the_single_model_path(tmp_path, monkeypatch):
    # Fusion reverted to opt-in: policy.default_model is deepseek-chat again,
    # so a request naming "auto" (or no model) no longer gets fused.
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "model": "auto"}, headers=H())
    assert r.status_code == 200
    assert r.json()["model"] == "deepseek-chat"
    assert "fusion" not in r.json()


def test_v1_models_lists_the_fusion_pseudo_model_when_configured(tmp_path, monkeypatch):
    # configs/gateway.toml keeps [fusion] configured even though it is no
    # longer the default -- a client can only reach it by name, so it must
    # be discoverable here.
    c = make_client(tmp_path, monkeypatch)
    r = c.get("/v1/models", headers=H())
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "fusion" in ids
    assert "deepseek-chat" in ids   # ordinary [models] entries still listed


NO_FUSION_CFG = """
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://example.invalid"
api_key_env = "P_KEY"
[models."m1"]
provider = "p"
upstream_model = "m1"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[policy]
version = "test-v0"
default_model = "m1"
"""


def test_v1_models_omits_fusion_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    p = tmp_path / "g.toml"
    p.write_text(NO_FUSION_CFG)
    app = create_app(p, tmp_path / "g.sqlite", clock=FakeClock())
    c = TestClient(app)
    r = c.get("/v1/models", headers=H())
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["m1"]
    assert "fusion" not in ids


def test_fallback_chain_on_provider_error(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=boom_handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200 and r.json()["model"] == "glm-4.5-flash"

def test_all_providers_down_502(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=boom_handler, glm=boom_handler)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_exhausted"

def test_admin_endpoints_gated(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.get("/admin/status", headers=H("tokA")).status_code == 403
    assert c.post("/admin/killswitch/release", headers=H("tokB")).status_code == 200

def test_create_app_from_env_reads_defaults(monkeypatch):
    captured = {}
    def fake_create_app(config_path, db_path):
        captured["config_path"] = config_path
        captured["db_path"] = db_path
        return "app-sentinel"
    monkeypatch.delenv("GATEWAY_CONFIG", raising=False)
    monkeypatch.delenv("GATEWAY_DB", raising=False)
    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    result = app_module.create_app_from_env()
    assert result == "app-sentinel"
    assert captured == {"config_path": "configs/gateway.toml", "db_path": "data/gateway.sqlite"}

def test_create_app_from_env_reads_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("GATEWAY_CONFIG", "configs/gateway.toml")
    monkeypatch.setenv("GATEWAY_DB", str(tmp_path / "g.sqlite"))
    app = app_module.create_app_from_env()
    client = TestClient(app)
    # This test is about the env overrides being honoured, not about the shape
    # of the health body -- /healthz also reports which config is loaded (see
    # test_admin_panel.py), so pinning the whole dict here would make every
    # future field an unrelated failure.
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["config_path"] == "configs/gateway.toml"

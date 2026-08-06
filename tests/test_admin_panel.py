"""/healthz identity and /admin/panel health -- the two things whose absence
made 2026-08-05's wasted benchmark runs invisible while they happened."""
import json
import httpx
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
[models."b"]
provider = "p"
upstream_model = "b"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
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

BODY = {"model": "fusion", "messages": [{"role": "user", "content": "hi"}]}


def ok_handler(req):
    body = json.loads(req.content)
    prompt = body["messages"][-1]["content"]
    text = ("VERDICT a correct ok\nVERDICT b correct ok" if "VERDICT" in prompt
            else "answer")
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": 3, "completion_tokens": 4}})


def make(tmp_path, monkeypatch, h=ok_handler):
    monkeypatch.setenv("GATEWAY_TOKENS", "u:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "g.toml"
    p.write_text(CFG)
    return TestClient(create_app(p, tmp_path / "g.sqlite", clock=FakeClock(),
                                 transports={"p": httpx.MockTransport(h)})), p


def test_healthz_identifies_the_config_actually_loaded(tmp_path, monkeypatch):
    """A stale process answering /healthz with {"ok": true} is
    indistinguishable from a fresh one -- which is how a gateway started before
    two deploys served two whole benchmark runs unnoticed."""
    import hashlib
    c, p = make(tmp_path, monkeypatch)
    body = c.get("/healthz").json()
    assert body["ok"] is True
    assert body["config_sha"] == hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    assert body["fusion_panel"] == ["a", "b"]
    assert body["started_at"]


def test_healthz_sha_changes_when_the_config_does(tmp_path, monkeypatch):
    """The whole point: the digest must track content, or a mismatch against
    the file on disk proves nothing."""
    c1, p1 = make(tmp_path / "one", monkeypatch)
    tmp2 = tmp_path / "two"
    tmp2.mkdir()
    monkeypatch.setenv("GATEWAY_TOKENS", "u:tokA,admin:tokB")
    p2 = tmp2 / "g.toml"
    p2.write_text(CFG.replace('fuser = "b"', 'fuser = "a"'))
    c2 = TestClient(create_app(p2, tmp2 / "g.sqlite", clock=FakeClock(),
                               transports={"p": httpx.MockTransport(ok_handler)}))
    assert c1.get("/healthz").json()["config_sha"] != \
        c2.get("/healthz").json()["config_sha"]


def test_panel_reports_every_member_healthy_on_a_good_run(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch)
    c.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer tokA"})
    d = c.get("/admin/panel", headers={"Authorization": "Bearer tokB"}).json()
    assert d["healthy"] is True and d["unhealthy"] == []
    assert d["members"]["a"]["ok"] >= 1 and d["members"]["b"]["ok"] >= 1
    assert d["paths"]


def test_panel_names_the_member_that_is_failing_and_why(tmp_path, monkeypatch):
    """The exact 2026-08-05 shape: one member 400s on every candidate call, the
    request still succeeds on the survivor, and nothing else would say so."""
    def h(req):
        body = json.loads(req.content)
        if body.get("model") == "a":
            return httpx.Response(400, json={"error": {
                "message": "invalid temperature: only 1 is allowed for this model"}})
        return ok_handler(req)

    c, _ = make(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY,
               headers={"Authorization": "Bearer tokA"})
    assert r.status_code == 200, "the request still succeeds -- that is the trap"

    d = c.get("/admin/panel", headers={"Authorization": "Bearer tokB"}).json()
    assert d["healthy"] is False
    assert d["unhealthy"] == ["a"]
    assert d["members"]["a"]["ok_rate"] == 0.0
    assert d["members"]["b"]["ok_rate"] == 1.0
    err = d["members"]["a"]["last_error"]
    assert err["status"] == 400
    assert "only 1 is allowed" in err["body"], "the body is the actionable part"


def test_panel_requires_admin(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch)
    assert c.get("/admin/panel", headers={"Authorization": "Bearer tokA"}).status_code == 403


def test_panel_is_quiet_when_no_fusion_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKENS", "u:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    p = tmp_path / "g.toml"
    p.write_text(CFG.split("[fusion]")[0] +
                 '[policy]\nversion = "v"\ndefault_model = "a"\n')
    c = TestClient(create_app(p, tmp_path / "g.sqlite", clock=FakeClock(),
                              transports={"p": httpx.MockTransport(ok_handler)}))
    assert c.get("/admin/panel", headers={"Authorization": "Bearer tokB"}).json() == {"fusion": None}

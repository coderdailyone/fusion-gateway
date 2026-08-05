"""Per-model request-parameter constraints.

A fusion panel is heterogeneous, but a client sends ONE body. Forwarding it
verbatim to every member assumes they accept the same parameters -- they do
not, and when they do not the member simply vanishes from the panel with
nothing in the response to say so. Measured on the M4 agentic run: kimi-k3
400'd on 877 of 881 candidate calls over `temperature`, deepseek-v4-flash on
840 over thinking-mode `reasoning_content`, leaving glm-5.2 answering alone
while the response still called itself a fusion.
"""
import pytest
from gateway.config import ModelCfg, load_config, ConfigError

BASE = dict(name="m", provider="p", upstream_model="m",
            in_usd_per_mtok=1.0, out_usd_per_mtok=1.0, fallback=())


def test_a_model_with_no_constraints_passes_the_body_through_untouched():
    cfg = ModelCfg(**BASE)
    body = {"messages": [], "temperature": 0.0}
    assert cfg.apply_params(body) is body, "no copy when there is nothing to do"


def test_an_override_forces_the_models_legal_value():
    cfg = ModelCfg(**BASE, param_overrides={"temperature": 1})
    assert cfg.apply_params({"temperature": 0.0})["temperature"] == 1


def test_an_override_is_added_even_when_the_client_never_sent_the_key():
    """deepseek needs thinking switched OFF; a client that never mentions
    thinking still gets the override, or the member 400s from turn two."""
    cfg = ModelCfg(**BASE, param_overrides={"thinking": {"type": "disabled"}})
    assert cfg.apply_params({"messages": []})["thinking"] == {"type": "disabled"}


def test_drop_removes_a_key_the_upstream_must_not_see():
    cfg = ModelCfg(**BASE, drop_params=("temperature",))
    assert "temperature" not in cfg.apply_params({"temperature": 0.5, "top_p": 1})


def test_the_clients_body_is_never_mutated():
    """The SAME dict is handed to every panel member in turn. Mutating it would
    let the first member's constraints leak into the second's request -- kimi's
    forced temperature=1 would silently become glm's temperature too."""
    cfg = ModelCfg(**BASE, param_overrides={"temperature": 1})
    body = {"temperature": 0.0}
    cfg.apply_params(body)
    assert body["temperature"] == 0.0


def test_other_parameters_survive_an_override():
    cfg = ModelCfg(**BASE, param_overrides={"temperature": 1})
    out = cfg.apply_params({"temperature": 0.0, "max_tokens": 99, "tools": ["t"]})
    assert out["max_tokens"] == 99 and out["tools"] == ["t"]


def test_config_parses_both_fields(tmp_path):
    p = tmp_path / "g.toml"
    p.write_text('''
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://x.invalid"
api_key_env = "K"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
param_overrides = { temperature = 1, thinking = { type = "disabled" } }
drop_params = ["top_p"]
[policy]
version = "v"
default_model = "a"
''')
    m = load_config(p).models["a"]
    assert m.param_overrides == {"temperature": 1, "thinking": {"type": "disabled"}}
    assert m.drop_params == ("top_p",)
    assert m.apply_params({"temperature": 0.0, "top_p": 0.9}) == {
        "temperature": 1, "thinking": {"type": "disabled"}}


def test_config_defaults_are_empty(tmp_path):
    p = tmp_path / "g.toml"
    p.write_text('''
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://x.invalid"
api_key_env = "K"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
[policy]
version = "v"
default_model = "a"
''')
    m = load_config(p).models["a"]
    assert m.param_overrides == {} and m.drop_params == ()


def test_the_shipped_config_pins_the_two_measured_constraints():
    """These are not cosmetic: without them two of three panel members drop out
    of every multi-turn conversation, and the gateway still reports success."""
    cfg = load_config("configs/gateway.toml")
    assert cfg.models["kimi-k3"].param_overrides == {"temperature": 1}
    assert cfg.models["deepseek-chat"].param_overrides == {
        "thinking": {"type": "disabled"}}

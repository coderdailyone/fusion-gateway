from pathlib import Path
import pytest
from gateway.config import load_config, ConfigError

GOOD = Path("configs/gateway.toml")

def test_loads_models_and_providers():
    cfg = load_config(GOOD)
    assert cfg.models["deepseek-chat"].fallback == ("glm-4.5-flash",)
    assert cfg.providers["glm"].api_key_env == "GLM_API_KEY"
    assert cfg.default_model in cfg.models
    assert cfg.budget_caps[cfg.active_budget] == 5.0

def test_unknown_fallback_rejected(tmp_path):
    bad = GOOD.read_text().replace('fallback = ["glm-4.5-flash"]', 'fallback = ["nope"]')
    p = tmp_path / "g.toml"; p.write_text(bad)
    with pytest.raises(ConfigError):
        load_config(p)

_BASE = """
[budget]
active = "T"
[budgets.T]
cap_usd = 1.0
[providers.p_openai]
base_url = "https://example.test/v1"
api_key_env = "X_KEY"
[providers.p_anthropic]
base_url = "https://example.test/anthropic"
api_key_env = "X_KEY"
wire = "anthropic"
[models."m1"]
provider = "p_openai"
upstream_model = "u1"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 2.0
[policy]
version = "test-v0"
default_model = "m1"
"""


def test_wire_defaults_to_openai_and_is_read_when_given(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(_BASE)
    cfg = load_config(p)
    assert cfg.providers["p_openai"].wire == "openai"      # default
    assert cfg.providers["p_anthropic"].wire == "anthropic"  # explicit


def test_unknown_wire_is_a_config_error(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(_BASE.replace('wire = "anthropic"', 'wire = "carrier-pigeon"'))
    with pytest.raises(ConfigError):
        load_config(p)

from pathlib import Path
import pytest
from gateway.config import load_config, ConfigError

GOOD = Path("configs/gateway.toml")

def test_loads_models_and_providers():
    cfg = load_config(GOOD)
    assert cfg.models["deepseek-chat"].fallback == ("glm-4.6",)
    assert cfg.providers["glm"].api_key_env == "GLM_API_KEY"
    assert cfg.default_model in cfg.models
    assert cfg.budget_caps[cfg.active_budget] == 5.0

def test_unknown_fallback_rejected(tmp_path):
    bad = GOOD.read_text().replace('fallback = ["glm-4.6"]', 'fallback = ["nope"]')
    p = tmp_path / "g.toml"; p.write_text(bad)
    with pytest.raises(ConfigError):
        load_config(p)

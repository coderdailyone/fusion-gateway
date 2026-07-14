import pytest
from pathlib import Path
from gateway.config import load_config
from gateway.policy import plan_route, UnknownModel

CFG = load_config(Path("configs/gateway.toml"))

def test_auto_routes_to_default_with_fallback():
    rp = plan_route(CFG, "auto")
    assert rp.chain == ("deepseek-chat", "glm-4.6")
    assert rp.policy_version == "static-v0"

def test_explicit_model_and_unknown():
    assert plan_route(CFG, "glm-4.6").chain == ("glm-4.6",)
    with pytest.raises(UnknownModel):
        plan_route(CFG, "gpt-999")

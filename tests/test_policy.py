import pytest
from pathlib import Path
from gateway.config import load_config
from gateway.policy import plan_route, UnknownModel

CFG = load_config(Path("configs/gateway.toml"))

def test_explicit_default_chain_still_routes():
    rp = plan_route(CFG, "deepseek-chat")
    assert rp.chain == ("deepseek-chat", "glm-4.5-flash")
    assert rp.policy_version == "static-v0"

def test_auto_now_resolves_to_the_fusion_pseudo_model():
    # app.py intercepts this before plan_route; plan_route itself has no
    # route for a pseudo-model, and saying so is correct.
    assert CFG.default_model == "fusion"
    with pytest.raises(UnknownModel):
        plan_route(CFG, "fusion")

def test_explicit_model_and_unknown():
    assert plan_route(CFG, "glm-4.5-flash").chain == ("glm-4.5-flash",)
    with pytest.raises(UnknownModel):
        plan_route(CFG, "gpt-999")

import pytest
from pathlib import Path
from gateway.config import load_config
from gateway.policy import plan_route, UnknownModel

CFG = load_config(Path("configs/gateway.toml"))

def test_explicit_default_chain_still_routes():
    rp = plan_route(CFG, "deepseek-chat")
    assert rp.chain == ("deepseek-chat", "glm-4.5-flash")
    assert rp.policy_version == "static-v0"

def test_auto_resolves_to_deepseek_chat_again_fusion_is_opt_in():
    # Fusion reverted to opt-in: default_model names a real model again, so
    # "auto"/"" take the single-model path. [fusion] is still configured
    # (selectable by naming it explicitly), and plan_route itself still has
    # no route for the pseudo-model -- app.py intercepts "fusion" before
    # ever calling plan_route, so this remains correct to assert directly.
    assert CFG.default_model == "deepseek-chat"
    assert CFG.fusion is not None and CFG.fusion.model == "fusion"
    with pytest.raises(UnknownModel):
        plan_route(CFG, "fusion")

def test_explicit_model_and_unknown():
    assert plan_route(CFG, "glm-4.5-flash").chain == ("glm-4.5-flash",)
    with pytest.raises(UnknownModel):
        plan_route(CFG, "gpt-999")

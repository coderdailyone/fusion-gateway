from evaluator.agentic.runner import build_agent_config

REGISTRY = {
    "deepseek-chat": {"model": "deepseek/deepseek-chat",
                      "api_base": None, "api_key": "sk-x", "max_tokens": 8192},
    "claude-opus-4-8": {"model": "anthropic/claude-opus-4-8",
                        "api_base": "https://mirror/claudecode", "api_key": "sk-y",
                        "max_tokens": 8192},
}


def test_build_agent_config_maps_registry_to_litellm():
    cfg = build_agent_config("claude-opus-4-8", REGISTRY)
    assert cfg["model"] == "anthropic/claude-opus-4-8"
    assert cfg["api_base"] == "https://mirror/claudecode"
    assert cfg["api_key"] == "sk-y"
    assert cfg["max_tokens"] == 8192


def test_build_agent_config_unknown_model_raises():
    import pytest
    with pytest.raises(KeyError):
        build_agent_config("nope", REGISTRY)

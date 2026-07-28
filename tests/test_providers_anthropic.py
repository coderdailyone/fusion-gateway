from gateway.config import ProviderCfg
from gateway.providers import ProviderAdapter, make_adapter
from gateway.providers_anthropic import AnthropicAdapter


def test_factory_picks_the_adapter_from_the_wire_field():
    openai_cfg = ProviderCfg("p", "https://example.test/v1", "X_KEY")
    anthropic_cfg = ProviderCfg("q", "https://example.test/anthropic", "X_KEY",
                                wire="anthropic")
    assert isinstance(make_adapter(openai_cfg), ProviderAdapter)
    assert isinstance(make_adapter(anthropic_cfg), AnthropicAdapter)

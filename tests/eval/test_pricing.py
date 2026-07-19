import pytest
from evaluator.pricing import load_prices, cost

def test_load_and_cost(tmp_path):
    p = tmp_path / "pricing.toml"
    p.write_text('["m"]\nin_usd_per_mtok = 1.0\nout_usd_per_mtok = 2.0\n')
    prices = load_prices(p)
    assert prices["m"] == (1.0, 2.0)
    # 1e6 in @ $1 + 1e6 out @ $2 = $3
    assert cost("m", 1_000_000, 1_000_000, prices) == pytest.approx(3.0)

def test_unknown_model_raises(tmp_path):
    p = tmp_path / "pricing.toml"
    p.write_text('["m"]\nin_usd_per_mtok = 1.0\nout_usd_per_mtok = 2.0\n')
    with pytest.raises(KeyError):
        cost("nope", 10, 10, load_prices(p))

def test_default_config_has_the_three_models():
    prices = load_prices()
    assert {"deepseek-chat", "kimi-k3", "glm-5.2"} <= set(prices)

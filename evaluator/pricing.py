from __future__ import annotations
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib


def load_prices(path: str | Path = "configs/pricing.toml") -> dict[str, tuple[float, float]]:
    """
    Load pricing data from a TOML file.

    Args:
        path: Path to the pricing.toml file (default: configs/pricing.toml)

    Returns:
        Dictionary mapping model names to (in_usd_per_mtok, out_usd_per_mtok) tuples
    """
    data = tomllib.loads(Path(path).read_text())
    prices = {}
    for model_name, model_data in data.items():
        prices[model_name] = (
            float(model_data["in_usd_per_mtok"]),
            float(model_data["out_usd_per_mtok"])
        )
    return prices


def cost(model: str, in_tokens: int, out_tokens: int, prices: dict[str, tuple[float, float]] | None = None) -> float:
    """
    Calculate the cost of a model query.

    Args:
        model: Model name
        in_tokens: Number of input tokens
        out_tokens: Number of output tokens
        prices: Dictionary of prices (if None, loads from default config)

    Returns:
        Cost in USD

    Raises:
        KeyError: If the model is not found in prices
    """
    if prices is None:
        prices = load_prices()

    if model not in prices:
        raise KeyError(f"Model {model} not found in prices")

    in_price, out_price = prices[model]
    return in_tokens * in_price / 1e6 + out_tokens * out_price / 1e6

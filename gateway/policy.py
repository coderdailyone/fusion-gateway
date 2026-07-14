from __future__ import annotations
from dataclasses import dataclass
from gateway.config import Config


class UnknownModel(Exception):
    pass


@dataclass(frozen=True)
class RoutePlan:
    policy_version: str
    chain: tuple[str, ...]


def plan_route(cfg: Config, requested_model: str) -> RoutePlan:
    """
    Plan a route for the requested model.

    If requested_model is "" or "auto", use the default model.
    If the resolved model is not in cfg.models, raise UnknownModel.
    The chain is the primary model followed by its configured fallbacks.
    """
    # Resolve model name
    model_name = cfg.default_model if requested_model in ("", "auto") else requested_model

    # Check if model exists
    if model_name not in cfg.models:
        raise UnknownModel(f"Model '{model_name}' not found in config")

    # Get the model and build chain with fallbacks
    model_cfg = cfg.models[model_name]
    chain = (model_name,) + model_cfg.fallback

    return RoutePlan(policy_version=cfg.policy_version, chain=chain)

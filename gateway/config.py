from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib

class ConfigError(Exception): pass

@dataclass(frozen=True)
class ProviderCfg:
    name: str; base_url: str; api_key_env: str; wire: str = "openai"

@dataclass(frozen=True)
class ModelCfg:
    name: str; provider: str; upstream_model: str
    in_usd_per_mtok: float; out_usd_per_mtok: float
    fallback: tuple[str, ...]

@dataclass(frozen=True)
class FusionCfg:
    model: str                      # the pseudo-model clients request
    panel: tuple[str, ...]          # every candidate, in preference order
    quorum: tuple[str, ...]         # agreement here short-circuits the slow leg
    reviewers: tuple[str, ...]      # who cross-reviews (may exclude slow models)
    fuser: str                      # writes the final answer
    review_max_tokens: int
    stage_timeout_s: float

@dataclass(frozen=True)
class Config:
    providers: dict[str, ProviderCfg]; models: dict[str, ModelCfg]
    policy_version: str; default_model: str
    active_budget: str; budget_caps: dict[str, float | None]   # None == no cap
    fusion: "FusionCfg | None" = None

def load_config(path: Path) -> Config:
    data = tomllib.loads(Path(path).read_text())
    providers = {}
    for n, p in data["providers"].items():
        wire = p.get("wire", "openai")
        if wire not in ("openai", "anthropic"):
            raise ConfigError(f"provider {n}: unknown wire {wire!r}")
        providers[n] = ProviderCfg(n, p["base_url"], p["api_key_env"], wire)
    models: dict[str, ModelCfg] = {}
    for n, m in data["models"].items():
        if m["provider"] not in providers:
            raise ConfigError(f"model {n}: unknown provider {m['provider']}")
        models[n] = ModelCfg(n, m["provider"], m["upstream_model"],
                             float(m["in_usd_per_mtok"]), float(m["out_usd_per_mtok"]),
                             tuple(m.get("fallback", [])))
    for n, m in models.items():
        for f in m.fallback:
            if f not in models:
                raise ConfigError(f"model {n}: unknown fallback {f}")

    fusion = None
    if "fusion" in data:
        f = data["fusion"]
        name = f["model"]
        if name in models:
            raise ConfigError(
                f"fusion.model {name!r} collides with a real model; the fusion "
                "pseudo-model must not shadow one"
            )
        panel = tuple(f["panel"])
        quorum = tuple(f["quorum"])
        reviewers = tuple(f["reviewers"])
        fuser = f["fuser"]
        if len(panel) < 2:
            raise ConfigError("fusion.panel needs at least 2 models")
        for field, names in (("panel", panel), ("quorum", quorum),
                             ("reviewers", reviewers), ("fuser", (fuser,))):
            for n in names:
                if n not in models:
                    raise ConfigError(f"fusion.{field} names unknown model {n!r}")
        if not set(quorum) <= set(panel):
            raise ConfigError("fusion.quorum must be a subset of fusion.panel")
        fusion = FusionCfg(
            model=name, panel=panel, quorum=quorum, reviewers=reviewers,
            fuser=fuser,
            review_max_tokens=int(f.get("review_max_tokens", 512)),
            stage_timeout_s=float(f.get("stage_timeout_s", 120)),
        )

    pol = data["policy"]
    # The fusion pseudo-model is deliberately absent from [models], so it is a
    # legitimate default even though it resolves to no ModelCfg.
    if pol["default_model"] not in models and not (
        fusion is not None and pol["default_model"] == fusion.model
    ):
        raise ConfigError("policy.default_model not in models")
    # cap_usd omitted == no ceiling. Kept explicit rather than defaulted to a
    # number, so "unbounded" is something an operator writes on purpose.
    caps = {
        k: (float(v["cap_usd"]) if "cap_usd" in v else None)
        for k, v in data["budgets"].items()
    }
    if data["budget"]["active"] not in caps:
        raise ConfigError("active budget has no [budgets.<name>] table")
    return Config(providers, models, pol["version"], pol["default_model"],
                  data["budget"]["active"], caps, fusion)

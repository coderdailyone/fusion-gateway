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
class Config:
    providers: dict[str, ProviderCfg]; models: dict[str, ModelCfg]
    policy_version: str; default_model: str
    active_budget: str; budget_caps: dict[str, float | None]   # None == no cap

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
    pol = data["policy"]
    if pol["default_model"] not in models:
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
                  data["budget"]["active"], caps)

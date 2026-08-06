from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib

class ConfigError(Exception): pass


def _has_assistant_turn(body: dict) -> bool:
    """Is this a continuation rather than the start of a conversation?

    The test is the presence of an assistant turn, not a message count: a
    system + user pair is still a first turn, and it is the assistant turn --
    the one a provider may demand extra state for -- that makes a request a
    continuation. Tolerant of a malformed body, because deciding a request's
    parameters must never be what raises on one.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "assistant" for m in msgs)

@dataclass(frozen=True)
class ProviderCfg:
    name: str; base_url: str; api_key_env: str; wire: str = "openai"

@dataclass(frozen=True)
class ModelCfg:
    name: str; provider: str; upstream_model: str
    in_usd_per_mtok: float; out_usd_per_mtok: float
    fallback: tuple[str, ...]
    # Request parameters this model constrains, applied to every call routed to
    # it. A fusion panel is HETEROGENEOUS but the client sends one body, and
    # forwarding it verbatim to every member assumes they accept the same
    # parameters. They do not: measured on the M4 agentic run, kimi-k3 rejected
    # SWE-agent's temperature with
    #     400 invalid temperature: only 1 is allowed for this model
    # on 877 of 881 candidate calls, and deepseek-v4-flash rejected the
    # multi-turn conversation with
    #     400 The `reasoning_content` in the thinking mode must be passed back
    # on 840. Two of three members were gone from turn two onward, so what ran
    # was glm-5.2 alone -- and nothing in the response said so.
    #
    # `param_overrides` FORCES a value; `drop_params` REMOVES a key. Overriding
    # is preferred where the upstream names a legal value (temperature=1),
    # dropping where the parameter simply must not appear.
    param_overrides: dict = field(default_factory=dict)
    drop_params: tuple[str, ...] = ()
    # Applied ONLY when the request already carries an assistant turn, i.e.
    # when the model is being asked to continue a conversation rather than
    # start one. Some constraints exist only in that case, and applying them
    # to every request pays their cost for nothing.
    #
    # deepseek-v4-flash is the motivating example. It answers in thinking mode
    # by default, and a thinking-mode reply must have its `reasoning_content`
    # handed back on the NEXT turn -- which a fusion gateway cannot do, since
    # the assistant turn a client sends back is the FUSED answer and belongs
    # to no single member. Switching thinking off is the only way to use it in
    # multi-turn work. But a request with no prior assistant turn has no next
    # turn to fail on, so switching thinking off there weakens the candidate
    # for no reason at all.
    param_overrides_multi_turn: dict = field(default_factory=dict)

    def apply_params(self, body: dict) -> dict:
        """Return `body` conformed to this model's constraints.

        A copy: the same dict is handed to every panel member in turn, so
        mutating it would let the first member's constraints leak into the
        second's request.
        """
        extra = (self.param_overrides_multi_turn
                 if self.param_overrides_multi_turn and _has_assistant_turn(body)
                 else {})
        if not self.param_overrides and not self.drop_params and not extra:
            return body
        out = dict(body)
        for k in self.drop_params:
            out.pop(k, None)
        out.update(self.param_overrides)
        out.update(extra)
        return out

@dataclass(frozen=True)
class FusionCfg:
    model: str                      # the pseudo-model clients request
    panel: tuple[str, ...]          # every candidate, in preference order
    quorum: tuple[str, ...]         # agreement here short-circuits the slow leg
    reviewers: tuple[str, ...]      # who cross-reviews (may exclude slow models)
    fuser: str                      # writes the final answer
    review_max_tokens: int
    stage_timeout_s: float
    readonly_tools: frozenset[str] = frozenset({"read", "ls", "grep", "find"})

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
                             tuple(m.get("fallback", [])),
                             dict(m.get("param_overrides", {})),
                             tuple(m.get("drop_params", [])),
                             dict(m.get("param_overrides_multi_turn", {})))
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
        # Validate that panel, quorum, reviewers are lists before converting
        if not isinstance(f["panel"], list):
            raise ConfigError(f"fusion.panel must be a list, got {type(f['panel']).__name__}")
        if not isinstance(f["quorum"], list):
            raise ConfigError(f"fusion.quorum must be a list, got {type(f['quorum']).__name__}")
        if not isinstance(f["reviewers"], list):
            raise ConfigError(f"fusion.reviewers must be a list, got {type(f['reviewers']).__name__}")

        panel = tuple(f["panel"])
        quorum = tuple(f["quorum"])
        reviewers = tuple(f["reviewers"])
        fuser = f["fuser"]
        if len(panel) < 2:
            raise ConfigError("fusion.panel needs at least 2 models")
        if len(set(panel)) != len(panel):
            # A duplicate collapses to one task in gather_panel's `{m:
            # asyncio.create_task(...) for m in fcfg.panel}` dict
            # comprehension, so `len(candidates) < len(fcfg.panel)` is
            # permanently true even when every distinct model answered --
            # `degraded` pins true forever for this panel.
            raise ConfigError("fusion.panel must not contain duplicates")
        if len(quorum) == 0:
            raise ConfigError("fusion.quorum must not be empty")
        if len(reviewers) == 0:
            raise ConfigError("fusion.reviewers must not be empty")
        for field, names in (("panel", panel), ("quorum", quorum),
                             ("reviewers", reviewers), ("fuser", (fuser,))):
            for n in names:
                if n not in models:
                    raise ConfigError(f"fusion.{field} names unknown model {n!r}")
        if not set(quorum) <= set(panel):
            raise ConfigError("fusion.quorum must be a subset of fusion.panel")
        if not set(reviewers) <= set(panel):
            raise ConfigError("fusion.reviewers must be a subset of fusion.panel")
        if not set(quorum) <= set(reviewers):
            # is_consensus() requires every candidate to also be a reviewer of
            # the others; a quorum member missing from reviewers can never be
            # judged, so consensus -- and the whole quorum short-circuit --
            # would be silently unreachable forever.
            raise ConfigError("fusion.quorum must be a subset of fusion.reviewers")

        review_max_tokens = int(f.get("review_max_tokens", 512))
        stage_timeout_s = float(f.get("stage_timeout_s", 120))
        if review_max_tokens <= 0:
            raise ConfigError(f"fusion.review_max_tokens must be > 0, got {review_max_tokens}")
        if stage_timeout_s <= 0:
            raise ConfigError(f"fusion.stage_timeout_s must be > 0, got {stage_timeout_s}")

        if "readonly_tools" in f:
            raw_ro = f["readonly_tools"]
            if not isinstance(raw_ro, list):
                raise ConfigError(
                    f"fusion.readonly_tools must be a list, got "
                    f"{type(raw_ro).__name__}"
                )
            for entry in raw_ro:
                if not isinstance(entry, str) or not entry:
                    raise ConfigError(
                        f"fusion.readonly_tools entries must be non-empty "
                        f"strings, got {entry!r}"
                    )
            if len(set(raw_ro)) != len(raw_ro):
                raise ConfigError("fusion.readonly_tools has duplicate entries")
            readonly_tools = frozenset(raw_ro)
        else:
            readonly_tools = frozenset({"read", "ls", "grep", "find"})

        fusion = FusionCfg(
            model=name, panel=panel, quorum=quorum, reviewers=reviewers,
            fuser=fuser,
            review_max_tokens=review_max_tokens,
            stage_timeout_s=stage_timeout_s,
            readonly_tools=readonly_tools,
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

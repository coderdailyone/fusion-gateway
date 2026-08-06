"""Register the gateway's `fusion` pseudo-model with LiteLLM.

SWE-agent prices every response with litellm.completion_cost(). For a model
litellm has never heard of that RAISES, and models.py turns the exception into
a hard ModelConfigurationError whenever a cost limit is set -- so the run dies
on its first call unless the limits are switched off entirely, which would
remove the only per-instance brake there is.

The rate below is deliberately an UPPER BOUND, not an estimate. Fusion tokens
come from three models at different prices (deepseek $0.14/$0.28, glm
$0.60/$2.20, kimi $0.60/$2.50 per Mtok), so no single rate can be correct; a
rate that over-prices makes SWE-agent's per-instance limit trip EARLIER, which
is the safe direction for a brake.

It is not the number the benchmark reports. The gateway's own ledger prices
each upstream call at that model's real rate and is the authoritative cost --
see the run's data/swe.sqlite.
"""
try:
    import litellm

    litellm.register_model({
        "fusion": {
            "input_cost_per_token": 0.60 / 1e6,    # highest input rate in the panel
            "output_cost_per_token": 2.50 / 1e6,   # highest output rate in the panel
            "max_tokens": 8192,
            "max_input_tokens": 128000,
            "max_output_tokens": 8192,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
        }
    })
except Exception:                      # never break the interpreter over this
    pass

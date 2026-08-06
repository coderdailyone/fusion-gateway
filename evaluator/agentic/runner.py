"""Drive SWE-agent on one instance with one model, returning an AgenticAttempt.

build_agent_config (pure, testable) maps our model registry to SWE-agent's
LiteLLM config. run() (Phase B) executes SWE-agent in the instance container.
"""
from __future__ import annotations


def build_agent_config(model_name: str, registry: dict) -> dict:
    """Map a registry entry to the LiteLLM config SWE-agent consumes."""
    entry = registry[model_name]  # KeyError on unknown model (intended)
    return {
        "model": entry["model"],
        "api_base": entry.get("api_base"),
        "api_key": entry["api_key"],
        "max_tokens": entry.get("max_tokens", 8192),
    }


def run(instance, model_name: str, registry: dict, work_dir, box):
    """Not implemented in-process. The rig in `scripts/agentic/` does this job.

    This was specified as "Phase B" and never built, because SWE-agent's own
    `run-batch` already does all of it -- container lifecycle, the agent loop,
    trajectory and patch capture, per-instance cost limits and concurrency --
    and reimplementing that in-process would be a second, worse copy that
    drifts from the harness the official grader expects.

    What actually runs an arm is `scripts/agentic/run_model.sh` (a model
    directly) or `run_fusion.sh` (a gateway pseudo-model), each writing
    per-instance `.pred`/`.traj` files that `build_cascade.py` /
    `build_fusion_preds.py` turn into prediction files for the official
    harness. `scripts/agentic/README.md` documents the host requirements and
    the four environment traps that make an arm fail silently.

    `build_agent_config` above is still the mapping those scripts implement,
    and stays here because it is pure and worth testing. Anything that calls
    `run()` expecting an in-process execution is looking in the wrong place --
    hence a raise rather than a stub that returns something plausible.
    """
    raise NotImplementedError(
        "agentic execution lives in scripts/agentic/ (run_model.sh / "
        "run_fusion.sh), not in-process -- see scripts/agentic/README.md"
    )

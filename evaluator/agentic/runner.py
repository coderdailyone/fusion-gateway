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
    """Run SWE-agent on `instance` with `model_name` inside its container.

    Implemented in Phase B (Task 9): starts the instance's Docker container on
    `box`, runs SWE-agent (config from build_agent_config) against the repo at
    instance.base_commit, captures the final `git diff` patch + trajectory +
    summed LiteLLM cost + step count, and returns an AgenticAttempt. On any
    failure returns status="error"/"timeout" with an empty patch.
    """
    raise NotImplementedError("run() is wired on the Docker box in Task 9")

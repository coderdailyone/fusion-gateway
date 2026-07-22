"""Frozen agentic-attempt store: one SWE-agent run's patch + trajectory + cost."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgenticAttempt:
    instance_id: str
    model: str
    patch: str            # unified git diff produced by the agent ("" if none)
    trajectory_path: str  # relative path to the saved SWE-agent trajectory
    n_steps: int          # number of agent turns
    cost_usd: float       # summed LiteLLM completion_cost over the trajectory
    status: str           # "ok" | "error" | "timeout"
    error: str | None


def append_attempt(run_dir, a: AgenticAttempt) -> None:
    path = Path(run_dir) / "attempts.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(a)) + "\n")


def read_attempts(run_dir) -> list[AgenticAttempt]:
    path = Path(run_dir) / "attempts.jsonl"
    out: list[AgenticAttempt] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    out.append(AgenticAttempt(**json.loads(line)))
    return out

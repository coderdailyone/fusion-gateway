"""Task-level escalation cascade: cheap runs the whole task; a proxy verifier
gates; escalate the whole task to the strong model only on failure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult


@dataclass(frozen=True)
class CascadeResult:
    instance_id: str
    accepted_patch: str
    escalated: bool
    cost_usd: float
    cheap: AgenticAttempt
    strong: AgenticAttempt | None
    verifier: VerifierResult


def run_cascade(
    instance,
    cheap_run: Callable[[object], AgenticAttempt],
    strong_run: Callable[[object], AgenticAttempt],
    verify: Callable[[object, str], VerifierResult],
) -> CascadeResult:
    cheap = cheap_run(instance)
    v = verify(instance, cheap.patch)
    if v.passed:
        return CascadeResult(cheap.instance_id, cheap.patch, False,
                             cheap.cost_usd, cheap, None, v)
    strong = strong_run(instance)
    return CascadeResult(cheap.instance_id, strong.patch, True,
                         cheap.cost_usd + strong.cost_usd, cheap, strong, v)

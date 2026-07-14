"""Candidate runner: executes one Task on one model via an injected completion
function, freezes the output, and guards against answer leakage into the prompt.
"""

import time
from dataclasses import dataclass
from typing import Callable

from evaluator.suite.types import Task


@dataclass(frozen=True)
class FrozenOutput:
    task_id: str
    source: str
    model: str
    prompt: str
    output_text: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    latency_ms: int
    status: str
    error: str | None


def build_prompt(task: Task) -> str:
    """Build the prompt sent to the model.

    Uses ONLY task.problem plus a fixed instruction. Must never include
    task.answer or task.tests content, to avoid leaking the answer.
    """
    return (
        "Solve the following problem. Put your final answer clearly at the end.\n\n"
        f"{task.problem}"
    )


def run_one(task: Task, model: str, completion_fn: Callable[[str, str], dict]) -> FrozenOutput:
    """Run one task on one model via the injected completion_fn.

    completion_fn(model, prompt) -> {"text", "in_tokens", "out_tokens", "cost_usd"}

    On success -> status="ok" with fields populated from completion_fn's result.
    On any exception from completion_fn -> status="error" with zeroed usage and
    the exception message captured in `error`.
    """
    prompt = build_prompt(task)
    # Defensive leakage guard: the answer must never appear in the prompt we send.
    if task.answer is not None:
        assert task.answer not in prompt, "answer leaked into prompt"

    start = time.monotonic()
    try:
        result = completion_fn(model, prompt)
        latency_ms = int((time.monotonic() - start) * 1000)
        return FrozenOutput(
            task_id=task.id,
            source=task.source,
            model=model,
            prompt=prompt,
            output_text=result["text"],
            in_tokens=result["in_tokens"],
            out_tokens=result["out_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        return FrozenOutput(
            task_id=task.id,
            source=task.source,
            model=model,
            prompt=prompt,
            output_text="",
            in_tokens=0,
            out_tokens=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )


def litellm_completion(model: str, prompt: str) -> dict:
    """Real adapter around litellm.completion. Not exercised in unit tests.

    litellm is imported lazily here (not at module scope) because it is not
    installed in this environment, and evaluator/runner.py must import fine
    without it.
    """
    import litellm

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.choices[0].message.content,
        "in_tokens": response.usage.prompt_tokens,
        "out_tokens": response.usage.completion_tokens,
        "cost_usd": litellm.completion_cost(response),
    }

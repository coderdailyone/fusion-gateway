"""Reference-answer self-test: the scorer MUST mark each dataset gold correct.

Constructs each task's own reference answer as a realistic model output and
asserts the delegated scorer returns correct. A failure is a definite scorer
bug (it cannot recognize the benchmark's own correct answer). Offline and
deterministic; intended both as a one-shot 1063-task audit and as a home for
the historical-bug regression cases.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.scorers import mcq, code
from evaluator.scorers import math as math_scorer


@dataclass(frozen=True)
class SelfTestFailure:
    task_id: str
    source: str
    reason: str
    detail: dict


def _synth_output(task: Task, ref: Reference) -> str:
    if task.source == "mmlu_pro":
        return f"Reasoning about the options. The answer is ({ref.gold})."
    if task.source == "math":
        if ref.solution:
            return ref.solution
        return f"Therefore the answer is \\boxed{{{ref.gold}}}."
    if task.source == "humaneval":
        body = (ref.prompt or "") + (ref.canonical_solution or "")
        return f"```python\n{body}\n```"
    return ref.gold or ""


_SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score}


def selftest_one(task: Task, ref: Reference) -> SelfTestFailure | None:
    output = _synth_output(task, ref)
    result = _SCORERS[task.source](task, output)
    if result.correct:
        return None
    return SelfTestFailure(task.id, task.source, "gold_not_recognized", dict(result.detail))


def selftest(tasks: list[Task], refs: dict[str, Reference]) -> list[SelfTestFailure]:
    failures: list[SelfTestFailure] = []
    for task in tasks:
        ref = refs.get(task.id)
        if ref is None:
            failures.append(SelfTestFailure(task.id, task.source, "missing_reference", {}))
            continue
        f = selftest_one(task, ref)
        if f is not None:
            failures.append(f)
    return failures

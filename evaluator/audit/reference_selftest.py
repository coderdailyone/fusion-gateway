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
    if task.source in ("mmlu_pro", "gpqa_diamond"):
        return f"Reasoning about the options. The answer is ({ref.gold})."
    if task.source in ("math", "aime", "math_l5"):
        if ref.solution:
            return ref.solution
        return f"Therefore the answer is \\boxed{{{ref.gold}}}."
    if task.source in ("humaneval", "livecodebench"):
        # KNOWN LIMITATION: we synthesize prompt + canonical_solution here,
        # but the real scoring path (evaluator/scorers/code.py) grades the
        # extracted completion ALONE — it never prepends the prompt. A real
        # completion that omits an import/setup line the prompt already
        # provided would fail real scoring while this self-test, which
        # always includes the prompt, cannot detect that class of bug. This
        # self-test proves the scorer recognizes a correct-and-complete
        # program; it is structurally blind to prompt-context leakage into
        # "correctness". See docs/BENCHMARK_REPORT.md's disagreement-audit
        # note for humaneval false-negatives.
        #
        # For livecodebench specifically: `code_generation_lite` ships no
        # gold/canonical completion at all (`ref.canonical_solution` is
        # always None for it — see `build_reference_index`), so this branch
        # degrades to a prompt-only body for LCB and will legitimately FAIL
        # to produce a passing program. That is expected and documented,
        # not a scorer bug: the LCB reference self-test is a best-effort
        # smoke (there is no independently-sourced gold solution to prove
        # the scorer recognizes), not a "gold recognized" guarantee like
        # the other sources get.
        body = (ref.prompt or "") + (ref.canonical_solution or "")
        return f"```python\n{body}\n```"
    return ref.gold or ""


_SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score,
            "gpqa_diamond": mcq.score, "aime": math_scorer.score,
            "math_l5": math_scorer.score, "livecodebench": code.score}


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

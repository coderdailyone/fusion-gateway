"""Official 0-shot chain-of-thought prompt templates per benchmark source.

Answer-format instructions follow each benchmark's official evaluation
protocol for chat/reasoning models: MMLU-Pro's "The answer is (X)." sink,
MATH's \\boxed{} final answer, HumanEval's single-code-block completion.
Consumes ONLY task.problem — never task.answer or task.tests (leakage guard).
"""
from __future__ import annotations

from evaluator.suite.types import Task

_MMLU = ("The following is a multiple choice question. Think step by step, "
         "then finish your response with a single line 'The answer is (X).' "
         "where X is the correct option letter.\n\n{problem}")
_MATH = ("Solve the following math problem. Reason step by step, and put your "
         "final answer within \\boxed{{}}.\n\n{problem}")
_HUMANEVAL = ("Complete the following Python function. Respond with a single "
              "Python code block containing the full function definition.\n\n{problem}")
_DEFAULT = "Solve the following problem. Put your final answer clearly at the end.\n\n{problem}"

_TEMPLATES = {"mmlu_pro": _MMLU, "math": _MATH, "humaneval": _HUMANEVAL}


def build(task: Task) -> str:
    template = _TEMPLATES.get(task.source, _DEFAULT)
    return template.format(problem=task.problem)

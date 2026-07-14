import re

from evaluator.suite.types import Task
from evaluator.scorers.base import Score

_LETTER = "[A-J]"

# Ordered by nothing in particular; extraction picks whichever match occurs
# LAST in the text (by end position), since models often restate options
# before concluding with a final answer.
_PATTERNS = [
    re.compile(rf"answer is\s*({_LETTER})", re.IGNORECASE),
    re.compile(rf"\b({_LETTER})\)", re.IGNORECASE),
    re.compile(rf"\\boxed\{{\s*({_LETTER})\s*\}}", re.IGNORECASE),
    re.compile(rf"\b({_LETTER})\b[.,!?]?\s*$", re.IGNORECASE),
]


def _extract_letter(text: str) -> str | None:
    best_letter = None
    best_end = -1
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if m.end() >= best_end:
                best_end = m.end()
                best_letter = m.group(1)
    return best_letter.upper() if best_letter else None


def score(task: Task, output_text: str) -> Score:
    letter = _extract_letter(output_text)
    if letter is None:
        return Score(False, {"reason": "unparseable"})
    expected = (task.answer or "").upper()
    correct = letter == expected
    return Score(correct, {"extracted": letter, "expected": expected})

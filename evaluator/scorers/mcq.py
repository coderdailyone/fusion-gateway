import re

from evaluator.suite.types import Task
from evaluator.scorers.base import Score

_L = "[A-J]"

# Priority 1: an explicit answer cue. Handles "answer is A", "Answer: A",
# "correct answer = A", "**Answer: A. 30 m**", "option A", "choice (A)".
# The letter may be wrapped in markdown/parens and followed by . ) : or text.
_CUE = re.compile(
    rf"(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is|:|=|-|–)?\s*\(?\*{{0,2}}({_L})\b"
    rf"|(?:option|choice)\s*\(?\*{{0,2}}({_L})\b",
    re.IGNORECASE,
)
# Fallbacks, each taking its LAST occurrence, tried in order.
_BOXED = re.compile(rf"\\boxed\{{\s*({_L})\s*\}}", re.IGNORECASE)
_PAREN = re.compile(rf"\(({_L})\)", re.IGNORECASE)
_FINAL = re.compile(rf"\b({_L})\b[.,!?*\s]*$", re.IGNORECASE)


def _extract_letter(text: str) -> str | None:
    cues = list(_CUE.finditer(text))
    if cues:
        m = cues[-1]
        return (m.group(1) or m.group(2)).upper()
    for pat in (_BOXED, _PAREN):
        hits = list(pat.finditer(text))
        if hits:
            return hits[-1].group(1).upper()
    m = _FINAL.search(text)
    return m.group(1).upper() if m else None


def score(task: Task, output_text: str) -> Score:
    letter = _extract_letter(output_text)
    if letter is None:
        return Score(False, {"reason": "unparseable"})
    expected = (task.answer or "").upper()
    correct = letter == expected
    return Score(correct, {"extracted": letter, "expected": expected})

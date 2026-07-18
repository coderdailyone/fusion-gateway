import re

from evaluator.suite.types import Task
from evaluator.scorers.base import Score
from evaluator.official.math_grade import math_equiv

# --- extraction -------------------------------------------------------

_BOXED_START = re.compile(r"\\boxed\{")
_ANSWER_PHRASE = re.compile(r"(?:answer|result)\s*(?:is|:)\s*", re.IGNORECASE)
_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def _find_last_boxed(text: str) -> str | None:
    """Return the contents of the LAST \\boxed{...} in text, honoring nested braces."""
    last = None
    pos = 0
    while True:
        m = _BOXED_START.search(text, pos)
        if m is None:
            break
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
        pos = m.end()
    return last


def _extract_after_phrase(text: str) -> str | None:
    matches = list(_ANSWER_PHRASE.finditer(text))
    if not matches:
        return None
    tail = text[matches[-1].end() :]
    m = _TOKEN.search(tail)
    return m.group(0) if m else None


def _extract_last_token(text: str) -> str | None:
    matches = list(_TOKEN.finditer(text))
    return matches[-1].group(0) if matches else None


def _extract_answer(text: str) -> str | None:
    boxed = _find_last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    after_phrase = _extract_after_phrase(text)
    if after_phrase is not None:
        return after_phrase
    return _extract_last_token(text)


# --- scoring -------------------------------------------------------


def score(task: Task, output_text: str) -> Score:
    extracted = _extract_answer(output_text)
    if extracted is None:
        return Score(False, {"reason": "unparseable"})
    expected_raw = task.answer if task.answer is not None else ""
    if math_equiv(extracted, expected_raw):
        return Score(True, {"extracted": extracted, "expected": expected_raw, "method": "official"})
    return Score(False, {"extracted": extracted, "expected": expected_raw, "method": "none"})

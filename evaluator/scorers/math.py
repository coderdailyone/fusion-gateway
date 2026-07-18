import re

import sympy

from evaluator.suite.types import Task
from evaluator.scorers.base import Score

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


# --- normalization / sympy parsing ------------------------------------

_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")
_FRAC = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
# LaTeX shorthand with single-digit numerator/denominator and no braces,
# e.g. \frac13  or  \dfrac12  (models emit this; the braced form above misses it).
_FRAC_SHORT = re.compile(r"\\d?frac(\d)(\d)")
# \text{...} units/labels — models box "3\text{ treeks}" when the answer is "3".
_TEXT = re.compile(r"\\(?:text|mbox|mathrm)\s*\{[^{}]*\}")
# Leading "variable relation" — MATH-500 answers like "x \in [-2,7]" or "x = 5";
# models box just the value ("[-2,7]" / "5"), so strip the prefix from both sides.
_VAR_PREFIX = re.compile(r"^[a-zA-Z]\s*(?:\\in|\\le|\\ge|\\leq|\\geq|=|<|>)\s*")


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = _TEXT.sub("", s)
    s = _VAR_PREFIX.sub("", s.strip())
    s = _TRAILING_PUNCT.sub("", s)
    return s.strip()


def _to_sympy(s: str):
    s = _normalize(s)
    if not s:
        return None
    t = s
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace("\\!", "").replace("\\,", "")
    t = _FRAC_SHORT.sub(r"(\1)/(\2)", t)
    t = _FRAC.sub(r"(\1)/(\2)", t)
    t = t.replace("^", "**")
    try:
        return sympy.sympify(t)
    except Exception:
        try:
            return sympy.nsimplify(t)
        except Exception:
            return None


# --- scoring -------------------------------------------------------


def score(task: Task, output_text: str) -> Score:
    extracted = _extract_answer(output_text)
    if extracted is None:
        return Score(False, {"reason": "unparseable"})

    expected_raw = task.answer if task.answer is not None else ""

    # 1) normalized exact string match first — reliably catches intervals/sets,
    #    unit-labelled numbers, and "x \in ..." answers where sympy can't compare.
    norm_extracted = _normalize(extracted).replace(" ", "")
    norm_expected = _normalize(expected_raw).replace(" ", "")
    if norm_extracted and norm_extracted == norm_expected:
        return Score(
            True,
            {"extracted": extracted, "expected": expected_raw, "method": "string"},
        )

    # 2) symbolic / numeric equivalence (1/2 == 0.5, 1+1 == 2, \frac shorthand).
    a = _to_sympy(extracted)
    b = _to_sympy(expected_raw)
    if a is not None and b is not None:
        try:
            if sympy.simplify(a - b) == 0:
                return Score(
                    True,
                    {"extracted": extracted, "expected": expected_raw, "method": "symbolic"},
                )
        except Exception:
            pass
        try:
            if abs(complex(a) - complex(b)) < 1e-9:
                return Score(
                    True,
                    {"extracted": extracted, "expected": expected_raw, "method": "numeric"},
                )
        except Exception:
            pass

    return Score(
        False,
        {"extracted": extracted, "expected": expected_raw, "method": "none"},
    )

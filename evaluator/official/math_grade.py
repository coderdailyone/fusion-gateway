"""Official MATH answer grading.

Ported faithfully from the Hendrycks MATH release `math_equivalence.py`
(https://github.com/hendrycks/math , commit `357963a`, MIT License) — the
`_strip_string` / `is_equiv` normalization used by the MATH paper and adopted
by EleutherAI lm-evaluation-harness (`minerva_math`). Local additions are
limited to `math_equiv`, which layers a sympy equivalence fallback on top of
the official string comparison (also the lm-eval recipe); every such edit is
commented. No randomness, no I/O — deterministic string/CAS comparison only.
"""
from __future__ import annotations

import re

import sympy


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_i, b_i = int(a), int(b)
        assert string == "{}/{}".format(a_i, b_i)
        return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    # Official: units appear on the right as "\text{ ...}".
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # official asserts len == 2; be defensive and take the left side
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # strip a short LHS assignment: "k = 5" -> "5" (official: len(lhs) <= 2)
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None) -> bool:
    """Official MATH string-normalized equality (Hendrycks)."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


# --- local addition: modern MATH-eval normalization + sympy fallback ------

# Symmetric normalization used by current MATH graders (Qwen2.5-Math /
# DeepSeek-Math) on top of the 2021 Hendrycks string comparison: drop LaTeX
# spacing commands and a leading variable-assignment / set-membership prefix,
# so a bare answer "[-2,7]" matches a gold written "x \in [-2,7]" (and vice
# versa). Applied to BOTH sides in math_equiv, so it never favors one model's
# phrasing. Hendrycks already strips a short "x =" LHS; it does NOT strip
# "x \in ..." / "x < ..." or the \, thin-space, which is what this adds.
_LATEX_SPACE = re.compile(r"\\[,!;: ]")
_VAR_PREFIX = re.compile(
    r"^\s*\$?\s*[a-zA-Z]\s*"
    r"(?:\\in|\\leq|\\geq|\\le|\\ge|=|<|>|\\subseteq|\\subsetneq|\\subset)\s*"
)


def _modern_normalize(s: str) -> str:
    s = _LATEX_SPACE.sub("", s)
    return _VAR_PREFIX.sub("", s.strip())


def _to_sympy(s: str):
    # _to_sympy must guard _strip_string itself: unlike is_equiv, it runs
    # outside is_equiv's try/except, so a bare trailing \frac/\sqrt (which
    # makes the upstream helpers raise IndexError) would otherwise crash us.
    try:
        t = _strip_string(s)
    except Exception:
        return None
    if not t:
        return None
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("^", "**")
    # translate \frac{a}{b} / \dfrac{a}{b} -> (a)/(b) (already brace-normalized
    # by _fix_fracs, so a and b are single braced groups)
    t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    try:
        return sympy.sympify(t)
    except Exception:
        try:
            return sympy.nsimplify(t)
        except Exception:
            return None


def math_equiv(extracted: str, gold: str) -> bool:
    """Official Hendrycks string match, then modern symmetric normalization,
    then a sympy equivalence fallback."""
    if is_equiv(extracted, gold):
        return True
    # modern MATH-eval normalization on BOTH sides (variable prefix, \, spaces)
    if is_equiv(_modern_normalize(extracted), _modern_normalize(gold)):
        return True
    a, b = _to_sympy(_modern_normalize(extracted)), _to_sympy(_modern_normalize(gold))
    if a is None or b is None:
        return False
    try:
        if sympy.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(complex(a) - complex(b)) < 1e-9
    except Exception:
        return False

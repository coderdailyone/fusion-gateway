import pytest

from evaluator.suite.types import Task
from evaluator.consistency.normalize import (ballot_key, doctest_signature,
                                             extract_doctests)


def _exec_runner(code, stdin="", **kw):
    """A real-enough stub: exec the assembled program and capture stdout."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        status = "ok"
    except Exception:
        status = "error"

    class R:
        pass
    r = R()
    r.status, r.stdout = status, buf.getvalue()
    return r


MCQ = Task(id="m1", source="mmlu_pro", problem="2+2?\nA) 3\nB) 4", answer="B",
           tests=(), meta={})
MATH = Task(id="q1", source="math", problem="Compute 1/2.", answer="\\frac{1}{2}",
            tests=(), meta={})
CODE = Task(id="c1", source="humaneval",
            problem=('def add(a, b):\n    """ Add two numbers.\n'
                     '    >>> add(1, 2)\n    3\n    >>> add(0, 0)\n    0\n    """\n'),
            answer=None, tests=({"kind": "pyfunc", "test": "assert True",
                                 "entry_point": "add"},), meta={})


def test_mcq_ballot_key_is_the_letter():
    assert ballot_key(MCQ, "reasoning...\nThe answer is (B).") == "B"


def test_unparseable_answer_is_a_spoiled_ballot():
    assert ballot_key(MCQ, "I have no idea") is None


def test_math_ballot_key_is_the_extracted_answer():
    assert ballot_key(MATH, "so the value is \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_doctests_reads_examples_from_the_problem_statement():
    got = extract_doctests(CODE.problem)
    assert got == [("add(1, 2)", "3"), ("add(0, 0)", "0")]


def test_doctest_signature_marks_pass_and_fail_differently():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"

    sig_good = doctest_signature(CODE, good, _exec_runner)
    sig_bad = doctest_signature(CODE, bad, _exec_runner)
    assert sig_good is not None and sig_good != sig_bad
    assert sig_good.startswith("PASS")      # all doctests satisfied
    assert sig_bad.startswith("FAIL")


def test_code_ballot_key_without_doctests_falls_back_to_code_text():
    plain = Task(id="c2", source="humaneval", problem="def f(): pass",
                 answer=None, tests=(), meta={})
    key = ballot_key(plain, "```python\ndef f():\n    return 1\n```",
                     runner=_exec_runner)
    assert key is not None and "return 1" in key


# --------------------------------------------------------------------------
# C2 -- real HumanEval docstring shapes the ">>> line, then the line below"
# regex mis-parses. Each must be SKIPPED, never turned into an expectation
# that a correct sample fails.
# --------------------------------------------------------------------------

# HumanEval/108, verbatim. The last example is the docstring's last line, so
# the "expected" the regex captures is the closing triple quote.
HE108 = Task(
    id="HumanEval/108", source="humaneval", answer=None, tests=(), meta={},
    problem='''def count_nums(arr):
    """
    Write a function count_nums which takes an array of integers and returns
    the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative:
    e.g. -123 has signed digits -1, 2, and 3.
    >>> count_nums([]) == 0
    >>> count_nums([-1, 11, -11]) == 1
    >>> count_nums([1, 1, 2]) == 3
    """
''')

# HumanEval/116, verbatim. The third example puts call and expected value on
# ONE line, which Python reads as a subscript.
HE116 = Task(
    id="HumanEval/116", source="humaneval", answer=None, tests=(), meta={},
    problem='''def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    It must be implemented like this:
    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]
    """
''')

# HumanEval/113, verbatim. The second example's expected value spans two lines
# and would be truncated to a dangling "[...," fragment.
HE113_PROBLEM = '''def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, return a list.
    Each element i of the output should be "the number of odd elements in the
    string i of the input." where all the i's should be replaced by the number
    of odd digits in the i'th string of the input.

    >>> odd_count(['1234567'])
    ["the number of odd elements 4n the str4ng 4 of the 4nput."]
    >>> odd_count(['3',"11111111"])
    ["the number of odd elements 1n the str1ng 1 of the 1nput.",
     "the number of odd elements 8n the str8ng 8 of the 8nput."]
    """
'''


def test_trailing_doctest_never_captures_the_docstring_terminator():
    # pre-fix this returned [('count_nums([1, 1, 2]) == 3', '"""')], so a
    # correct count_nums printed "True", missed '"""', and scored FAIL.
    assert extract_doctests(HE108.problem) == []


def test_inline_call_and_expected_on_one_line_is_skipped():
    assert extract_doctests(HE116.problem) == []


def test_inline_call_and_expected_is_skipped_even_when_more_text_follows():
    # the /116 shape but NOT the docstring's last line, so the terminator rule
    # cannot be what saves us -- this pins the inline rule specifically.
    problem = ('def sort_array(arr):\n    """\n'
               '    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]\n'
               '    Note: negatives keep their relative order.\n'
               '    """\n')
    assert extract_doctests(problem) == []


def test_multi_line_expected_is_skipped_not_truncated():
    # the single-line example survives; the two-line one is dropped rather than
    # captured as the dangling first line.
    got = extract_doctests(HE113_PROBLEM)
    assert got == [("odd_count(['1234567'])",
                    '["the number of odd elements 4n the str4ng 4 of the 4nput."]')]


def test_blank_expected_line_is_skipped():
    problem = 'def f(x):\n    """\n    >>> f(1)\n\n    more prose\n    """\n'
    assert extract_doctests(problem) == []


def test_ordinary_doctests_still_extract_after_the_skip_rules():
    assert extract_doctests(CODE.problem) == [("add(1, 2)", "3"), ("add(0, 0)", "0")]


def test_correct_code_is_not_marked_failing_by_a_mis_parsed_docstring():
    """The user-visible consequence of C2: 20 of 217 officially-correct
    HumanEval samples were given a FAIL signature."""
    correct = ("```python\ndef count_nums(arr):\n"
               "    def sd(n):\n"
               "        s = str(abs(n))\n"
               "        d = [int(c) for c in s]\n"
               "        if n < 0:\n"
               "            d[0] = -d[0]\n"
               "        return sum(d)\n"
               "    return sum(1 for x in arr if sd(x) > 0)\n```")
    # no runnable case survives, so voting honestly falls back to text
    assert doctest_signature(HE108, correct, _exec_runner) is None
    key = ballot_key(HE108, correct, runner=_exec_runner)
    assert key is not None and not key.startswith("FAIL")


# --------------------------------------------------------------------------
# T4 -- a missing runner must be loud, not a silent tier-wide degradation.
# --------------------------------------------------------------------------

def test_code_ballot_key_without_a_runner_raises():
    with pytest.raises(ValueError) as exc:
        ballot_key(CODE, "```python\ndef add(a, b):\n    return a + b\n```")
    assert "run_code" in str(exc.value)


def test_livecodebench_ballot_key_without_a_runner_raises():
    lcb = Task(id="l1", source="livecodebench", problem="solve it", answer=None,
               tests=(), meta={})
    with pytest.raises(ValueError):
        ballot_key(lcb, "print(1)")


def test_non_code_sources_still_work_without_a_runner():
    assert ballot_key(MCQ, "The answer is (B).") == "B"
    assert ballot_key(MATH, "so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_a_runner_that_raises_yields_a_fail_signature_not_a_crash():
    """One OSError from Popen under load must not kill a multi-hour paid batch."""
    def boom(code, stdin="", **kw):
        raise OSError(24, "Too many open files")

    assert doctest_signature(CODE, "```python\ndef add(a, b):\n"
                                   "    return a + b\n```", boom) == "FAIL:exec"

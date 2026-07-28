from evaluator.suite.types import Task
from evaluator.consistency.normalize import (ballot_key, doctest_signature,
                                             extract_doctests)

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

    def runner(code, stdin="", **kw):
        # a real-enough stub: exec the assembled program and capture stdout
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

    sig_good = doctest_signature(CODE, good, runner)
    sig_bad = doctest_signature(CODE, bad, runner)
    assert sig_good is not None and sig_good != sig_bad
    assert sig_good.startswith("PASS")      # all doctests satisfied
    assert sig_bad.startswith("FAIL")


def test_code_ballot_key_without_doctests_falls_back_to_code_text():
    plain = Task(id="c2", source="humaneval", problem="def f(): pass",
                 answer=None, tests=(), meta={})
    key = ballot_key(plain, "```python\ndef f():\n    return 1\n```", runner=None)
    assert key is not None and "return 1" in key

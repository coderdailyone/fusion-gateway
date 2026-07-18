from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.audit.reference_selftest import selftest_one

def _t(source, answer=None, tests=()):
    return Task(id="q", source=source, problem="P", answer=answer, tests=tests, meta={})

def test_mmlu_gold_recognized():
    t = _t("mmlu_pro", answer="C")
    ref = Reference("q", "mmlu_pro", gold="C")
    assert selftest_one(t, ref) is None

def test_math_gold_via_solution_recognized():
    t = _t("math", answer="42")
    ref = Reference("q", "math", gold="42", solution="Adding up we get \\boxed{42}.")
    assert selftest_one(t, ref) is None

def test_math_unit_answer_recognized():  # historical bug #3
    t = _t("math", answer="3")
    ref = Reference("q", "math", gold="3", solution="So \\boxed{3\\text{ treeks}}.")
    assert selftest_one(t, ref) is None

def test_humaneval_canonical_passes():
    tests = ({"kind": "pyfunc",
              "test": "def check(candidate):\n    assert candidate(2) == 3\n",
              "entry_point": "f"},)
    t = _t("humaneval", tests=tests)
    ref = Reference("q", "humaneval", prompt="def f(x):\n", canonical_solution="    return x + 1\n",
                    entry_point="f", test=tests[0]["test"])
    assert selftest_one(t, ref) is None

def test_detects_broken_gold():
    # a gold the scorer cannot match surfaces as a failure
    t = _t("mmlu_pro", answer="C")
    ref = Reference("q", "mmlu_pro", gold="Z")  # not A-J -> unparseable/incorrect
    assert selftest_one(t, ref) is not None

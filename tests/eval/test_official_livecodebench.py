from evaluator.official.livecodebench_exec import (
    normalize_tests, build_functional_check, functional_entry_point)
from evaluator.scorers.code import score
from evaluator.suite.types import Task


def test_stdin_case_shape():
    row = {"test_type": "stdin",
           "cases": [{"input": "3\n", "output": "9\n"}]}
    out = normalize_tests(row)
    assert out == ({"kind": "stdin", "stdin": "3\n", "expected_stdout": "9\n"},)


def test_functional_check_asserts_each_case():
    src = build_functional_check("sq", [("3", "9"), ("4", "16")])
    assert "def check(candidate):" in src
    assert "candidate(3) == 9" in src
    assert "candidate(4) == 16" in src


def test_functional_case_shape():
    row = {"test_type": "functional", "fn_name": "sq",
           "cases": [{"input": "3", "output": "9"}]}
    out = normalize_tests(row)
    assert len(out) == 1 and out[0]["kind"] == "pyfunc"
    # entry_point resolves a class-method callable (LeetCode style), not a bare name
    assert out[0]["entry_point"] == "(Solution().sq if 'Solution' in dir() else sq)"
    assert "candidate(3) == 9" in out[0]["test"]


def test_leetcode_class_solution_completion_passes():
    # regression from the paid hard-tier smoke: LCB functional problems are
    # LeetCode `class Solution` methods; a correct completion must PASS, not
    # NameError on a bare entry-point.
    tests = normalize_tests({"test_type": "functional", "fn_name": "sq",
                             "cases": [{"input": "3", "output": "9"},
                                       {"input": "4", "output": "16"}]})
    task = Task(id="q", source="livecodebench", problem="", answer=None,
                tests=tests, meta={})
    completion = "```python\nclass Solution:\n    def sq(self, x):\n        return x * x\n```"
    assert score(task, completion).correct

    top_level = "```python\ndef sq(x):\n    return x * x\n```"  # fallback path
    assert score(task, top_level).correct

    wrong = "```python\nclass Solution:\n    def sq(self, x):\n        return x + 1\n```"
    assert not score(task, wrong).correct

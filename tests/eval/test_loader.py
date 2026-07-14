import pytest
from evaluator.suite.manifest import SourceSpec, Manifest, content_sha
from evaluator.suite.loader import load_suite, parse_task, SuiteHashMismatch

MMLU_REC = {"id": "q1", "question": "2+2?", "options": ["3", "4", "5"], "answer": "B"}

def test_parse_mmlu_pro():
    t = parse_task("mmlu_pro", MMLU_REC)
    assert t.source == "mmlu_pro" and t.answer == "B"
    assert "4" in t.problem and "B" in t.problem  # options rendered into the problem

def test_parse_humaneval():
    rec = {"id": "HumanEval/0", "prompt": "def f(x):\n    ...",
           "test": "def check(candidate):\n    assert candidate(1) == 2",
           "entry_point": "f"}
    t = parse_task("humaneval", rec)
    assert t.source == "humaneval" and t.answer is None
    assert t.problem.startswith("def f(x)")
    assert t.tests[0]["entry_point"] == "f" and t.tests[0]["kind"] == "pyfunc"

def test_load_suite_verifies_hash():
    recs = [MMLU_REC]
    good = content_sha(recs)
    spec = SourceSpec("mmlu_pro", "d", "r", "test", ("q1",), good)
    m = Manifest(1, (spec,))
    tasks = load_suite(m, {"mmlu_pro": lambda s: recs})
    assert len(tasks) == 1 and tasks[0].id == "q1"

def test_load_suite_rejects_tampered_data():
    recs = [MMLU_REC]
    spec = SourceSpec("mmlu_pro", "d", "r", "test", ("q1",), "0" * 64)  # wrong sha
    m = Manifest(1, (spec,))
    with pytest.raises(SuiteHashMismatch):
        load_suite(m, {"mmlu_pro": lambda s: recs})

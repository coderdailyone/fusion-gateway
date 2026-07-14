from evaluator.suite.types import Task
from evaluator.scorers.mcq import score

T = Task(id="q", source="mmlu_pro", problem="...", answer="B", tests=(), meta={})

def test_correct_various_formats():
    for out in ["The answer is B.", "Final: (B)", "\\boxed{B}", "... so B"]:
        assert score(T, out).correct

def test_incorrect():
    assert not score(T, "The answer is A.").correct

def test_unparseable():
    s = score(T, "I am not sure.")
    assert not s.correct and s.detail["reason"] == "unparseable"

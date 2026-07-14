from evaluator.suite.types import Task
from evaluator.scorers.math import score

def T(ans): return Task(id="q", source="math", problem="...", answer=ans, tests=(), meta={})

def test_exact_and_boxed():
    assert score(T("42"), "so \\boxed{42}").correct
    assert score(T("42"), "the answer is 42").correct

def test_symbolic_equivalence():
    assert score(T("1/2"), "\\boxed{0.5}").correct
    assert score(T("2"), "\\boxed{1+1}").correct

def test_wrong():
    assert not score(T("42"), "\\boxed{41}").correct

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

def test_interval_exact_boxed():
    # interval answers still match when both sides agree (official is_equiv)
    assert score(T("[-2,7]"), "so \\boxed{[-2,7]}").correct
    assert not score(T("[-2,7]"), "\\boxed{[0,5]}").correct

def test_unit_labelled_number():
    # official _remove_right_units strips the \text{...} unit label
    assert score(T("3"), "\\boxed{3\\text{ treeks}}").correct

def test_var_in_prefix_now_strict():
    # documents the intentional alignment change: "x \in" prefix is not stripped
    assert not score(T("x \\in [-2,7]"), "so \\boxed{[-2,7]}").correct

def test_frac_shorthand():
    # Regression from the pilot: models write \frac13 (no braces).
    assert score(T("\\frac{1}{3}"), "so \\boxed{\\frac13}").correct
    assert score(T("1/2"), "\\boxed{\\dfrac12}").correct

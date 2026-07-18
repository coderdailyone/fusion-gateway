from evaluator.suite.types import Task
from evaluator.official.prompts import build

def _t(source, problem="PROBLEM", answer="X"):
    return Task(id="q", source=source, problem=problem, answer=answer, tests=(), meta={})

def test_mmlu_prompt_official_format():
    p = build(_t("mmlu_pro"))
    assert "The answer is (X)" in p or "answer is (X)" in p
    assert "step by step" in p.lower()
    assert "PROBLEM" in p

def test_math_prompt_boxed():
    p = build(_t("math"))
    assert "\\boxed" in p
    assert "step by step" in p.lower()
    assert "PROBLEM" in p

def test_humaneval_prompt_codeblock():
    p = build(_t("humaneval"))
    assert "PROBLEM" in p

def test_prompt_never_contains_answer():
    # leakage: the gold answer must not be injected
    p = build(_t("math", problem="compute", answer="SECRET_GOLD"))
    assert "SECRET_GOLD" not in p

import pytest
from evaluator.suite.types import Task
from evaluator.runner import run_one, build_prompt, FrozenOutput

T = Task(id="q1", source="math", problem="What is 2+2?", answer="4", tests=(), meta={})

def test_prompt_excludes_answer():
    p = build_prompt(T)
    assert "2+2" in p and "4" not in p.replace("2+2", "")  # the answer '4' not leaked

def test_run_one_captures_usage():
    def fake(model, prompt):
        return {"text": "The answer is 4", "in_tokens": 10, "out_tokens": 5, "cost_usd": 0.0001}
    fo = run_one(T, "deepseek-chat", fake)
    assert fo.status == "ok" and fo.output_text == "The answer is 4"
    assert fo.in_tokens == 10 and fo.cost_usd == 0.0001 and fo.model == "deepseek-chat"

def test_run_one_handles_provider_error():
    def boom(model, prompt): raise RuntimeError("upstream 500")
    fo = run_one(T, "glm-4.6", boom)
    assert fo.status == "error" and "upstream 500" in fo.error and fo.out_tokens == 0

def test_run_one_does_not_crash_when_answer_appears_in_problem():
    # Regression: the answer string ("5") legitimately appears in the problem.
    # run_one must NOT raise — leakage protection is structural, not a
    # substring assert that would false-positive here.
    task = Task(id="q2", source="math", problem="A pentagon has 5 sides. How many?",
                answer="5", tests=(), meta={})
    def fake(model, prompt):
        return {"text": "5", "in_tokens": 3, "out_tokens": 1, "cost_usd": 0.0}
    fo = run_one(task, "deepseek-chat", fake)
    assert fo.status == "ok" and fo.output_text == "5"

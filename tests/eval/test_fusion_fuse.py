from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.review import Verdict
from evaluator.fusion.fuse import FusionResult, fuse

TASK = Task(id="t1", source="mmlu_pro", problem="2+2?", answer="B", tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"a-model": "says A", "b-model": "says B"})
REVIEWS = {"a-model": {"b-model": Verdict("correct", "4 is right")}}


def test_fuse_calls_the_named_fuser_and_returns_its_answer():
    calls = []

    def fake_completion(model, prompt):
        calls.append((model, prompt))
        return {"text": "The answer is (B).", "in_tokens": 5, "out_tokens": 3,
                "cost_usd": 0.002}

    res = fuse(TASK, CASE, REVIEWS, fake_completion, fuser="glm-5.2")
    assert isinstance(res, FusionResult)
    assert res.answer == "The answer is (B)."
    assert res.fuser == "glm-5.2" and calls[0][0] == "glm-5.2"
    assert res.cost_usd == 0.002 and res.status == "ok"
    assert "4 is right" in calls[0][1]        # review evidence reached the fuser


def test_fuse_records_error_instead_of_raising():
    def boom(model, prompt):
        raise RuntimeError("mirror 503")

    res = fuse(TASK, CASE, REVIEWS, boom, fuser="glm-5.2")
    assert res.status == "error" and res.answer == "" and res.cost_usd == 0.0
    assert "mirror 503" in (res.error or "")

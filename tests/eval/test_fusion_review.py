from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.review import Verdict, cross_review, parse_review

TASK = Task(id="t1", source="mmlu_pro", problem="2+2?", answer="B", tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"a-model": "says A", "b-model": "says B"})


def test_parse_review_reads_verdict_lines():
    out = parse_review(
        "Some preamble\n"
        "VERDICT a-model wrong picked A but 4 is right\n"
        "VERDICT b-model correct matches 4\n",
        valid_targets={"a-model", "b-model"})
    assert out["a-model"] == Verdict("wrong", "picked A but 4 is right")
    assert out["b-model"].verdict == "correct"


def test_parse_review_drops_malformed_and_unknown_lines():
    out = parse_review(
        "VERDICT a-model banana nonsense verdict\n"     # invalid verdict word
        "VERDICT ghost-model correct not in panel\n"     # unknown target
        "totally unrelated line\n"
        "VERDICT b-model unsure cannot tell\n",
        valid_targets={"a-model", "b-model"})
    assert set(out) == {"b-model"}
    assert out["b-model"].verdict == "unsure"


def test_cross_review_never_asks_a_model_about_itself():
    seen = {}

    def fake_completion(model, prompt):
        seen[model] = prompt
        other = "b-model" if model == "a-model" else "a-model"
        return {"text": f"VERDICT {other} correct looks right",
                "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

    reviews = cross_review(TASK, CASE, fake_completion)
    assert set(reviews) == {"a-model", "b-model"}
    assert set(reviews["a-model"]) == {"b-model"}     # reviewed only the other
    assert "says A" not in seen["a-model"]            # own answer not in its prompt

from evaluator.suite.types import Task
from evaluator.sampler import sample
from evaluator.store import new_run_dir


def mk_tasks():
    return [Task(id="t1", source="math", problem="1+1?", answer="2", tests=(), meta={}),
            Task(id="t2", source="math", problem="2+2?", answer="4", tests=(), meta={})]


def price(model, in_t, out_t):        # deterministic fake pricing
    return 0.001


def test_scores_and_prices(tmp_path):
    rd = new_run_dir(tmp_path, "s", "t0")
    def good(model, prompt):  # always answers the problem's own text back... use fixed
        return {"text": "2" if "1+1" in prompt else "4", "in_tokens": 5, "out_tokens": 1, "cost_usd": 9.9}
    rows = sample({"m1": good}, mk_tasks(), rd, cost_fn=price)
    assert len(rows) == 2
    assert all(r.correct for r in rows)
    assert all(r.cost_usd == 0.001 for r in rows)   # pricing, NOT the 9.9 from completion


def test_resumable_skips_done(tmp_path):
    rd = new_run_dir(tmp_path, "s", "t0")
    calls = {"n": 0}
    def counting(model, prompt):
        calls["n"] += 1
        return {"text": "2" if "1+1" in prompt else "4", "in_tokens": 5, "out_tokens": 1, "cost_usd": 0.0}
    sample({"m1": counting}, mk_tasks(), rd, cost_fn=price)
    assert calls["n"] == 2
    sample({"m1": counting}, mk_tasks(), rd, cost_fn=price)   # rerun
    assert calls["n"] == 2          # no new calls — all (task,model) already frozen

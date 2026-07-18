from dataclasses import dataclass

from evaluator.suite.types import Task
from scripts.resample_official import budget_gate, estimate_cost, run_budgeted


def test_budget_gate_states():
    assert budget_gate(spent=0.0, next_cost=1.0, ceiling=10.0) == "ok"
    assert budget_gate(spent=8.5, next_cost=0.1, ceiling=10.0) == "warn"   # >=80%
    assert budget_gate(spent=9.99, next_cost=0.1, ceiling=10.0) == "stop"  # would cross 100%


def test_budget_gate_exact_boundaries():
    # money-guard edges must be pinned: a `>`/`>=` swap here over- or under-spends.
    # spent + next_cost == ceiling exactly -> NOT stop (it exactly fits), and since
    # spent (9.0) is >= 80% it is a warn.
    assert budget_gate(spent=9.0, next_cost=1.0, ceiling=10.0) == "warn"
    # spent exactly at the 80% warn threshold -> warn.
    assert budget_gate(spent=8.0, next_cost=0.0, ceiling=10.0) == "warn"
    # just below 80% with room to spare -> ok.
    assert budget_gate(spent=7.999, next_cost=0.0, ceiling=10.0) == "ok"
    # a hair over the ceiling -> stop.
    assert budget_gate(spent=9.0, next_cost=1.0001, ceiling=10.0) == "stop"


def test_estimate_cost_sums():
    cost_fn = lambda model, i, o: 0.001 * (i + o)
    rows = [("m", 100, 200), ("m", 0, 100)]
    assert abs(estimate_cost(rows, cost_fn) - (0.001*300 + 0.001*100)) < 1e-9


# --- run_budgeted: fake deps so the loop is offline/deterministic -----------


@dataclass(frozen=True)
class _FakeFrozen:
    """Minimal frozen-output stand-in: only the fields run_budgeted touches
    (task_id/model for the resume `done` set, in_tokens/out_tokens for
    cost_fn)."""
    task_id: str
    model: str
    in_tokens: int
    out_tokens: int


def _mk_tasks(n):
    return [Task(id=f"t{i}", source="math", problem="x" * 40, answer="1",
                 tests=(), meta={}) for i in range(n)]


def _make_fake_deps(store: list):
    """In-memory read_frozen/append_frozen backed by `store`, plus a fake
    run_one that never calls a real model and a fake build_prompt."""

    def read_frozen(run_dir):
        return list(store)

    def append_frozen(run_dir, fo):
        store.append(fo)

    def build_prompt(task):
        return task.problem

    def run_one(task, model_name, fn):
        return _FakeFrozen(task_id=task.id, model=model_name, in_tokens=50, out_tokens=50)

    return {
        "read_frozen": read_frozen,
        "append_frozen": append_frozen,
        "build_prompt": build_prompt,
        "run_one": run_one,
    }


def test_run_budgeted_hard_stops():
    tasks = _mk_tasks(10)
    models = {"fake-model": lambda model, prompt: None}  # never invoked by fake run_one
    flat_cost_fn = lambda model, i, o: 1.0  # every completed call costs exactly $1
    store: list = []
    deps = _make_fake_deps(store)

    result = run_budgeted(models, tasks, "unused-run-dir", ceiling=3.0,
                           cost_fn=flat_cost_fn, deps=deps)

    assert result["stopped"] is True
    assert result["completed"] <= 3
    assert result["spent"] <= 3.0
    assert len(store) == result["completed"]

    # Resumed call: already-frozen pairs must be skipped (no re-spend), and
    # the loop continues from where it left off under a higher ceiling.
    result2 = run_budgeted(models, tasks, "unused-run-dir", ceiling=10.0,
                            cost_fn=flat_cost_fn, deps=deps)

    assert result2["completed"] == len(tasks) - result["completed"]
    assert len(store) == len(tasks)
    assert result2["spent"] <= 10.0

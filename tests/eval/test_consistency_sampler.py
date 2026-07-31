import pytest

from evaluator.consistency.sampler import (
    accepts_kwarg, ballot_counts, frozen_by_pair, freezer, sample_dir,
    sample_dirs, sample_many, sample_task,
)
from evaluator.store import read_frozen
from evaluator.suite.types import Task


def mk_tasks(n=2):
    return [Task(id=f"t{i}", source="math", problem=f"{i}+{i}?", answer=str(2 * i),
                 tests=(), meta={}) for i in range(n)]


def price(model, in_t, out_t):        # deterministic fake pricing
    return 0.01


def counter_fn(calls, text="ans"):
    def fn(model, prompt):
        calls.append((model, prompt))
        return {"text": f"{text}{len(calls)}", "in_tokens": 5, "out_tokens": 7,
                "cost_usd": 0.0}
    return fn


# --- sample_task -----------------------------------------------------------


def test_sample_task_draws_k_independent_samples(tmp_path):
    calls = []
    texts = sample_task(mk_tasks(1)[0], "m1", 3, counter_fn(calls))
    assert len(calls) == 3                 # k real calls, not one cached
    assert texts == ["ans1", "ans2", "ans3"]


def test_sample_task_freezes_every_sample(tmp_path):
    task = mk_tasks(1)[0]
    sample_task(task, "m1", 3, counter_fn([]), on_frozen=freezer(tmp_path))
    rows = read_frozen(tmp_path)
    assert len(rows) == 3                  # every paid sample hits disk
    assert all(r.task_id == "t0" and r.model == "m1" for r in rows)


def test_sample_task_freezes_before_a_later_call_explodes(tmp_path):
    """An interrupted run must keep the samples already paid for."""
    state = {"n": 0}

    def flaky(model, prompt):
        state["n"] += 1
        if state["n"] == 3:
            raise KeyboardInterrupt("user hit ctrl-c")
        return {"text": "x", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

    with pytest.raises(KeyboardInterrupt):
        sample_task(mk_tasks(1)[0], "m1", 5, flaky, on_frozen=freezer(tmp_path))
    assert len(read_frozen(tmp_path)) == 2


def test_sample_task_records_api_error_as_a_row(tmp_path):
    def boom(model, prompt):
        raise RuntimeError("502 upstream")

    texts = sample_task(mk_tasks(1)[0], "m1", 2, boom, on_frozen=freezer(tmp_path))
    assert texts == ["", ""]
    rows = read_frozen(tmp_path)
    assert [r.status for r in rows] == ["error", "error"]


def test_sample_task_forwards_temperature_when_the_fn_takes_one():
    seen = []

    def fn(model, prompt, temperature=None):
        seen.append(temperature)
        return {"text": "a", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

    sample_task(mk_tasks(1)[0], "m1", 2, fn, temperature=0.8)
    assert seen == [0.8, 0.8]


def test_sample_task_refuses_to_drop_an_unsupported_temperature():
    """Silently sampling at the provider default would look identical to a
    correct run — k identical ballots — so it must raise instead."""
    def fn(model, prompt):
        return {"text": "a", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

    with pytest.raises(ValueError, match="temperature"):
        sample_task(mk_tasks(1)[0], "m1", 2, fn, temperature=0.8)


def test_sample_task_rejects_k_below_one():
    with pytest.raises(ValueError):
        sample_task(mk_tasks(1)[0], "m1", 0, counter_fn([]))


def test_accepts_kwarg():
    assert accepts_kwarg(lambda m, p, temperature=None: None, "temperature")
    assert accepts_kwarg(lambda m, p, **kw: None, "temperature")
    assert not accepts_kwarg(lambda m, p: None, "temperature")


# --- layout / readback -----------------------------------------------------


def test_sample_dirs_are_indexed_and_taggable(tmp_path):
    assert sample_dir(tmp_path, 0).name == "k0"
    assert sample_dir(tmp_path, 2, "kimi").name == "k2_kimi"
    for name in ("k1", "k0", "k10", "notes"):
        (tmp_path / name).mkdir()
    assert [d.name for d in sample_dirs(tmp_path)] == ["k0", "k1", "k10"]


def test_frozen_by_pair_merges_shards(tmp_path):
    task = mk_tasks(1)[0]
    sample_task(task, "m1", 1, counter_fn([]), on_frozen=freezer(sample_dir(tmp_path, 0)))
    sample_task(task, "m1", 1, counter_fn([]), on_frozen=freezer(sample_dir(tmp_path, 1)))
    sample_task(task, "m2", 1, counter_fn([]),
                on_frozen=freezer(sample_dir(tmp_path, 0, "shard1")))
    pairs = frozen_by_pair(tmp_path)
    assert ballot_counts(tmp_path) == {("t0", "m1"): 2, ("t0", "m2"): 1}
    assert len(pairs[("t0", "m1")]) == 2


# --- sample_many -----------------------------------------------------------


def test_sample_many_draws_k_per_pair_and_freezes_them(tmp_path):
    calls = []
    res = sample_many({"m1": counter_fn(calls)}, mk_tasks(2), tmp_path, 3,
                      ceiling=100.0, cost_fn=price, log=lambda s: None)
    assert len(calls) == 6                     # 2 tasks x k=3
    assert res["completed"] == 6 and not res["stopped"]
    assert ballot_counts(tmp_path) == {("t0", "m1"): 3, ("t1", "m1"): 3}
    assert [d.name for d in sample_dirs(tmp_path)] == ["k0", "k1", "k2"]


def test_sample_many_is_resumable_at_zero_cost(tmp_path):
    calls = []
    fn = counter_fn(calls)
    sample_many({"m1": fn}, mk_tasks(2), tmp_path, 3, 100.0, price,
                log=lambda s: None)
    assert len(calls) == 6
    res = sample_many({"m1": fn}, mk_tasks(2), tmp_path, 3, 100.0, price,
                      log=lambda s: None)
    assert len(calls) == 6                     # pairs already at k -> no calls
    assert res["completed"] == 0


def test_sample_many_tops_up_a_partial_pair(tmp_path):
    calls = []
    fn = counter_fn(calls)
    sample_many({"m1": fn}, mk_tasks(2), tmp_path, 1, 100.0, price,
                log=lambda s: None)
    assert len(calls) == 2
    sample_many({"m1": fn}, mk_tasks(2), tmp_path, 3, 100.0, price,
                log=lambda s: None)
    assert len(calls) == 6                     # only the missing 4 are bought
    assert ballot_counts(tmp_path) == {("t0", "m1"): 3, ("t1", "m1"): 3}


def test_sample_many_ceiling_is_shared_across_the_k_passes(tmp_path):
    """The budget is per-root, not per-pass: k passes must not each spend it."""
    calls = []
    res = sample_many({"m1": counter_fn(calls)}, mk_tasks(2), tmp_path, 5,
                      ceiling=0.05, cost_fn=price, log=lambda s: None)
    # price() is $0.01/call, so $0.05 buys 5 calls, not 5 per pass.
    assert len(calls) == 5
    assert res["stopped"] and abs(res["spent"] - 0.05) < 1e-9


def test_sample_many_keeps_models_in_separate_slices(tmp_path):
    seen = []

    def fn_for(name):
        def fn(model, prompt):
            seen.append((name, model))
            return {"text": "a", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}
        return fn

    sample_many({"m1": fn_for("m1"), "m2": fn_for("m2")}, mk_tasks(1), tmp_path,
                2, 100.0, price, log=lambda s: None)
    # make_completion_fn ignores its model argument, so each fn must only ever
    # be handed its own model's work.
    assert all(bound == asked for bound, asked in seen)
    assert ballot_counts(tmp_path) == {("t0", "m1"): 2, ("t0", "m2"): 2}

"""Resumable model x task sampler.

Two-phase design so partial/interrupted runs can be resumed without re-paying
for completions that already succeeded:

  Phase 1 (sampling, resumable): for every (model, task) pair not already
  present in the run's frozen-output store, call the model once via
  runner.run_one and append the frozen result. Pairs already frozen are
  skipped entirely -- no completion_fn call is made for them.

  Phase 2 (scoring, offline): re-read ALL frozen outputs for the run and
  score + price every one of them. This never calls a model; scoring and
  pricing are pure functions over already-frozen data, so re-running sample()
  on a fully-sampled run_dir is cheap and deterministic.
"""

from __future__ import annotations

from typing import Callable

from evaluator import pricing
from evaluator.report import ResultRow
from evaluator.runner import run_one
from evaluator.scorers import code, mcq
from evaluator.scorers import math as math_scorer
from evaluator.store import append_frozen, read_frozen
from evaluator.suite.types import Task

SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score,
           "livecodebench": code.score}


def sample(
    models: dict[str, Callable],
    tasks: list[Task],
    run_dir,
    scorers: dict[str, Callable] = SCORERS,
    cost_fn: Callable[[str, int, int], float] = pricing.cost,
) -> list[ResultRow]:
    """Run every (model, task) pair not yet frozen in run_dir, then score all
    frozen outputs (old and new) into ResultRow entries.

    Args:
        models: mapping of model_name -> completion_fn(model, prompt) -> dict.
        tasks: tasks to sample/score.
        run_dir: directory holding this run's frozen.jsonl (see store.py).
        scorers: source -> score(task, output_text) -> Score.
        cost_fn: model, in_tokens, out_tokens -> cost_usd. Pricing is always
            recomputed from cost_fn; the completion's own cost_usd (frozen
            for audit purposes) is never used for the returned rows.

    Returns:
        One ResultRow per frozen output currently in run_dir.
    """
    # Phase 1: resumable sampling.
    done = {(fo.task_id, fo.model) for fo in read_frozen(run_dir)}
    for model_name, fn in models.items():
        for task in tasks:
            if (task.id, model_name) in done:
                continue
            fo = run_one(task, model_name, fn)
            append_frozen(run_dir, fo)

    # Phase 2: offline scoring over everything frozen so far.
    task_by_id = {task.id: task for task in tasks}
    rows = []
    for fo in read_frozen(run_dir):
        task = task_by_id[fo.task_id]
        correct = fo.status == "ok" and scorers[task.source](task, fo.output_text).correct
        rows.append(ResultRow(
            task.id,
            task.source,
            fo.model,
            correct,
            cost_fn(fo.model, fo.in_tokens, fo.out_tokens),
        ))
    return rows

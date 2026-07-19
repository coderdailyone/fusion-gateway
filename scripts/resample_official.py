"""Budget-gated official-pipeline re-sample over all feasible models.

Real API calls -> run manually on an isolated host with keys in
runs/secrets/.env. build_prompt already emits official 0-shot-CoT prompts, so
run_budgeted re-samples the locked suite through the official pipeline,
consulting a preflight budget gate before every new (model, task) call.
Writes a NEW run dir; already-frozen (task, model) pairs are skipped, so a
run that stops early (budget ceiling reached) can be resumed by rerunning
against the same run_dir.
"""
from __future__ import annotations

from typing import Callable

FEASIBLE_MODELS = [
    "deepseek-chat", "claude-sonnet-5", "claude-opus-4-8",
    "gpt-5.6-sol", "gpt-5.5", "glm-5.2", "kimi-k3",
]
BUDGET_CEILING_USD = 35.0
WARN_FRACTION = 0.80


def estimate_cost(rows_est, cost_fn) -> float:
    return sum(cost_fn(m, i, o) for (m, i, o) in rows_est)


def budget_gate(spent: float, next_cost: float, ceiling: float) -> str:
    if spent + next_cost > ceiling:
        return "stop"
    if spent >= WARN_FRACTION * ceiling:
        return "warn"
    return "ok"


def run_budgeted(
    models: dict[str, Callable],
    tasks: list,
    run_dir,
    ceiling: float,
    cost_fn: Callable[[str, int, int], float],
    *,
    deps: dict | None = None,
) -> dict:
    """Resumable, budget-gated sampling loop over (model, task) pairs.

    Mirrors evaluator/sampler.py::sample's resume/skip logic (a (task, model)
    pair already present in run_dir's frozen store is never re-called), but
    additionally consults `budget_gate` before every new call and enforces a
    hard spend ceiling as a backstop.

    Args:
        models: model_name -> completion_fn(model, prompt) -> dict, same
            shape sample()/run_one() expect.
        tasks: tasks to sample.
        run_dir: this run's directory (frozen.jsonl lives here).
        ceiling: hard USD spend ceiling for this call to run_budgeted.
        cost_fn: model, in_tokens, out_tokens -> cost_usd (e.g. pricing.cost).
        deps: test seam. A dict optionally providing "run_one", "build_prompt",
            "read_frozen", "append_frozen" to replace the real
            evaluator.runner/evaluator.store functions with fakes. Any key
            omitted falls back to the real implementation. None (default)
            uses the real implementations throughout.

    Returns:
        {"spent": total USD spent (frozen-so-far + this call's real spend),
         "completed": number of NEW (model, task) pairs sampled this call,
         "stopped": True if the loop broke early on the budget gate or the
                    hard-ceiling backstop, False if every pending pair ran}.
    """
    real_deps = {}
    if deps is None or set(deps) < {"run_one", "build_prompt", "read_frozen", "append_frozen"}:
        from evaluator.runner import run_one as _run_one, build_prompt as _build_prompt
        from evaluator.store import read_frozen as _read_frozen, append_frozen as _append_frozen
        real_deps = {
            "run_one": _run_one,
            "build_prompt": _build_prompt,
            "read_frozen": _read_frozen,
            "append_frozen": _append_frozen,
        }
    deps = {**real_deps, **(deps or {})}
    run_one = deps["run_one"]
    build_prompt = deps["build_prompt"]
    read_frozen = deps["read_frozen"]
    append_frozen = deps["append_frozen"]

    frozen = read_frozen(run_dir)
    done = {(fo.task_id, fo.model) for fo in frozen}
    spent = sum(cost_fn(fo.model, fo.in_tokens, fo.out_tokens) for fo in frozen)

    pending = [
        (model_name, task)
        for model_name, fn in models.items()
        for task in tasks
        if (task.id, model_name) not in done
    ]

    n_done = 0
    stopped = False
    warned = False
    for model_name, task in pending:
        est_in = max(16, len(build_prompt(task)) // 4)
        est_out = 1024
        est_next = cost_fn(model_name, est_in, est_out)
        gate = budget_gate(spent, est_next, ceiling)

        if gate == "stop":
            remaining = len(pending) - n_done
            print(
                f"[budget] stopping before next call: spent=${spent:.4f} "
                f"ceiling=${ceiling:.2f}; {remaining} pair(s) left — "
                f"resumable, rerun to continue."
            )
            stopped = True
            break
        if gate == "warn" and not warned:
            print(
                f"[budget] warning: spent=${spent:.4f} has reached "
                f"{int(WARN_FRACTION*100)}% of ceiling ${ceiling:.2f}"
            )
            warned = True

        fn = models[model_name]
        fo = run_one(task, model_name, fn)
        append_frozen(run_dir, fo)
        n_done += 1
        spent += cost_fn(fo.model, fo.in_tokens, fo.out_tokens)

        if spent >= ceiling:
            print(
                f"[budget] hard ceiling reached: spent=${spent:.4f} >= "
                f"ceiling=${ceiling:.2f}; stopping — resumable, rerun to continue."
            )
            stopped = True
            break

    return {"spent": spent, "completed": n_done, "stopped": stopped}


def main() -> None:
    import sys
    from datetime import datetime, timezone
    from evaluator import validate, pricing
    from evaluator.pilot import stratified_subset
    from evaluator.store import new_run_dir
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher

    validate.load_secrets()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1063  # full locked suite
    model_names = [m for m in FEASIBLE_MODELS if m in validate.MODELS]
    models = {name: validate.MODELS[name]() for name in model_names}

    manifest = load("configs/suite.manifest.json")
    all_tasks = load_suite(manifest, {s.name: make_fetcher(s.name) for s in manifest.sources})
    tasks = stratified_subset(all_tasks, n, seed=1234)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir("evaluator", "official_resample", ts)
    print(f"re-sampling {len(tasks)} tasks x {len(models)} models -> {run_dir}")
    print(f"budget ceiling ${BUDGET_CEILING_USD}; warn at {int(WARN_FRACTION*100)}%")

    result = run_budgeted(models, tasks, run_dir, BUDGET_CEILING_USD, pricing.cost)
    print(
        f"resample done: completed={result['completed']} new pair(s), "
        f"spent=${result['spent']:.4f}, stopped_early={result['stopped']}"
    )


if __name__ == "__main__":
    main()

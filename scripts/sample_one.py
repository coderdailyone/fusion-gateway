#!/usr/bin/env python3
"""Sample ONE model over the locked suite, into its own run dir. (REAL API CALLS)

`scripts/resample_official.py` samples the whole feasible pool; this is the same
budget-gated, resumable loop pointed at a single model — used when a provider
ships a new tier and the question is only "how does this one score on the same
frozen suite everything else was scored on".

Resumable, but only if you point it at the SAME run dir: a (task, model) pair
already frozen there is never re-called. Passing a fresh dir re-samples and
re-pays from scratch, so a run stopped by the ceiling, by Ctrl-C, or by the
provider 503ing is continued by re-running with the dir the first invocation
printed.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sample_one.py <model> [n] [ceiling] [run_dir]

    model      a key of evaluator.validate.MODELS
    n          tasks from the locked suite (default 1063 = all)
    ceiling    hard spend ceiling in USD (default 5.00)
    run_dir    resume into this dir instead of creating a new one
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone


def main() -> int:
    from evaluator import pricing, validate
    from evaluator.pilot import stratified_subset
    from evaluator.store import new_run_dir
    from evaluator.suite.loader import load_suite
    from evaluator.suite.manifest import load
    from evaluator.hf_fetchers import make_fetcher
    from scripts.resample_official import run_budgeted

    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1063
    ceiling = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    validate.load_secrets()
    if name not in validate.MODELS:
        print(f"unknown model {name!r}; known: {sorted(validate.MODELS)}")
        return 2

    manifest = load("configs/suite.manifest.json")
    all_tasks = load_suite(manifest, {s.name: make_fetcher(s.name)
                                      for s in manifest.sources})
    # Same seed and helper the pool-wide resample uses, so the task set is
    # identical to what every other model was scored on.
    tasks = stratified_subset(all_tasks, n, seed=1234)

    if len(sys.argv) > 4:
        from pathlib import Path
        run_dir = Path(sys.argv[4])
        if not run_dir.exists():
            print(f"run_dir {run_dir} does not exist; omit it to create a new one")
            return 2
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = new_run_dir("evaluator", f"sample_{name}", ts)
    prices = pricing.load_prices()

    def cost_fn(model: str, in_tok: int, out_tok: int) -> float:
        return pricing.cost(model, in_tok, out_tok, prices)

    print(f"model={name}  tasks={len(tasks)}  ceiling=${ceiling}  run_dir={run_dir}")
    summary = run_budgeted({name: validate.MODELS[name]()}, tasks, run_dir,
                           ceiling, cost_fn)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

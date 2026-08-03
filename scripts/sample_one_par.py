#!/usr/bin/env python3
"""Sample ONE model over the locked suite, CONCURRENTLY. (REAL API CALLS)

`scripts/resample_official.py::run_budgeted` is serial: it awaits each call
before starting the next. That is fine for a fast model, but a reasoning model
that spends 9 s of wall clock per task turns a 1063-task suite into ~2.7 hours
of pure waiting — and measured on deepseek-v4-pro it was worse, ~7 hours, since
its long-tail tasks run to 145 s.

The bottleneck is entirely network wait, so this driver fans the calls out over
a thread pool while keeping the two things that must stay serial serial:

  * **writes** — `store.append_frozen` opens the file in "a" mode per row, so
    concurrent writers would interleave partial lines. One writer thread owns
    the file.
  * **spend accounting** — the ceiling is checked and the running total updated
    under a lock, so the budget cannot be overshot by N racing workers. Because
    calls already in flight cannot be un-spent, the ceiling is enforced at
    *dispatch* time: once the projected spend crosses it, no new task starts,
    and the in-flight ones are allowed to finish and be recorded rather than
    thrown away after being paid for.

Resume works exactly as the serial version: a (task, model) pair already frozen
in the run dir is never re-called, so pointing this at an existing dir tops it
up. That includes rows frozen with status != "ok" — a failed row is not retried,
same as the serial loop.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sample_one_par.py <model> [n] [ceiling] [run_dir] [workers]
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    from evaluator import pricing, validate
    from evaluator.pilot import stratified_subset
    from evaluator.runner import run_one
    from evaluator.store import append_frozen, read_frozen, new_run_dir
    from evaluator.suite.loader import load_suite
    from evaluator.suite.manifest import load
    from evaluator.hf_fetchers import make_fetcher

    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1063
    ceiling = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    run_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    validate.load_secrets()
    if name not in validate.MODELS:
        print(f"unknown model {name!r}")
        return 2

    manifest = load("configs/suite.manifest.json")
    all_tasks = load_suite(manifest, {s.name: make_fetcher(s.name)
                                      for s in manifest.sources})
    tasks = stratified_subset(all_tasks, n, seed=1234)

    if run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = new_run_dir("evaluator", f"sample_{name}", ts)
    elif not run_dir.exists():
        print(f"run_dir {run_dir} does not exist")
        return 2

    prices = pricing.load_prices()
    cost = lambda i, o: pricing.cost(name, i, o, prices)  # noqa: E731

    frozen = list(read_frozen(run_dir))
    done = {fo.task_id for fo in frozen if fo.model == name}
    spent = sum(cost(fo.in_tokens, fo.out_tokens) for fo in frozen)
    pending = [t for t in tasks if t.id not in done]
    print(f"model={name} workers={workers} pending={len(pending)} "
          f"already={len(done)} spent=${spent:.4f} ceiling=${ceiling} dir={run_dir}")

    fn = validate.MODELS[name]()
    lock = threading.Lock()
    state = {"spent": spent, "stopped": False, "done": 0}

    def work(task):
        # Dispatch gate: never START a call once the ceiling is reached. Calls
        # already in flight are still recorded — they were paid for.
        with lock:
            if state["stopped"] or state["spent"] >= ceiling:
                state["stopped"] = True
                return None
        return task, run_one(task, name, fn)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, t) for t in pending]
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except Exception as exc:              # never lose the whole run
                print(f"  worker error: {type(exc).__name__}: {exc}"[:160])
                continue
            if got is None:
                continue
            _task, fo = got
            with lock:                            # single writer + accounting
                append_frozen(run_dir, fo)
                state["spent"] += cost(fo.in_tokens, fo.out_tokens)
                state["done"] += 1
                if state["done"] % 25 == 0:
                    print(f"  {state['done']}/{len(pending)}  ${state['spent']:.4f}")

    print({"spent": round(state["spent"], 6), "completed": state["done"],
           "stopped": state["stopped"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

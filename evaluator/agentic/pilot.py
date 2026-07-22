"""Resumable, budget-gated driver over instances. One unit of work = one cascade.

Resume: any instance whose attempt is already frozen in run_dir is skipped.
Budget: cumulative spend is re-derived from the frozen attempts on disk each
call (plus any spent_so_far the caller threads in), so re-running to resume
after a transient error is SAFE BY DEFAULT — a fresh process never grants a
fresh full budget. If the next cascade's worst-case projected spend would
breach the ceiling, stop cleanly. Mirrors scripts/resample_official.run_budgeted.
"""
from __future__ import annotations

from evaluator.agentic.records import append_attempt, read_attempts
from evaluator.agentic.cascade import run_cascade

# Realistic worst-case per-instance projection for the gate (a cheap+strong
# cascade can cost on the order of dollars). Kept as a realistic default so the
# real $50 pilot stops with enough headroom; the unit test injects a small value.
WORST_CASE_PER_INSTANCE = 3.0


def run_pilot(instances, cheap_run, strong_run, verify, run_dir, ceiling,
              spent_so_far: float = 0.0,
              worst_case_per_instance: float = WORST_CASE_PER_INSTANCE):
    frozen = read_attempts(run_dir)
    done = {a.instance_id for a in frozen}
    # cumulative spend from disk -> a resumed/restarted process counts prior
    # spend against the ceiling instead of starting fresh.
    spent = spent_so_far + sum(a.cost_usd for a in frozen)
    results = []
    for inst in instances:
        if inst.instance_id in done:
            continue
        if spent + worst_case_per_instance > ceiling:
            break  # stop cleanly before risking a breach
        res = run_cascade(inst, cheap_run, strong_run, verify)
        append_attempt(run_dir, res.cheap)
        if res.strong is not None:
            append_attempt(run_dir, res.strong)
        spent += res.cost_usd
        results.append(res)
    return results

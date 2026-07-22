"""Resumable, budget-gated driver over instances. One unit of work = one cascade.

Resume: any instance whose attempt is already frozen in run_dir (matched by
instance_id via `read_attempts`) is skipped.
Budget: a hard ceiling backstop — before running the next cascade, if this
call's own projected spend (spent_so_far plus what this call has spent, plus a
conservative WORST_CASE_PER_INSTANCE margin) would breach the ceiling, stop
cleanly (frozen work stays intact and re-gradeable). Each call gets a fresh
runway: work already frozen by an earlier call is skipped, not re-charged
against this call's ceiling — pass `spent_so_far` explicitly to carry a
cumulative total across calls. Mirrors scripts/resample_official.run_budgeted's
resume/skip + ceiling-backstop discipline.
"""
from __future__ import annotations

from evaluator.agentic.records import append_attempt, read_attempts
from evaluator.agentic.cascade import run_cascade

# Conservative per-instance cost margin for the gate: how much headroom under
# the ceiling this call must have before it dares start one more cascade.
WORST_CASE_PER_INSTANCE = 0.5


def run_pilot(instances, cheap_run, strong_run, verify, run_dir, ceiling,
              spent_so_far: float = 0.0):
    done = {a.instance_id for a in read_attempts(run_dir)}
    spent = spent_so_far
    results = []
    for inst in instances:
        if inst.instance_id in done:
            continue
        if spent + WORST_CASE_PER_INSTANCE > ceiling:
            break  # stop cleanly before risking a breach
        res = run_cascade(inst, cheap_run, strong_run, verify)
        append_attempt(run_dir, res.cheap)
        if res.strong is not None:
            append_attempt(run_dir, res.strong)
        spent += res.cost_usd
        results.append(res)
    return results

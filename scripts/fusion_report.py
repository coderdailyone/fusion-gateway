"""M5 fusion metrics: oracle ceiling, fix/break, reviewer agreement, gate curve.

Reuses scripts/hard_report.wilson_ci and mcnemar_p for the report rendering.
"""
from __future__ import annotations

from scripts.hard_report import mcnemar_p, wilson_ci  # noqa: F401  (used by main)


def oracle(correct_by_model: dict[str, dict[str, bool]]) -> float:
    """Fraction of tasks where AT LEAST ONE model is correct (fusion's ceiling)."""
    task_ids: set[str] = set()
    for per_task in correct_by_model.values():
        task_ids |= set(per_task)
    if not task_ids:
        return 0.0
    hit = sum(1 for t in task_ids
              if any(per_task.get(t, False) for per_task in correct_by_model.values()))
    return hit / len(task_ids)


def fix_break(baseline: dict[str, bool], fused: dict[str, bool]) -> dict:
    """fix = baseline wrong -> fused right; break_ = baseline right -> fused wrong."""
    ids = [t for t in baseline if t in fused]
    fix = sum(1 for t in ids if not baseline[t] and fused[t])
    brk = sum(1 for t in ids if baseline[t] and not fused[t])
    return {"fix": fix, "break_": brk, "net": fix - brk, "n": len(ids)}


def reviewer_agreement(reviews_by_task: dict[str, dict[str, dict[str, str]]]) -> float:
    """Of (task, target) pairs judged by >=2 reviewers, fraction where all agree."""
    pairs = 0
    agree = 0
    for _tid, by_reviewer in reviews_by_task.items():
        targets: dict[str, list[str]] = {}
        for _reviewer, verdicts in by_reviewer.items():
            for target, verdict in verdicts.items():
                targets.setdefault(target, []).append(verdict)
        for _target, verdicts in targets.items():
            if len(verdicts) < 2:
                continue
            pairs += 1
            if len(set(verdicts)) == 1:
                agree += 1
    return (agree / pairs) if pairs else float("nan")


def gate_curve(candidates_by_task: dict[str, dict[str, str]],
               fused_correct: dict[str, bool],
               baseline_correct: dict[str, bool],
               fusion_cost: dict[str, float]) -> list[dict]:
    """Simulate gate policies over frozen results ($0).

    - "always": fuse every task.
    - "on_disagreement": if all candidate answers are identical, adopt that
      answer for free (its correctness equals the baseline's on that task);
      otherwise pay for fusion.
    """
    ids = sorted(fused_correct)
    rows = []
    always_cost = sum(fusion_cost.get(t, 0.0) for t in ids)
    rows.append({"policy": "always",
                 "cost": always_cost,
                 "correct": sum(1 for t in ids if fused_correct[t]),
                 "n": len(ids)})
    cost = 0.0
    correct = 0
    for t in ids:
        answers = set(candidates_by_task.get(t, {}).values())
        if len(answers) <= 1:                     # unanimous -> free adopt
            correct += 1 if baseline_correct.get(t, False) else 0
        else:
            cost += fusion_cost.get(t, 0.0)
            correct += 1 if fused_correct[t] else 0
    rows.append({"policy": "on_disagreement", "cost": cost, "correct": correct,
                 "n": len(ids)})
    return rows

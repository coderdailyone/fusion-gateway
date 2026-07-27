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


def reviewer_agreement(reviews_by_task: dict[str, dict[str, dict[str, object]]]) -> float:
    """Of (task, target) pairs judged by >=2 reviewers, fraction where all agree.

    `verdicts` values may be either plain "correct"/"wrong"/"unsure" strings or
    `evaluator.fusion.review.Verdict` dataclass instances (the real shape
    `cross_review` returns). Verdict is a hashable frozen dataclass of
    (verdict, reason), so comparing raw Verdict objects compares the free-text
    `reason` too — two reviewers who agree "wrong" for different reasons would
    never compare equal, silently collapsing agreement toward 0. Normalize to
    the bare verdict string before comparing.
    """
    pairs = 0
    agree = 0
    for _tid, by_reviewer in reviews_by_task.items():
        targets: dict[str, list[str]] = {}
        for _reviewer, verdicts in by_reviewer.items():
            for target, verdict in verdicts.items():
                targets.setdefault(target, []).append(getattr(verdict, "verdict", verdict))
        for _target, verdicts in targets.items():
            if len(verdicts) < 2:
                continue
            pairs += 1
            if len(set(verdicts)) == 1:
                agree += 1
    return (agree / pairs) if pairs else float("nan")


def _is_unanimous(texts: dict[str, str], extract) -> bool:
    """True iff every candidate text in `texts` agrees.

    With `extract` given, agreement is judged on `extract(text)` (the parsed
    answer) rather than the raw chain-of-thought text — three different
    models essentially never produce byte-identical prose even when they
    agree on the answer, which would otherwise make unanimity vacuous. A
    `None` extraction (unparseable candidate) is given its own distinct
    sentinel value per occurrence so that e.g. two unparseable candidates
    never count as "agreeing by virtue of both failing to parse".
    """
    if extract is None:
        return len(set(texts.values())) <= 1
    values = set()
    for text in texts.values():
        v = extract(text)
        values.add(v if v is not None else object())
    return len(values) <= 1


def gate_curve(candidates_by_task: dict[str, dict[str, str]],
               fused_correct: dict[str, bool],
               baseline_correct: dict[str, bool],
               fusion_cost: dict[str, float],
               extract=None,
               review_cost: dict[str, float] | None = None) -> list[dict]:
    """Simulate gate policies over frozen results ($0).

    - "always": fuse every task (and, if `review_cost` is given, also pays
      the review cost of every task — cross-review always runs regardless of
      whether candidates agree).
    - "on_disagreement": if all candidate answers are unanimous, adopt that
      answer for free (its correctness equals the baseline's on that task);
      otherwise pay for fusion. Candidate agreement is knowable at $0 straight
      from the frozen candidates, before any review call is made, so this
      policy only charges `review_cost` for the tasks it actually fuses.

    `extract`: optional `str -> str | None` used to decide unanimity on the
    PARSED answer instead of raw candidate text (see `_is_unanimous`). When
    `None`, falls back to raw-text comparison.
    """
    ids = sorted(fused_correct)
    rows = []
    always_cost = sum(fusion_cost.get(t, 0.0) for t in ids)
    if review_cost:
        always_cost += sum(review_cost.get(t, 0.0) for t in ids)
    rows.append({"policy": "always",
                 "cost": always_cost,
                 "correct": sum(1 for t in ids if fused_correct[t]),
                 "n": len(ids)})
    cost = 0.0
    correct = 0
    for t in ids:
        texts = candidates_by_task.get(t, {})
        if _is_unanimous(texts, extract):          # unanimous -> free adopt
            correct += 1 if baseline_correct.get(t, False) else 0
        else:
            cost += fusion_cost.get(t, 0.0)
            if review_cost:
                cost += review_cost.get(t, 0.0)
            correct += 1 if fused_correct[t] else 0
    rows.append({"policy": "on_disagreement", "cost": cost, "correct": correct,
                 "n": len(ids)})
    return rows

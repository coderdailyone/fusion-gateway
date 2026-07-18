"""Human-eyeball disagreement audit (no LLM judge).

Surfaces candidate scorer false-negatives for manual spot-check: tasks a model
got 'wrong' while at least one other model got them right (the
``wrong_but_others_right`` kind). Ranks by how many other models were right
(higher = more suspicious). Works purely off ``ResultRow`` aggregation plus the
frozen output text — no model calls, no re-scoring.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from evaluator.report import ResultRow


@dataclass(frozen=True)
class DisagreementCase:
    task_id: str
    source: str
    model: str
    kind: str
    detail: dict


def find_cases(rows: list[ResultRow],
               frozen_by_key: dict[tuple, str]) -> list[DisagreementCase]:
    by_task: dict[str, list[ResultRow]] = defaultdict(list)
    for r in rows:
        by_task[r.task_id].append(r)

    cases: list[DisagreementCase] = []
    for task_id, task_rows in by_task.items():
        n_right = sum(1 for r in task_rows if r.correct)
        for r in task_rows:
            if not r.correct and n_right > 0:
                cases.append(DisagreementCase(
                    task_id, r.source, r.model, "wrong_but_others_right",
                    {"n_other_right": n_right,
                     "output": frozen_by_key.get((task_id, r.model), "")[:400]}))
    cases.sort(key=lambda c: (-c.detail.get("n_other_right", 0), c.task_id, c.model))
    return cases


def render(cases: list[DisagreementCase]) -> str:
    lines = ["## Disagreement audit (human spot-check)", ""]
    lines.append(f"total cases: {len(cases)}")
    lines.append("")
    lines.append("| task_id | source | model | kind | n_other_right |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in cases:
        lines.append(f"| {c.task_id} | {c.source} | {c.model} | {c.kind} | "
                     f"{c.detail.get('n_other_right', '')} |")
    return "\n".join(lines) + "\n"

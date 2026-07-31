"""Assemble per-task candidate answers from frozen single-model runs ($0).

Candidates come from the frozen M2c/M2d outputs, so building a panel costs
nothing; only cross-review and fusion make new API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PanelCase:
    task_id: str
    source: str
    candidates: dict[str, str]  # model name -> its answer text


def load_frozen_by_model(run_dirs: dict[str, str]) -> dict[str, dict[str, str]]:
    """model -> {task_id: output_text} for status=="ok" rows only."""
    from evaluator.store import read_frozen

    out: dict[str, dict[str, str]] = {}
    for model, run_dir in run_dirs.items():
        got: dict[str, str] = {}
        for fo in read_frozen(Path(run_dir)):
            if fo.status == "ok" and fo.task_id not in got:
                got[fo.task_id] = fo.output_text
        out[model] = got
    return out


def assemble(frozen_by_model: dict[str, dict[str, str]], task_ids: list[str],
             min_candidates: int = 2,
             sources: dict[str, str] | None = None) -> tuple[list[PanelCase], list[str]]:
    """Build one PanelCase per task from whatever candidates exist.

    A task with fewer than `min_candidates` answers is excluded (reported), not
    an error — models legitimately error or run out of quota on some tasks.
    """
    cases: list[PanelCase] = []
    excluded: list[str] = []
    for tid in task_ids:
        cands = {m: texts[tid] for m, texts in frozen_by_model.items() if tid in texts}
        if len(cands) < min_candidates:
            excluded.append(tid)
            continue
        cases.append(PanelCase(task_id=tid,
                               source=(sources or {}).get(tid, ""),
                               candidates=cands))
    return cases, excluded

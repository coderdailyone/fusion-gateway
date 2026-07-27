"""The fuser: reads task + all candidates + cross-review, writes the final answer.

The fuser is a DOMESTIC model (default glm-5.2) — deliberately not the strongest
candidate, so fusion cannot degenerate into "always echo deepseek". Using a
frontier model here would make the headline claim circular.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.fusion.prompts import build_fusion_prompt


@dataclass(frozen=True)
class FusionResult:
    task_id: str
    answer: str
    fuser: str
    cost_usd: float
    status: str          # "ok" | "error"
    error: str | None


def fuse(task, case, reviews, completion_fn, fuser: str = "glm-5.2") -> FusionResult:
    prompt = build_fusion_prompt(task, case, reviews)
    try:
        got = completion_fn(fuser, prompt)
    except Exception as exc:  # never crash a batch on one task
        return FusionResult(case.task_id, "", fuser, 0.0, "error", str(exc))
    return FusionResult(case.task_id, got.get("text", "") or "", fuser,
                        float(got.get("cost_usd", 0.0) or 0.0), "ok", None)

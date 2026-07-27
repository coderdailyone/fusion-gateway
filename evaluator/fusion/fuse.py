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
    """Call `completion_fn(fuser, prompt)` and wrap the result as a FusionResult.

    completion_fn CONTRACT (see also `evaluator/fusion/review.py` module
    docstring): `completion_fn(model, prompt)` MUST dispatch on the `model`
    name argument — it must actually call the model named `model`. This
    matters here specifically: `evaluator.validate.make_completion_fn` builds
    a closure bound to ONE litellm model that IGNORES the `model` argument, so
    passing e.g. `MODELS["glm-5.2"]()` directly (instead of a name-dispatching
    wrapper built from `MODELS`) would silently call the same bound model
    regardless of the `fuser` value passed in. Phase B must build a dispatcher
    such as `lambda model, prompt: fns[model](model, prompt)` from `MODELS`
    and pass that as `completion_fn`.
    """
    prompt = build_fusion_prompt(task, case, reviews)
    try:
        got = completion_fn(fuser, prompt)
    except Exception as exc:  # never crash a batch on one task
        return FusionResult(case.task_id, "", fuser, 0.0, "error", str(exc))
    return FusionResult(case.task_id, got.get("text", "") or "", fuser,
                        float(got.get("cost_usd", 0.0) or 0.0), "ok", None)

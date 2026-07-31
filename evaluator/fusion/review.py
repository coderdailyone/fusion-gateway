"""Cross-review: each panel model judges the OTHER candidates.

Output is structured (VERDICT lines), not prose, so the fuser receives evidence
and reviewer agreement is measurable. Reviews inform fusion only — grading stays
objective (evaluator/scorers/*). Malformed lines are dropped, never fatal.

completion_fn CONTRACT (binding on every caller of `cross_review` and of
`fuse` in `evaluator/fusion/fuse.py`): `completion_fn(model, prompt)` MUST
dispatch on the `model` name argument — i.e. it must actually call the model
named `model`, not some other model it happens to be bound to. This matters
because `evaluator.validate.make_completion_fn` builds a closure bound to ONE
litellm model that IGNORES the `model` argument entirely. So a naive Phase B
wiring such as `MODELS["glm-5.2"]()` passed directly as `completion_fn` would
make every "reviewer" actually be glm-5.2 reviewing its own answer — silently
violating no-self-review at the model level, even though `cross_review`
correctly excludes the reviewer's own candidate text from the prompt. Phase B
must instead build a name-dispatching wrapper from `MODELS`, e.g.:

    fns = {name: MODELS[name]() for name in needed_models}
    dispatch = lambda model, prompt: fns[model](model, prompt)

and pass `dispatch` (not a single bound `make_completion_fn` result) as the
`completion_fn` argument to both `cross_review` and `fuse`.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.fusion.prompts import build_review_prompt

VALID = {"correct", "wrong", "unsure"}


@dataclass(frozen=True)
class Verdict:
    verdict: str  # "correct" | "wrong" | "unsure"
    reason: str


def parse_review(text: str, valid_targets: set[str]) -> dict[str, Verdict]:
    out: dict[str, Verdict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "VERDICT":
            continue
        target, verdict = parts[1], parts[2].lower()
        if target not in valid_targets or verdict not in VALID:
            continue
        out[target] = Verdict(verdict, parts[3] if len(parts) > 3 else "")
    return out


def cross_review(task, case, completion_fn) -> tuple[dict[str, dict[str, Verdict]], float]:
    """Return (reviews, review_cost_usd): reviewer -> {target: Verdict}, and the
    summed cost_usd of every review call made (missing/None cost_usd counts as
    0.0). Each model reviews only the others.

    `completion_fn(model, prompt)` MUST dispatch on `model` — see the
    module-level CONTRACT note above. Callers building the cost side of a
    budget gate (e.g. `scripts/fusion_report.gate_curve`) need this returned
    cost: with a 3-model panel there are 3 review calls per task versus 1
    fusion call, so review spend dominates and must not be discarded.
    """
    reviews: dict[str, dict[str, Verdict]] = {}
    total_cost = 0.0
    for reviewer in sorted(case.candidates):
        targets = {m for m in case.candidates if m != reviewer}
        if not targets:
            continue
        prompt = build_review_prompt(task, case, reviewer=reviewer)
        try:
            got = completion_fn(reviewer, prompt)
            text = got["text"]
        except Exception:
            reviews[reviewer] = {}
            continue
        total_cost += float(got.get("cost_usd", 0.0) or 0.0)
        reviews[reviewer] = parse_review(text, targets)
    return reviews, total_cost

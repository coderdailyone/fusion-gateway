"""Cross-review: each panel model judges the OTHER candidates.

Output is structured (VERDICT lines), not prose, so the fuser receives evidence
and reviewer agreement is measurable. Reviews inform fusion only — grading stays
objective (evaluator/scorers/*). Malformed lines are dropped, never fatal.
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


def cross_review(task, case, completion_fn) -> dict[str, dict[str, Verdict]]:
    """reviewer -> {target: Verdict}. Each model reviews only the others."""
    reviews: dict[str, dict[str, Verdict]] = {}
    for reviewer in sorted(case.candidates):
        targets = {m for m in case.candidates if m != reviewer}
        if not targets:
            continue
        prompt = build_review_prompt(task, case, reviewer=reviewer)
        try:
            text = completion_fn(reviewer, prompt)["text"]
        except Exception:
            reviews[reviewer] = {}
            continue
        reviews[reviewer] = parse_review(text, targets)
    return reviews

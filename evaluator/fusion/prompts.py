"""Prompts for cross-review and fusion.

The answer-format sentences are copied verbatim from
`evaluator/official/prompts.py` so a fused answer satisfies exactly the
extractor the official grader uses. If those templates change, change these.
"""
from __future__ import annotations

_MCQ_FMT = ("Finish your response with a single line 'The answer is (X).' "
            "where X is the correct option letter.")
_MATH_FMT = "Put your final answer within \\boxed{}."
_CODE_FMT = "Respond with a single Python code block containing the complete solution."
_DEFAULT_FMT = "Put your final answer clearly at the end."

_FORMAT = {"mmlu_pro": _MCQ_FMT, "gpqa_diamond": _MCQ_FMT,
           "math": _MATH_FMT, "aime": _MATH_FMT, "math_l5": _MATH_FMT,
           "humaneval": _CODE_FMT, "livecodebench": _CODE_FMT}


def format_instruction(task) -> str:
    return _FORMAT.get(task.source, _DEFAULT_FMT)


def _candidate_block(candidates: dict[str, str], exclude: str | None = None) -> str:
    parts = []
    for model, text in sorted(candidates.items()):
        if model == exclude:
            continue
        parts.append(f"--- Candidate {model} ---\n{text}")
    return "\n\n".join(parts)


def build_review_prompt(task, case, reviewer: str) -> str:
    """Ask `reviewer` to judge the OTHER candidates (never its own answer)."""
    return (
        "You are reviewing other models' answers to a problem. For EACH candidate "
        "below, judge whether its final answer is correct.\n\n"
        f"Problem:\n{task.problem}\n\n"
        f"{_candidate_block(case.candidates, exclude=reviewer)}\n\n"
        "For each candidate, output one line in exactly this format:\n"
        "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>\n"
        "Judge only correctness of the final answer, not style."
    )


def build_fusion_prompt(task, case, reviews) -> str:
    """Fuser sees every candidate plus the cross-review evidence."""
    lines = []
    for reviewer, verdicts in sorted(reviews.items()):
        for target, v in sorted(verdicts.items()):
            lines.append(f"{reviewer} says {target} is {v.verdict}: {v.reason}")
    review_block = "\n".join(lines) if lines else "(no reviews available)"
    return (
        "Several models answered the problem below, and reviewed each other. "
        "Produce the single best final answer.\n\n"
        f"Problem:\n{task.problem}\n\n"
        f"{_candidate_block(case.candidates)}\n\n"
        f"--- Peer review ---\n{review_block}\n\n"
        "Rules:\n"
        "- If the candidates agree and no review objects, adopt that answer.\n"
        "- If they disagree, decide using the specific objections raised, and "
        "write a corrected answer (you may combine correct parts of several).\n"
        f"- {format_instruction(task)}"
    )

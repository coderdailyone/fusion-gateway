"""Prompts for cross-review and fusion.

The answer-format sentences are copied verbatim from
`evaluator/official/prompts.py` so a fused answer satisfies exactly the
extractor the official grader uses. If those templates change, change these.
"""
from __future__ import annotations

_MCQ_FMT = ("Finish your response with a single line 'The answer is (X).' "
            "where X is the correct option letter.")
_MATH_FMT = "Put your final answer within \\boxed{}."
_HUMANEVAL_FMT = "Respond with a single Python code block containing the full function definition."
_LIVECODEBENCH_FMT = "Respond with a single Python code block containing the complete solution."
_DEFAULT_FMT = "Put your final answer clearly at the end."

_FORMAT = {"mmlu_pro": _MCQ_FMT, "gpqa_diamond": _MCQ_FMT,
           "math": _MATH_FMT, "aime": _MATH_FMT, "math_l5": _MATH_FMT,
           "humaneval": _HUMANEVAL_FMT, "livecodebench": _LIVECODEBENCH_FMT}

_MCQ_SOURCES = {"mmlu_pro", "gpqa_diamond"}

# The official MCQ extractor (evaluator/official/mmlu_extract.py) takes the
# FIRST occurrence of "answer is" in the text. The fusion prompt below asks
# the fuser to "decide using the specific objections raised", which invites
# restating a rejected candidate's answer while explaining why it's wrong
# (e.g. "Candidate deepseek-chat's answer is (A), but that is wrong ... The
# answer is (B)."). That extracts (A) and scores a correct fusion WRONG. So
# for MCQ-format tasks, the fusion prompt must explicitly forbid the phrase
# anywhere except the final answer line.
_MCQ_NO_EARLY_ANSWER_IS_RULE = (
    "- Do not write the phrase \"answer is\" anywhere in your response except "
    "on the final answer line. The grader extracts the FIRST occurrence of "
    "that phrase, so writing it earlier (e.g. while explaining why a "
    "candidate's answer is wrong) would be mis-scored as your final answer."
)


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
    # The dominant failure mode measured on the standard tier was `break` —
    # fusion talking itself out of an answer the panel already had right (26 of
    # 1055 tasks). The majority rule below is deliberately blunt: when the panel
    # already agrees, copying is strictly better than rewriting.
    rules = [
        "- If a majority of the candidates give the SAME final answer and no "
        "review calls it wrong, COPY that answer verbatim. Do not rewrite, "
        "reword, or 'improve' it — copying is the correct action here.",
        "- Only depart from the majority answer when a review identifies a "
        "concrete error in it.",
        "- If they disagree, decide using the specific objections raised, and "
        "write a corrected answer (you may combine correct parts of several).",
    ]
    if task.source in _MCQ_SOURCES:
        rules.append(_MCQ_NO_EARLY_ANSWER_IS_RULE)
    rules.append(f"- {format_instruction(task)}")
    return (
        "Several models answered the problem below, and reviewed each other. "
        "Produce the single best final answer.\n\n"
        f"Problem:\n{task.problem}\n\n"
        f"{_candidate_block(case.candidates)}\n\n"
        f"--- Peer review ---\n{review_block}\n\n"
        "Rules:\n" + "\n".join(rules)
    )

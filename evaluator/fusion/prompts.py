"""Build review prompts for cross-review task.

MINIMAL implementation: prompt includes problem, other candidates' answers,
and instructs reviewer to emit VERDICT lines. Task 2 will expand this file
with domain-specific fusion prompts.
"""


def build_review_prompt(task, case, reviewer):
    """Build prompt for reviewer to judge OTHER candidates' answers.

    Args:
        task: Task(id, source, problem, answer, tests, meta)
        case: PanelCase(task_id, source, candidates: dict[model -> answer_text])
        reviewer: model name of the reviewer (excluded from review targets)

    Returns:
        str: prompt that instructs reviewer to emit VERDICT lines
    """
    # Collect other candidates (everyone except the reviewer)
    others = {m: ans for m, ans in case.candidates.items() if m != reviewer}

    lines = [
        f"Problem:\n{task.problem}\n",
        "Candidate answers:",
    ]

    for model, answer in sorted(others.items()):
        lines.append(f"  {model}: {answer}")

    lines.extend([
        "",
        "Review each candidate. For each, emit a line:",
        "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>",
        "",
        "Possible verdicts: 'correct' if the answer is right, 'wrong' if incorrect, 'unsure' if unclear.",
    ])

    return "\n".join(lines)

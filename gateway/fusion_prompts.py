"""Prompt construction and verdict parsing for the online fusion panel.

Ported from `evaluator/fusion/prompts.py` and the parsing half of
`evaluator/fusion/review.py`. Pure: no IO, no network, no gateway imports, so
every rule below is unit-testable without a server.

DELIBERATELY NOT PORTED: the per-benchmark answer-format machinery. Those
exist to satisfy official graders' extractors for specific benchmarks and
answer formats. A chat gateway has no grader and no benchmark source, so those
instructions would only corrupt ordinary answers.

KEPT VERBATIM, because M5 measured them working: the structured VERDICT line
(which took reviewer agreement from 0.63-0.74 to 0.9157), no self-review, and
the majority-copy rule (the countermeasure to `break` -- fusion talking itself
out of an answer the panel already had right, 26 of 1055 tasks).
"""
from __future__ import annotations

from dataclasses import dataclass

VALID_VERDICTS = {"correct", "wrong", "unsure"}


@dataclass(frozen=True)
class Verdict:
    verdict: str  # "correct" | "wrong" | "unsure"
    reason: str


def _as_text(value) -> str:
    """Coerce an upstream/client-supplied value to text without raising.

    `app.py` calls `request.json()` with no validation, so `messages` can hold
    any JSON shape a client cares to send.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # OpenAI content parts: [{"type": "text", "text": "..."}, ...]
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value)


def render_conversation(messages) -> str:
    """Render OpenAI `messages` as a transcript for the review/fusion prompts.

    The candidates receive the client's `messages` verbatim; only the reviewer
    and the fuser need the conversation as text.
    """
    if not isinstance(messages, list):
        return ""
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _as_text(m.get("role")) or "user"
        content = _as_text(m.get("content"))
        if not content:
            continue
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _candidate_block(candidates: dict[str, str], exclude: str | None = None) -> str:
    parts = []
    for model, text in sorted(candidates.items()):
        if model == exclude:
            continue
        parts.append(f"--- Candidate {model} ---\n{text}")
    return "\n\n".join(parts)


def build_review_prompt(conversation: str, candidates: dict[str, str],
                        reviewer: str) -> str:
    """Ask `reviewer` to judge the OTHER candidates -- never its own answer."""
    return (
        "You are reviewing other models' answers to the conversation below. "
        "For EACH candidate, judge whether its answer is correct and "
        "responsive.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"{_candidate_block(candidates, exclude=reviewer)}\n\n"
        "For each candidate, output one line in exactly this format:\n"
        "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>\n"
        "Judge only correctness, not style. Output nothing else."
    )


def build_fusion_prompt(conversation: str, candidates: dict[str, str],
                        reviews: dict[str, dict[str, Verdict]]) -> str:
    """The fuser sees every candidate plus the cross-review evidence."""
    lines = []
    for reviewer, verdicts in sorted(reviews.items()):
        for target, v in sorted(verdicts.items()):
            lines.append(f"{reviewer} says {target} is {v.verdict}: {v.reason}")
    review_block = "\n".join(lines) if lines else "(no reviews available)"
    rules = [
        "- If a majority of the candidates give the SAME answer and no review "
        "calls it wrong, COPY that answer verbatim. Do not rewrite, reword, or "
        "'improve' it -- copying is the correct action here.",
        "- Only depart from the majority answer when a review identifies a "
        "concrete error in it.",
        "- If they disagree, decide using the specific objections raised, and "
        "write a corrected answer (you may combine correct parts of several).",
        "- Reply with the answer itself. Do not mention the candidates, the "
        "reviews, or that several models were consulted.",
    ]
    return (
        "Several models answered the conversation below, and reviewed each "
        "other. Produce the single best final answer.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"{_candidate_block(candidates)}\n\n"
        f"--- Peer review ---\n{review_block}\n\n"
        "Rules:\n" + "\n".join(rules)
    )


def parse_review(text, valid_targets: set[str]) -> dict[str, Verdict]:
    """Extract VERDICT lines. Malformed lines are dropped, never fatal."""
    out: dict[str, Verdict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "VERDICT":
            continue
        target, verdict = parts[1], parts[2].lower()
        if target not in valid_targets or verdict not in VALID_VERDICTS:
            continue
        out[target] = Verdict(verdict, parts[3] if len(parts) > 3 else "")
    return out

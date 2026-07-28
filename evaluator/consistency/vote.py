"""Aggregate ballots into one answer using OBJECTIVE rules only.

Plurality everywhere, with two source-specific twists:
  * math  — equivalent answers are merged before counting (is_equiv), so
            "0.5" and "\\frac{1}{2}" do not split the vote.
  * code  — a sample that PASSES the problem's public doctests beats any number
            of failing samples. Execution outranks popularity.
A genuine tie is reported (tied=True) for the caller's LLM tie-break; it is
never resolved here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

_MATH = {"math", "aime", "math_l5"}
_CODE = {"humaneval", "livecodebench"}


@dataclass(frozen=True)
class Ballot:
    model: str
    key: str | None      # None => spoiled
    text: str


@dataclass(frozen=True)
class VoteResult:
    winner_text: str | None
    winner_key: str | None
    # PRE-filter counts: for code tasks this still holds the FAIL keys, so it is
    # NOT the tally the winner was drawn from -- do not derive "did the winner
    # have a majority?" from it.
    tally: dict[str, int]
    tied: bool
    spoiled: int
    n: int


def tally_keys(task, ballots) -> dict[str, int]:
    """Count keys; for math, merge equivalent answers into a single candidate."""
    valid = [b for b in ballots if b.key is not None]
    if task.source not in _MATH:
        return dict(Counter(b.key for b in valid))
    merged: dict[str, int] = {}
    from evaluator.official.math_grade import is_equiv

    for b in valid:
        for existing in merged:
            if is_equiv(existing, b.key):
                merged[existing] += 1
                break
        else:
            merged[b.key] = 1
    return merged


def first_k(ballots, k: int):
    """First k ballots PER MODEL — the yield curve compares equal budgets."""
    seen: dict[str, int] = {}
    out = []
    for b in ballots:
        c = seen.get(b.model, 0)
        if c < k:
            out.append(b)
            seen[b.model] = c + 1
    return out


def vote(task, ballots) -> VoteResult:
    n = len(ballots)
    spoiled = sum(1 for b in ballots if b.key is None)
    tally = tally_keys(task, ballots)
    if not tally:
        return VoteResult(None, None, {}, False, spoiled, n)

    if task.source in _CODE:
        # The colon matters: doctest_signature emits "PASS:<n>" / "FAIL:<...>",
        # but ballot_key falls back to the RAW SOURCE whenever a problem has no
        # ">>>" doctests -- which is every livecodebench problem. Without the
        # colon a sample starting with e.g. "PASS_THRESHOLD = 3" would be
        # counted as a verified passing run and outrank a real plurality.
        passing = {k: v for k, v in tally.items() if k.startswith("PASS:")}
        if passing:
            tally_for_pick = passing
        else:
            tally_for_pick = tally
    else:
        tally_for_pick = tally

    ordered = sorted(tally_for_pick.items(), key=lambda kv: -kv[1])
    top_count = ordered[0][1]
    winners = [k for k, v in ordered if v == top_count]
    tied = len(winners) > 1
    winner_key = winners[0]
    winner_text = next(
        (b.text for b in ballots
         if b.key == winner_key or (task.source in _MATH and _equiv(b.key, winner_key))),
        None)
    return VoteResult(winner_text, winner_key, tally, tied, spoiled, n)


def _equiv(a, b) -> bool:
    if a is None or b is None:
        return False
    from evaluator.official.math_grade import is_equiv

    return is_equiv(a, b)

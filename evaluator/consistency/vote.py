"""Aggregate ballots into one answer using OBJECTIVE rules only.

Plurality everywhere, with two source-specific twists:
  * math  — equivalent answers are merged before counting (is_equiv), so
            "0.5" and "\\frac{1}{2}" do not split the vote.
  * code  — a sample that PASSES the problem's public doctests beats any number
            of failing samples. Execution outranks popularity.
A genuine tie is reported (tied=True) for the caller's LLM tie-break; it is
never resolved here. The placeholder winner returned alongside tied=True is the
lexicographically smallest top-count key, so it does not depend on the order the
ballots were listed in.
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
    # The two fields below describe the tally the winner was ACTUALLY drawn
    # from -- i.e. after the "PASS:" filter for code -- so a caller can compute
    # agreement (winner_count / effective_n) without reimplementing that filter.
    winner_count: int            # count of winner_key in that tally
    effective_n: int             # non-spoiled ballots that tally was computed over
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
        return VoteResult(winner_text=None, winner_key=None, tally={},
                          winner_count=0, effective_n=0, tied=False,
                          spoiled=spoiled, n=n)

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

    top_count = max(tally_for_pick.values())
    winners = sorted(k for k, v in tally_for_pick.items() if v == top_count)
    tied = len(winners) > 1
    # min(winners), NOT the first key encountered. Sorting by count alone is
    # stable, so "first" meant "whichever model happened to be listed first" --
    # a zero-signal variable. On the frozen HumanEval data 51.2% of tasks tie,
    # and permuting only the ballot order swung vote accuracy 149/164 -> 160/164.
    # The caller still routes tied=True to the fuser; this only makes the
    # placeholder winner deterministic and model-independent.
    winner_key = winners[0]
    winner_text = next(
        (b.text for b in ballots
         if b.key == winner_key or (task.source in _MATH and _equiv(b.key, winner_key))),
        None)
    return VoteResult(winner_text=winner_text, winner_key=winner_key,
                      tally=tally, winner_count=top_count,
                      effective_n=sum(tally_for_pick.values()),
                      tied=tied, spoiled=spoiled, n=n)


def _equiv(a, b) -> bool:
    if a is None or b is None:
        return False
    from evaluator.official.math_grade import is_equiv

    return is_equiv(a, b)

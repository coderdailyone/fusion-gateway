"""Turn a model answer into a comparable ballot key.

One rule per source:
  mmlu_pro / gpqa_diamond -> the extracted option letter
  math / aime / math_l5   -> the extracted answer string (equivalent strings are
                             merged at tally time by vote.py using is_equiv,
                             because is_equiv is a pairwise predicate, not a hash)
  humaneval / livecodebench -> a PASS/FAIL signature from running the DOCTESTS
                             embedded in the problem statement

IRON RULE: never read the Task field holding the official grading suite. For
HumanEval that field is built straight from the dataset's `test` column (see
evaluator/hf_fetchers.py), so voting on it would select samples on the grader
and invalidate the benchmark number. The doctests used below are public -- they
are printed in the prompt the model already saw.
"""
from __future__ import annotations

import hashlib
import re

_MCQ = {"mmlu_pro", "gpqa_diamond"}
_MATH = {"math", "aime", "math_l5"}
_CODE = {"humaneval", "livecodebench"}

_DOCTEST_RE = re.compile(r"^\s*>>>\s*(.+?)\s*$\n^\s*(.+?)\s*$", re.M)

# The official MMLU-Pro chain ends in a last-resort scrape of the final
# standalone A-J letter, so "I have no idea" extracts as "I" (locked by
# tests/eval/test_official_mmlu.py -- the grader must always produce a guess).
# A vote must be stricter: a scraped letter is noise, not a choice. A ballot
# therefore counts only when the sample actually declares an option in the
# format the official prompt asks for ("The answer is (X)." / "Answer: X").
# This gates which samples enter the tally; grading still uses extract_answer.
_MCQ_DECLARED = re.compile(r"(?i:answer\s*(?:is|:))\s*\(?([A-J])\)?")


def extract_doctests(problem: str) -> list[tuple[str, str]]:
    """Return (call_expression, expected_repr) pairs from a problem statement."""
    out: list[tuple[str, str]] = []
    for call, expected in _DOCTEST_RE.findall(problem or ""):
        if expected.startswith(">>>"):      # a call with no shown output
            continue
        out.append((call.strip(), expected.strip()))
    return out


def doctest_signature(task, text: str, runner) -> str | None:
    """Run the problem's own doctests against the candidate code.

    Returns "PASS:<n>" when every doctest matches, "FAIL:<digest>" otherwise, or
    None when there is nothing to run (caller falls back to text plurality).
    """
    from evaluator.scorers.code import extract_code

    cases = extract_doctests(task.problem)
    code = extract_code(text)
    if not cases or not code.strip() or runner is None:
        return None
    lines = [code, ""]
    for call, expected in cases:
        lines.append(f"print(repr({call}))")
    program = "\n".join(lines)
    result = runner(program, stdin="", timeout_s=8.0, mem_mb=512, cpu_s=8)
    if getattr(result, "status", "") != "ok":
        return "FAIL:exec"
    got = [ln.strip() for ln in (result.stdout or "").splitlines() if ln.strip()]
    want = [e for _c, e in cases]
    if len(got) == len(want) and all(_same(g, w) for g, w in zip(got, want)):
        return f"PASS:{len(want)}"
    return "FAIL:" + hashlib.sha256("|".join(got).encode()).hexdigest()[:12]


def _same(got: str, want: str) -> bool:
    """Compare a repr() line to a doctest's expected text, tolerating quoting."""
    if got == want:
        return True
    return got.strip("'\"") == want.strip("'\"")


def ballot_key(task, text: str, runner=None) -> str | None:
    """Comparable key for one sample. None => spoiled ballot (dropped from tally)."""
    if task.source in _MCQ:
        from evaluator.official.mmlu_extract import extract_answer

        if not _MCQ_DECLARED.search(text or ""):
            return None
        return extract_answer(text or "")
    if task.source in _MATH:
        from evaluator.scorers.math import _extract_answer

        return _extract_answer(text or "")
    if task.source in _CODE:
        sig = doctest_signature(task, text or "", runner)
        if sig is not None:
            return sig
        from evaluator.scorers.code import extract_code

        code = extract_code(text or "")
        return code.strip() or None
    return (text or "").strip() or None

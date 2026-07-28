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


_TERMINATORS = ('"""', "'''")

# Characters that can only continue the expression on their left (comparison,
# arithmetic, attribute access, ...). A tail starting with one of these is part
# of the call, never an expected value printed beside it.
_CONTINUATION = set("+-*/%<>=!&|^@,.:)]}~")


def _is_expression(text: str) -> bool:
    """True when `text` on its own is a complete Python expression."""
    try:
        compile(text, "<doctest>", "eval")
    except Exception:
        return False
    return True


def _tail_after_call(expr: str) -> str | None:
    """Text following the first top-level balanced bracket group in `expr`."""
    depth = 0
    started = False
    quote: str | None = None
    i = 0
    while i < len(expr):
        ch = expr[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
            started = True
        elif ch in ")]}":
            depth -= 1
            if started and depth == 0:
                return expr[i + 1:]
        i += 1
    return None


def _has_inline_expected(call: str) -> bool:
    """True when a ">>>" line carries its expected value on the SAME line.

    HumanEval/116 writes `>>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]`.
    Python reads that as a subscript, so it executes and yields garbage rather
    than the comparison the author meant. The tell is a whitespace-separated
    tail that is itself a complete value expression -- `== 0` and `# comment`
    (the only other tails in the real suite) are not.
    """
    tail = _tail_after_call(call)
    if tail is None or not tail[:1].isspace():
        return False                      # `f(x)`, `f(x)[0]`, `f(x).lower()`
    rest = tail.strip()
    if not rest or rest[0] in _CONTINUATION:
        return False
    return _is_expression(rest)


def extract_doctests(problem: str) -> list[tuple[str, str]]:
    """Return (call_expression, expected_repr) pairs from a problem statement.

    Only cases we can honestly execute are returned. The regex pairs a ">>>"
    line with the line below it, and real HumanEval docstrings break that
    assumption in four ways -- each is SKIPPED rather than turned into a bogus
    expectation that fails correct code:

      * the example is the last line of the docstring, so the line below is the
        closing `\"\"\"` (HumanEval/108, /128, /156, /162);
      * the line below is blank;
      * the expected value spans several lines and would be truncated to its
        first line (HumanEval/113) -- detected because the truncation is not a
        complete expression. Multi-line expectations are deliberately NOT
        supported; skipping is the honest outcome;
      * call and expected share one line (HumanEval/116).
    """
    out: list[tuple[str, str]] = []
    for raw_call, raw_expected in _DOCTEST_RE.findall(problem or ""):
        call, expected = raw_call.strip(), raw_expected.strip()
        if expected.startswith(">>>"):      # a call with no shown output
            continue
        if not expected or expected in _TERMINATORS:
            continue
        if not _is_expression(expected):   # truncated multi-line expectation
            continue
        if _has_inline_expected(call):
            continue
        out.append((call, expected))
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
    try:
        result = runner(program, stdin="", timeout_s=8.0, mem_mb=512, cpu_s=8)
    except Exception:
        # A multi-hour paid batch must not die because Popen hit EAGAIN/EMFILE
        # under load. An unrunnable sample is indistinguishable from a failing
        # one for voting purposes, so record it as such and keep going.
        return "FAIL:exec"
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
    """Comparable key for one sample. None => spoiled ballot (dropped from tally).

    `runner` is OPTIONAL for mcq/math but MANDATORY for code: only 45.7% of
    HumanEval problems (and no livecodebench problem) carries ">>>" examples, so
    the raw-source fallback is already the majority path for code. A forgotten
    runner would quietly turn the whole code tier into text plurality --
    all-singleton tallies, universal ties, a score decided by ballot order --
    with no error and no log line. In a paid one-shot run that is unrecoverable,
    so it raises instead.
    """
    if task.source in _MCQ:
        from evaluator.official.mmlu_extract import extract_answer

        if not _MCQ_DECLARED.search(text or ""):
            return None
        return extract_answer(text or "")
    if task.source in _MATH:
        from evaluator.scorers.math import _extract_answer

        return _extract_answer(text or "")
    if task.source in _CODE:
        if runner is None:
            raise ValueError(
                f"ballot_key: {task.source} task {task.id!r} needs a code runner "
                "-- pass runner=evaluator.sandbox.run_code. With runner=None "
                "every code ballot silently degrades to raw-text plurality.")
        sig = doctest_signature(task, text or "", runner)
        if sig is not None:
            return sig
        from evaluator.scorers.code import extract_code

        code = extract_code(text or "")
        return code.strip() or None
    return (text or "").strip() or None

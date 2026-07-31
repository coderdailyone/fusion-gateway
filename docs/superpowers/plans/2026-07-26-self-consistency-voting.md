# M6 — Self-Consistency Resampling + Objective Voting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample each domestic model 10× per task, aggregate the 30 samples with **objective** rules (letter plurality / math-equivalence plurality / doctest execution), and beat claude-opus-4-8's 0.913 on the standard tier using only domestic models.

**Architecture:** Three isolated units under `evaluator/consistency/` — `sampler` (k samples per task/model, frozen, sharded, resumable), `normalize` (answer → comparable ballot key per source), `vote` (objective aggregation + tie detection). Scoring reuses the existing official scorers. Every sample is frozen, so the yield curve and the stop-rule oracle are computed at $0.

**Tech Stack:** Python 3.10, pytest, LiteLLM via `evaluator/validate.py`'s `MODELS`, the official graders (`evaluator/official/*`), the resource-limited sandbox (`evaluator/sandbox.py`), frozen JSONL store (`evaluator/store.py`).

## ⚠️ Spec correction adopted by this plan (read first)

The spec says code voting runs "the task's **public** tests … never the hidden
grader." **For HumanEval that distinction does not exist as specced:**
`evaluator/hf_fetchers.py:27` builds `task.tests` from the dataset's `row["test"]`
— that *is* the official grading test set. Voting by running `task.tests` would
select samples on the grader, trivially reaching oracle and producing an invalid
benchmark number.

**Corrected rule, used throughout this plan:** the public signal for code is the
**doctest examples embedded in the problem statement** (`task.problem`), which the
model already sees in its prompt. Those are extracted and executed; `task.tests`
is used *only* by the official scorer, after voting. When a task has no
extractable doctest, code voting falls back to plain text plurality.

## Global Constraints

- **No frontier model in any role.** Panel = `deepseek-chat`, `glm-5.2`, `kimi-k3`; the tie-break fuser is `glm-5.2`. A frontier model anywhere invalidates the milestone's claim.
- **No LLM judge in scoring.** Grading is `evaluator/scorers/*` only. The tie-break fuser produces an *answer*, never a grade.
- **Never vote using `task.tests`** (see the correction above) — that is the grader.
- **Isolation:** `evaluator/consistency/*` imports no `gateway.*`; never touches gateway SQLite.
- **Manifests byte-unchanged:** `configs/suite.manifest.json`, `configs/suite.hard.manifest.json`.
- **Secrets** only from `runs/secrets/.env` (mode 600, gitignored); never printed, never committed.
- **Frozen / re-scorable:** every sample persisted; re-voting and re-scoring cost $0.
- **Sampling:** k = **10** per model per task, **temperature 0.8**, all **1063** standard-tier tasks. Budget ceiling **$40** as a runaway backstop, not a target. Sharded + resumable.
- **Stop rule:** if the new 30-sample oracle does **not** exceed **0.9308** (M5's 3-candidate oracle), stop and report — do not spend further.
- **Baselines (M2c/M5, same suite, same graders):** opus **0.913** · gpt-5.5 0.905 · sonnet 0.898 · M5 fusion **0.897** · gpt-5.6-sol 0.894 · kimi-k3 0.884 · M5 oracle **0.9308**.
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Branch** `feat/m0-m1-gateway`; **venv** `.venv` (Python 3.10); **tests** `tests/eval/` (baseline **153** passing).

---

## File Structure

**Phase A — pure logic, no network, full TDD:**
- `evaluator/consistency/__init__.py` — package marker.
- `evaluator/consistency/normalize.py` — answer → ballot key, per source (+ doctest extraction/execution for code).
- `evaluator/consistency/vote.py` — objective aggregation, tie detection, first-k slicing for the yield curve.
- Tests: `tests/eval/test_consistency_normalize.py`, `test_consistency_vote.py`, `test_consistency_isolation.py`.

**Phase B — paid execution:**
- `evaluator/consistency/sampler.py` — k samples per (task, model), frozen, resumable.
- `scripts/run_consistency.py` — probe / sample / oracle-gate / vote+report driver.
- `docs/M6_VOTING_REPORT.md` — the published result.

---

## Phase A — pure logic (no network, no spend)

### Task 1: Ballot normalization per source

**Files:**
- Create: `evaluator/consistency/__init__.py`
- Create: `evaluator/consistency/normalize.py`
- Test: `tests/eval/test_consistency_normalize.py`

**Interfaces:**
- Produces: `ballot_key(task, text, runner=None) -> str | None` — `None` means a spoiled ballot; `extract_doctests(problem: str) -> list[tuple[str, str]]` returning `(call_expr, expected_repr)` pairs; `doctest_signature(task, text, runner) -> str | None`.
- Consumes: `evaluator.official.mmlu_extract.extract_answer(text) -> str | None`; `evaluator.official.math_grade.is_equiv(a, b) -> bool`; `evaluator.scorers.code.extract_code(text) -> str`; `evaluator.sandbox.run_code(code, stdin="", timeout_s=..., mem_mb=..., cpu_s=...) -> SandboxResult` with `.status` and `.stdout`.

> **Why math returns the raw extracted answer here:** `is_equiv` is a *pairwise*
> predicate, not a hash, so equivalent strings cannot be keyed directly. Task 2's
> vote merges equivalent keys using `is_equiv`. `ballot_key` therefore returns the
> extracted answer string for math and the merge happens at tally time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_consistency_normalize.py
from evaluator.suite.types import Task
from evaluator.consistency.normalize import (ballot_key, doctest_signature,
                                             extract_doctests)

MCQ = Task(id="m1", source="mmlu_pro", problem="2+2?\nA) 3\nB) 4", answer="B",
           tests=(), meta={})
MATH = Task(id="q1", source="math", problem="Compute 1/2.", answer="\\frac{1}{2}",
            tests=(), meta={})
CODE = Task(id="c1", source="humaneval",
            problem=('def add(a, b):\n    """ Add two numbers.\n'
                     '    >>> add(1, 2)\n    3\n    >>> add(0, 0)\n    0\n    """\n'),
            answer=None, tests=({"kind": "pyfunc", "test": "assert True",
                                 "entry_point": "add"},), meta={})


def test_mcq_ballot_key_is_the_letter():
    assert ballot_key(MCQ, "reasoning...\nThe answer is (B).") == "B"


def test_unparseable_answer_is_a_spoiled_ballot():
    assert ballot_key(MCQ, "I have no idea") is None


def test_math_ballot_key_is_the_extracted_answer():
    assert ballot_key(MATH, "so the value is \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_doctests_reads_examples_from_the_problem_statement():
    got = extract_doctests(CODE.problem)
    assert got == [("add(1, 2)", "3"), ("add(0, 0)", "0")]


def test_doctest_signature_marks_pass_and_fail_differently():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"

    def runner(code, stdin="", **kw):
        # a real-enough stub: exec the assembled program and capture stdout
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {})
            status = "ok"
        except Exception:
            status = "error"

        class R:
            pass
        r = R()
        r.status, r.stdout = status, buf.getvalue()
        return r

    sig_good = doctest_signature(CODE, good, runner)
    sig_bad = doctest_signature(CODE, bad, runner)
    assert sig_good is not None and sig_good != sig_bad
    assert sig_good.startswith("PASS")      # all doctests satisfied
    assert sig_bad.startswith("FAIL")


def test_code_ballot_key_without_doctests_falls_back_to_code_text():
    plain = Task(id="c2", source="humaneval", problem="def f(): pass",
                 answer=None, tests=(), meta={})
    key = ballot_key(plain, "```python\ndef f():\n    return 1\n```", runner=None)
    assert key is not None and "return 1" in key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/eval/test_consistency_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.consistency'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/consistency/__init__.py
"""M6 self-consistency voting. Isolated from gateway.* like the rest of evaluator/."""
```

```python
# evaluator/consistency/normalize.py
"""Turn a model answer into a comparable ballot key.

One rule per source:
  mmlu_pro / gpqa_diamond -> the extracted option letter
  math / aime / math_l5   -> the extracted answer string (equivalent strings are
                             merged at tally time by vote.py using is_equiv,
                             because is_equiv is a pairwise predicate, not a hash)
  humaneval / livecodebench -> a PASS/FAIL signature from running the DOCTESTS
                             embedded in the problem statement

IRON RULE: never use `task.tests` here. For HumanEval that field is the official
grading test set (see evaluator/hf_fetchers.py), so voting on it would select
samples on the grader. The doctests below are public — they are printed in the
prompt the model already saw.
"""
from __future__ import annotations

import hashlib
import re

_MCQ = {"mmlu_pro", "gpqa_diamond"}
_MATH = {"math", "aime", "math_l5"}
_CODE = {"humaneval", "livecodebench"}

_DOCTEST_RE = re.compile(r"^\s*>>>\s*(.+?)\s*$\n^\s*(.+?)\s*$", re.M)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/eval/test_consistency_normalize.py -v`
Expected: PASS (6 tests). If `evaluator/scorers/math.py`'s extractor is named differently, import the actual name — read that file rather than guessing.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest tests/eval -q`
Expected: `159 passed` (153 baseline + 6).

- [ ] **Step 6: Commit**

```bash
git add evaluator/consistency/__init__.py evaluator/consistency/normalize.py tests/eval/test_consistency_normalize.py
git commit -m "feat(M6): ballot normalization (letters / math answers / public doctests)

Code voting uses the problem statement's doctests, never task.tests — that
field is the official grader, so voting on it would select on the grader.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Objective voting + tie detection + first-k slicing

**Files:**
- Create: `evaluator/consistency/vote.py`
- Test: `tests/eval/test_consistency_vote.py`

**Interfaces:**
- Consumes: `ballot_key` (Task 1); `evaluator.official.math_grade.is_equiv`.
- Produces: `Ballot(model: str, key: str | None, text: str)`; `VoteResult(winner_text: str | None, winner_key: str | None, tally: dict[str, int], tied: bool, spoiled: int, n: int)`; `tally_keys(task, ballots) -> dict[str, int]` (math keys merged by `is_equiv`); `vote(task, ballots) -> VoteResult`; `first_k(ballots, k) -> list[Ballot]` (first k ballots **per model**, for the yield curve).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_consistency_vote.py
from evaluator.suite.types import Task
from evaluator.consistency.vote import Ballot, first_k, tally_keys, vote

MCQ = Task(id="m1", source="mmlu_pro", problem="?", answer="B", tests=(), meta={})
MATH = Task(id="q1", source="math", problem="?", answer="0.5", tests=(), meta={})
CODE = Task(id="c1", source="humaneval", problem="?", answer=None, tests=(), meta={})


def _b(model, key, text=""):
    return Ballot(model=model, key=key, text=text or f"text-{key}")


def test_plurality_winner_and_tally():
    res = vote(MCQ, [_b("a", "B"), _b("b", "B"), _b("c", "A")])
    assert res.winner_key == "B" and res.tally == {"B": 2, "A": 1}
    assert res.tied is False and res.spoiled == 0 and res.n == 3
    assert res.winner_text == "text-B"


def test_exact_tie_is_flagged_for_the_fuser():
    res = vote(MCQ, [_b("a", "A"), _b("b", "B")])
    assert res.tied is True


def test_spoiled_ballots_are_dropped_but_counted():
    res = vote(MCQ, [_b("a", "B"), _b("b", None), _b("c", None)])
    assert res.winner_key == "B" and res.spoiled == 2 and res.tied is False


def test_all_spoiled_yields_no_winner():
    res = vote(MCQ, [_b("a", None), _b("b", None)])
    assert res.winner_key is None and res.winner_text is None and res.tied is False


def test_math_equivalent_answers_merge_into_one_candidate():
    # "0.5" and "\\frac{1}{2}" are the same answer and must not split the vote
    t = tally_keys(MATH, [_b("a", "0.5"), _b("b", "\\frac{1}{2}"), _b("c", "3")])
    assert max(t.values()) == 2 and len(t) == 2


def test_code_passing_sample_beats_more_numerous_failing_ones():
    res = vote(CODE, [_b("a", "FAIL:aaa"), _b("b", "FAIL:aaa"), _b("c", "PASS:2")])
    assert res.winner_key == "PASS:2"       # execution beats popularity


def test_first_k_takes_k_ballots_per_model():
    ballots = [_b("a", "A"), _b("a", "A"), _b("a", "B"),
               _b("b", "A"), _b("b", "B")]
    got = first_k(ballots, 2)
    assert len(got) == 4                     # 2 from "a", 2 from "b"
    assert [x.model for x in got].count("a") == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/eval/test_consistency_vote.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.consistency.vote'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/consistency/vote.py
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
        passing = {k: v for k, v in tally.items() if k.startswith("PASS")}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/eval/test_consistency_vote.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest tests/eval -q`
Expected: `166 passed` (159 + 7).

- [ ] **Step 6: Commit**

```bash
git add evaluator/consistency/vote.py tests/eval/test_consistency_vote.py
git commit -m "feat(M6): objective voting — plurality, math-equivalence merge, execution beats popularity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Isolation test

**Files:**
- Test: `tests/eval/test_consistency_isolation.py`

- [ ] **Step 1: Write the test**

```python
# tests/eval/test_consistency_isolation.py
import ast
import pathlib

CONSISTENCY = pathlib.Path("evaluator/consistency")


def test_consistency_modules_never_import_gateway():
    files = list(CONSISTENCY.glob("*.py"))
    assert files, "no modules found — is cwd the repo root?"
    for py in files:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("gateway"), f"{py} imports {n.name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("gateway"), \
                    f"{py} imports from {node.module}"


def test_voting_never_reads_the_grading_tests():
    """task.tests is the official grader for humaneval — voting must not use it."""
    for name in ("normalize.py", "vote.py"):
        src = (CONSISTENCY / name).read_text()
        assert ".tests" not in src, f"{name} references task.tests (the grader)"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/eval/test_consistency_isolation.py -v`
Expected: PASS (2 tests). If the second test fails, the implementation is reading the grading tests — fix the implementation, not the test.

- [ ] **Step 3: Full suite + commit**

Run: `.venv/bin/pytest tests/eval -q`
Expected: `168 passed`.

```bash
git add tests/eval/test_consistency_isolation.py
git commit -m "test(M6): isolation + no-grader-in-voting guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase B — paid execution (keys + spend)

### Task 4: Temperature probe + 20-task smoke

**Files:**
- Create: `evaluator/consistency/sampler.py`
- Create: `scripts/run_consistency.py` (subcommands `probe`, `sample`)

**Interfaces:**
- Produces: `sample_task(task, model, k, completion_fn, temperature) -> list[str]`; the driver's `probe` and `sample` subcommands.
- Consumes: `evaluator.validate.MODELS` (each entry is a factory; `make_completion_fn(**overrides)` forwards overrides to `litellm.completion`, so `temperature=0.8` is passed as an override); `evaluator.runner.run_one`; `evaluator.store.append_frozen/read_frozen`; `scripts.resample_official.run_budgeted`.

- [ ] **Step 1: Probe whether each model honours `temperature`**
For each of `deepseek-chat`, `glm-5.2`, `kimi-k3`: sample ONE task 3× at temperature 0.8 and compare the three outputs.
Expected: deepseek/glm produce non-identical outputs. **kimi-k3 is a reasoning model and may reject or ignore the parameter** — if the call errors, drop `temperature` for kimi and rely on its own sampling variance; if outputs are byte-identical, record that kimi contributes k identical ballots (which the tally will collapse) and report it. Record the finding; it changes how kimi's votes are interpreted.

- [ ] **Step 2: 20-task smoke**
Sample 20 tasks stratified across the three sources, k=3 per model, ceiling **$1**. Freeze to `evaluator/runs/m6_consistency/smoke/`.
Expected: 20 × 3 models × 3 samples = 180 rows; report the spoiled-ballot rate from `ballot_key` and the measured $/task, and project the full run's cost against the $40 backstop.

- [ ] **Step 3: Commit the sampler + driver**

```bash
git add evaluator/consistency/sampler.py scripts/run_consistency.py
git commit -m "feat(M6): k-sample sampler + driver (probe, sample)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full resampling (1063 tasks × 3 models × k=10)

- [ ] **Step 1: Launch sharded sampling**
8 shards per model, resumable (skip any (task, model) already having k frozen samples), ceiling **$40** total, temperature 0.8 (minus kimi if Step-1 of Task 4 showed it rejects the parameter). Freeze to `evaluator/runs/m6_consistency/<model>-s<shard>/`.
Expected: ≈31,890 samples; report per-model completion counts and error counts. Re-run to resume after transient mirror errors.

- [ ] **Step 2: Report sampling completeness**
Print, per model, how many (task, sample-index) cells are filled. Tasks with fewer than 2 models' samples are excluded from the headline and counted.

---

### Task 6: The free stop-rule gate (oracle over 30 samples)

- [ ] **Step 1: Compute the new oracle at $0**
Score every frozen sample with the official scorers; a task counts as covered if **any** sample is correct.
Expected output: `new_oracle` over the evaluated set, alongside M5's 0.9308.

- [ ] **Step 2: Apply the stop rule**
- **new_oracle > 0.9308** → resampling surfaced answers the 3-candidate panel lacked; proceed to Task 7.
- **new_oracle ≤ 0.9308** → **STOP.** Write the finding to `docs/M6_VOTING_REPORT.md` (resampling produced no new answers; the next lever is per-task compute on disputed items, not more samples) and report to the user. Do not spend further.

---

### Task 7: Vote, score, and publish

**Files:**
- Create: `docs/M6_VOTING_REPORT.md`
- Modify: `scripts/run_consistency.py` (subcommands `vote`, `report`)

- [ ] **Step 1: Vote and score**
For every task: build ballots (`ballot_key` over all frozen samples, with `evaluator.sandbox.run_code` as the runner for code), `vote(...)`, and on `tied=True` call the existing `evaluator/fusion/fuse.py` fuser (glm-5.2) with the candidates **and the tally**. Score the winning answer with the official scorers. Freeze the voted answers.

- [ ] **Step 2: Yield curve at $0**
Re-vote using `first_k(ballots, k)` for **k = 1, 3, 5, 10** and score each. This shows where marginal returns die.

- [ ] **Step 3: Compute the report numbers**
Voting accuracy with **Wilson 95% CI**; **McNemar** vs opus (0.913) and vs M5 fusion (0.897); per-source breakdown (mmlu_pro / math / humaneval); tie-break rate; spoiled-ballot rate; new oracle vs 0.9308.

- [ ] **Step 4: The amplification diagnostic**
Count **"unanimous but wrong"** tasks before (3 candidates, M5) and after (30 samples). mmlu_pro is 79% unanimous at 0.851 today; if this count does not fall, self-consistency is amplifying systematic error on those tasks — say so plainly and name per-task compute as the next lever.

- [ ] **Step 5: Write `docs/M6_VOTING_REPORT.md` and commit**
Include the verdict: does a domestic pool under objective voting beat opus 0.913, and is the win significant.

Run: `.venv/bin/pytest tests/eval -q`
Expected: PASS.

```bash
git add docs/M6_VOTING_REPORT.md scripts/run_consistency.py
git commit -m "docs(M6): self-consistency voting report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** k=10 / temp 0.8 / all 1063 → Tasks 4–5 ✓ · objective voting per source → Tasks 1–2 ✓ · LLM tie-break only on genuine ties → Task 2 (`tied`) + Task 7 Step 1 ✓ · frozen + resumable + $40 backstop → Tasks 4–5 ✓ · free oracle stop rule → Task 6 ✓ · yield curve k=1/3/5/10 → Task 2 (`first_k`) + Task 7 Step 2 ✓ · report contents (Wilson, McNemar vs opus & M5, per-source, tie/spoiled rates, unanimous-but-wrong) → Task 7 ✓ · isolation + no-frontier + no-LLM-judge → Global Constraints + Task 3 ✓ · kimi temperature unknown → Task 4 Step 1 ✓.

**Spec deviation (flagged, deliberate):** the spec's "run the task's public tests" is impossible as written for HumanEval because `task.tests` *is* the grader; this plan substitutes the problem statement's doctests and adds a test (`test_voting_never_reads_the_grading_tests`) that enforces it structurally.

**Placeholder scan:** none. Phase-A code and tests are complete; Phase-B steps carry exact ceilings, run dirs, expected outputs, and the stop rule.

**Type consistency:** `Ballot(model, key, text)` and `VoteResult(winner_text, winner_key, tally, tied, spoiled, n)` are used identically in Tasks 2 and 7; `ballot_key(task, text, runner)` matches its callers; `first_k(ballots, k)` matches the yield-curve step; `wilson_ci(k, n)` / `mcnemar_p(b, c)` match `scripts/hard_report.py`.

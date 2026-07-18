# M2c Benchmark Official Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the self-written objective scorers with vendored **official-benchmark** judging, re-sample every feasible model through the official 0-shot-CoT pipeline, and recompute the true per-model and router numbers so the SOTA/Pareto claim rests on official scoring.

**Architecture:** A new `evaluator/official/` package holds faithfully-ported official judging cores (MATH `is_equiv`+`_strip_string`, MMLU-Pro extraction regex chain, HumanEval `check_correctness` assembly) plus official 0-shot-CoT prompt templates. The three `evaluator/scorers/*` modules become thin wrappers that delegate to `official/`, keeping the `Score` interface stable so `router/` and `sampler.py` are untouched. A new `evaluator/audit/` package side-loads dataset gold/canonical answers (independent of the frozen, hash-gated suite records) and runs a reference-answer self-test plus a human-eyeball disagreement report. Two scripts re-sample under a budget gate and recompute final numbers.

**Tech Stack:** Python 3, `sympy` (already a dep), `sklearn` (router, already a dep), `datasets` (lazy, for reference side-load only), `pytest`. No new third-party dependencies; official graders are vendored as source, not pip-installed.

## Global Constraints

- **Isolation:** nothing under `evaluator/` or `router/` may `import gateway.*` or touch the gateway SQLite. (existing discipline)
- **Frozen suite:** never change the records returned by `evaluator/hf_fetchers.py::extract` — `load_suite` gates them against `source.content_sha` and any field change raises `SuiteHashMismatch`. Reference answers for the audit are side-loaded separately.
- **Determinism:** scoring is a pure function of (task, output_text). No `random`, no network, no clock in any scorer or in `official/`. Re-scoring the same frozen output twice must yield an identical `Score`.
- **Provenance:** every file in `evaluator/official/` carries a header comment naming the upstream repo, commit/tag, and license (all MIT). Ported logic is copied faithfully; every local edit is commented.
- **Prompts:** 0-shot CoT only. Prompt templates contain only an answer-format instruction plus `task.problem`; never `task.answer` or `task.tests`.
- **Secrets:** API keys live only in `runs/secrets/.env` (mode 600, gitignored); never printed in full, never committed.
- **Budget:** the re-sample preflight estimates cost from token counts × `configs/pricing.toml`; warn at 80%, hard-stop at 100% of the budget ceiling before any offending paid call.
- **Score type (unchanged):** `evaluator.scorers.base.Score(correct: bool, detail: dict)`.
- **Task type (unchanged):** `evaluator.suite.types.Task(id, source, problem, answer: str|None, tests: tuple[dict,...], meta: dict)`.

---

## File structure

Create:
- `evaluator/official/__init__.py`
- `evaluator/official/math_grade.py` — Hendrycks `_strip_string` + `is_equiv`; `math_equiv(extracted, gold)` = strip-equal OR sympy-equivalent.
- `evaluator/official/mmlu_extract.py` — TIGER-Lab MMLU-Pro extraction chain; returns letter or `None`.
- `evaluator/official/humaneval_exec.py` — official program assembly `build_check_program(completion, test, entry_point)`.
- `evaluator/official/prompts.py` — 0-shot CoT prompt templates per source.
- `evaluator/audit/__init__.py`
- `evaluator/audit/references.py` — side-channel gold/canonical loader (cached), independent of the hash-gated suite.
- `evaluator/audit/reference_selftest.py` — gold→correct self-test over the suite.
- `evaluator/audit/disagreements.py` — human-eyeball audit report.
- `scripts/resample_official.py` — budget-gated resumable re-sample over all feasible models.
- `scripts/final_numbers.py` — score→matrix→router→numbers; writes report data.
- `docs/BENCHMARK_REPORT.md` — generated final report (skeleton created in Task 9).
- Tests: `tests/eval/test_official_math.py`, `tests/eval/test_official_mmlu.py`, `tests/eval/test_official_humaneval.py`, `tests/eval/test_official_prompts.py`, `tests/eval/test_references.py`, `tests/eval/test_reference_selftest.py`, `tests/eval/test_disagreements.py`, `tests/eval/test_resample_budget.py`, `tests/eval/test_final_numbers.py`.

Modify:
- `evaluator/scorers/math.py`, `evaluator/scorers/mcq.py`, `evaluator/scorers/code.py` — delegate to `official/`.
- `evaluator/runner.py` — `build_prompt` delegates to `official/prompts.py`, keeps the leakage guard.
- `tests/eval/test_scorers_math.py`, `test_scorers_mcq.py`, `test_scorers_code.py`, `test_runner.py` — re-point / extend.
- `docs/M3B_REPORT.md` — revise cost wording on corrected numbers (Task 9).
- `.gitignore` — ignore the references cache path.

---

### Task 1: Official MATH grader + delegate `scorers/math.py`

**Files:**
- Create: `evaluator/official/__init__.py` (empty), `evaluator/official/math_grade.py`
- Create: `tests/eval/test_official_math.py`
- Modify: `evaluator/scorers/math.py`
- Modify: `tests/eval/test_scorers_math.py`

**Interfaces:**
- Produces: `evaluator.official.math_grade.is_equiv(a: str|None, b: str|None) -> bool` (Hendrycks string-normalized equality); `evaluator.official.math_grade.math_equiv(extracted: str, gold: str) -> bool` (is_equiv OR sympy-equivalent). `evaluator.scorers.math.score(task, output_text) -> Score` unchanged signature.
- Consumes: nothing from other tasks.

**Design note (read before coding):** We adopt the **lm-evaluation-harness MATH recipe**: Hendrycks `_strip_string` normalization + a sympy equivalence fallback. This is *stricter* than the repo's current stopgap `math.py`: `_remove_right_units` recovers unit-labelled answers (`3\text{ treeks}` == `3`), but there is **no** `x \in` prefix stripping — that leniency was ours, not official, and is intentionally dropped here. The disagreement audit (Task 7) will quantify how many tasks that affects. Keep extraction (last `\boxed{...}`, then "answer is" phrase, then last number) — that stays in the scorer, only the *equivalence* moves to official.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_official_math.py`

```python
from evaluator.official.math_grade import is_equiv, math_equiv

def test_strip_equal_units():
    # official _remove_right_units strips a trailing \text{...} unit label
    assert is_equiv("3\\text{ treeks}", "3")

def test_strip_equal_fracs_and_sqrt():
    assert is_equiv("\\frac12", "\\frac{1}{2}")
    assert is_equiv("\\sqrt3", "\\sqrt{3}")
    assert is_equiv("\\tfrac{1}{2}", "\\frac{1}{2}")

def test_strip_equal_leading_var_assignment():
    # official strips a short "k =" / "x =" LHS (len(lhs) <= 2)
    assert is_equiv("x=5", "5")

def test_sympy_fallback_numeric():
    # string forms differ but are numerically equal -> sympy fallback
    assert math_equiv("0.5", "\\frac{1}{2}")
    assert math_equiv("1+1", "2")

def test_var_in_prefix_is_NOT_stripped():
    # deliberate: official does not strip "x \in"; our old leniency is dropped
    assert not is_equiv("[-2,7]", "x \\in [-2,7]")

def test_wrong():
    assert not is_equiv("41", "42")
    assert not math_equiv("[0,5]", "[-2,7]")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_official_math.py -q`
Expected: FAIL — `ModuleNotFoundError: evaluator.official.math_grade`.

- [ ] **Step 3: Implement `evaluator/official/math_grade.py`** (faithful Hendrycks port + sympy fallback)

```python
"""Official MATH answer grading.

Ported faithfully from the Hendrycks MATH release `math_equivalence.py`
(https://github.com/hendrycks/math , commit `357963a`, MIT License) — the
`_strip_string` / `is_equiv` normalization used by the MATH paper and adopted
by EleutherAI lm-evaluation-harness (`minerva_math`). Local additions are
limited to `math_equiv`, which layers a sympy equivalence fallback on top of
the official string comparison (also the lm-eval recipe); every such edit is
commented. No randomness, no I/O — deterministic string/CAS comparison only.
"""
from __future__ import annotations

import sympy


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_i, b_i = int(a), int(b)
        assert string == "{}/{}".format(a_i, b_i)
        return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    # Official: units appear on the right as "\text{ ...}".
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # official asserts len == 2; be defensive and take the left side
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # strip a short LHS assignment: "k = 5" -> "5" (official: len(lhs) <= 2)
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None) -> bool:
    """Official MATH string-normalized equality (Hendrycks)."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


# --- local addition: sympy equivalence fallback (lm-eval recipe) ----------

def _to_sympy(s: str):
    t = _strip_string(s)
    if not t:
        return None
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("^", "**")
    # translate \frac{a}{b} / \dfrac{a}{b} -> (a)/(b) (already brace-normalized
    # by _fix_fracs, so a and b are single braced groups)
    import re
    t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    try:
        return sympy.sympify(t)
    except Exception:
        try:
            return sympy.nsimplify(t)
        except Exception:
            return None


def math_equiv(extracted: str, gold: str) -> bool:
    """Official string match, then a sympy equivalence fallback."""
    if is_equiv(extracted, gold):
        return True
    a, b = _to_sympy(extracted), _to_sympy(gold)
    if a is None or b is None:
        return False
    try:
        if sympy.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(complex(a) - complex(b)) < 1e-9
    except Exception:
        return False
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_official_math.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Delegate `scorers/math.py` to the official grader** — keep extraction, replace the equivalence block. Replace the body of `score()` (lines that do string/sympy comparison) so it reads:

```python
from evaluator.official.math_grade import math_equiv
# ... keep _extract_answer and helpers ...

def score(task: Task, output_text: str) -> Score:
    extracted = _extract_answer(output_text)
    if extracted is None:
        return Score(False, {"reason": "unparseable"})
    expected_raw = task.answer if task.answer is not None else ""
    if math_equiv(extracted, expected_raw):
        return Score(True, {"extracted": extracted, "expected": expected_raw, "method": "official"})
    return Score(False, {"extracted": extracted, "expected": expected_raw, "method": "none"})
```

Delete the now-unused `_VAR_PREFIX`, `_TEXT`, `_normalize`, `_to_sympy`, `_FRAC*`, `_TRAILING_PUNCT` helpers from `scorers/math.py` (their job is now in `official/math_grade.py`). Keep `_find_last_boxed`, `_extract_after_phrase`, `_extract_last_token`, `_extract_answer`.

- [ ] **Step 6: Update `tests/eval/test_scorers_math.py`** — the `x \in` interval case now reflects official strictness. Replace `test_interval_and_set_answers` and `test_unit_labelled_number` with:

```python
def test_interval_exact_boxed():
    # interval answers still match when both sides agree (official is_equiv)
    assert score(T("[-2,7]"), "so \\boxed{[-2,7]}").correct
    assert not score(T("[-2,7]"), "\\boxed{[0,5]}").correct

def test_unit_labelled_number():
    # official _remove_right_units strips the \text{...} unit label
    assert score(T("3"), "\\boxed{3\\text{ treeks}}").correct

def test_var_in_prefix_now_strict():
    # documents the intentional alignment change: "x \in" prefix is not stripped
    assert not score(T("x \\in [-2,7]"), "so \\boxed{[-2,7]}").correct
```

- [ ] **Step 7: Run the full math scorer test file**

Run: `.venv/bin/pytest tests/eval/test_scorers_math.py tests/eval/test_official_math.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evaluator/official/__init__.py evaluator/official/math_grade.py evaluator/scorers/math.py tests/eval/test_official_math.py tests/eval/test_scorers_math.py
git commit -m "feat(eval): official MATH grader (Hendrycks is_equiv + sympy), delegate math scorer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Official MMLU-Pro extraction + delegate `scorers/mcq.py`

**Files:**
- Create: `evaluator/official/mmlu_extract.py`, `tests/eval/test_official_mmlu.py`
- Modify: `evaluator/scorers/mcq.py`, `tests/eval/test_scorers_mcq.py`

**Interfaces:**
- Produces: `evaluator.official.mmlu_extract.extract_answer(text: str) -> str | None` (letter A–J or None). `scorers.mcq.score(task, output_text) -> Score` unchanged.
- Consumes: nothing.

**Design note:** Port the TIGER-Lab MMLU-Pro three-stage extraction verbatim. Official returns a *random* letter on total failure; per the determinism constraint we instead return `None` and the scorer marks it incorrect. This is the single documented deviation (repeat it in the code header and in the Task 9 report).

- [ ] **Step 1: Write failing tests** — `tests/eval/test_official_mmlu.py`

```python
from evaluator.official.mmlu_extract import extract_answer

def test_answer_is_paren():
    assert extract_answer("... The answer is (C).") == "C"

def test_answer_colon():
    assert extract_answer("Reasoning...\nAnswer: B") == "B"

def test_last_letter_fallback():
    assert extract_answer("I think it is D") == "D"

def test_unparseable_returns_none():
    assert extract_answer("no letter here at all") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_official_mmlu.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/official/mmlu_extract.py`** (verbatim TIGER-Lab chain)

```python
"""Official MMLU-Pro answer extraction.

Ported verbatim from TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py`
(https://github.com/TIGER-AI-Lab/MMLU-Pro , commit `43be4b8`, MIT License).
DEVIATION (deliberate, for determinism): the official code returns a RANDOM
option when all three patterns miss; our frozen outputs must be reproducibly
re-scorable, so we return None instead (the scorer marks it incorrect). This
is the one documented divergence from the official harness.
"""
from __future__ import annotations

import re


def extract_answer(text: str) -> str | None:
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    return _extract_again(text)


def _extract_again(text: str) -> str | None:
    m = re.search(r".*[aA]nswer:\s*\(?([A-J])\)?", text)
    if m:
        return m.group(1)
    return _extract_final(text)


def _extract_final(text: str) -> str | None:
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if m:
        return m.group(0)
    return None
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_official_mmlu.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Delegate `scorers/mcq.py`** — replace the whole file body with:

```python
from evaluator.suite.types import Task
from evaluator.scorers.base import Score
from evaluator.official.mmlu_extract import extract_answer


def score(task: Task, output_text: str) -> Score:
    letter = extract_answer(output_text)
    if letter is None:
        return Score(False, {"reason": "unparseable", "method": "none"})
    expected = (task.answer or "").upper()
    return Score(letter.upper() == expected,
                 {"extracted": letter.upper(), "expected": expected, "method": "official"})
```

- [ ] **Step 6: Update `tests/eval/test_scorers_mcq.py`** — the official chain is letter-only. Ensure the file's expectations match official behavior: outputs following the official format ("The answer is (X)" / "Answer: X") extract correctly; a value-only answer with no letter is `unparseable`. Read the existing tests; for any case that assumed the old lenient value-boxing extraction, update the expected result to official (`unparseable` → incorrect). Add:

```python
def test_official_format_extracts():
    assert score(T("A"), "After analysis, the answer is (A).").correct
    assert not score(T("A"), "The answer is (B).").correct

def test_value_only_is_unparseable_under_official():
    # historical GLM value-boxing: no letter present -> official cannot parse
    r = score(T("A"), "\\boxed{30\\text{ m}}")
    assert not r.correct
```

(Here `T(ans)` is the file's existing task factory; mirror its definition if absent.)

- [ ] **Step 7: Run**

Run: `.venv/bin/pytest tests/eval/test_official_mmlu.py tests/eval/test_scorers_mcq.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evaluator/official/mmlu_extract.py evaluator/scorers/mcq.py tests/eval/test_official_mmlu.py tests/eval/test_scorers_mcq.py
git commit -m "feat(eval): official MMLU-Pro extraction chain, delegate mcq scorer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Official HumanEval execution + delegate `scorers/code.py`

**Files:**
- Create: `evaluator/official/humaneval_exec.py`, `tests/eval/test_official_humaneval.py`
- Modify: `evaluator/scorers/code.py`, `tests/eval/test_scorers_code.py`

**Interfaces:**
- Produces: `evaluator.official.humaneval_exec.build_check_program(completion: str, test: str, entry_point: str) -> str` — assembles the official check program.
- Consumes: `evaluator.sandbox.run_code` (existing) for isolated execution.

**Design note:** The official `human_eval` builds `problem["prompt"] + completion + "\n" + test + "\n" + f"check({entry_point})"`. Our models return the *full* function (prompt says "the full function definition"), so the extracted `code` already contains the signature; the official-equivalent assembly for full-function completions is `completion + "\n" + test + "\n" + f"check({entry_point})\n"`. Only the assembly moves to `official/`; execution stays on our sandbox (isolation discipline).

- [ ] **Step 1: Write failing tests** — `tests/eval/test_official_humaneval.py`

```python
from evaluator.official.humaneval_exec import build_check_program

def test_assembly_shape():
    prog = build_check_program("def f(x):\n    return x+1",
                               "def check(candidate):\n    assert candidate(1) == 2",
                               "f")
    assert prog.endswith("check(f)\n")
    assert "def f(x):" in prog
    assert "def check(candidate):" in prog
    # completion appears before the test, test before the check call
    assert prog.index("def f(x)") < prog.index("def check") < prog.index("check(f)")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_official_humaneval.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/official/humaneval_exec.py`**

```python
"""Official HumanEval check-program assembly.

Ported from openai/human-eval `execution.py::check_correctness`
(https://github.com/openai/human-eval , commit `463c980`, MIT License).
Only the *program assembly* is vendored here; execution is delegated to the
project's isolated sandbox (`evaluator.sandbox.run_code`) rather than
human-eval's in-process `exec`, to preserve our isolation discipline. The
assembly matches human-eval for full-function completions: completion, then
the problem's `test` (which defines `check`), then a `check(entry_point)` call.
"""
from __future__ import annotations


def build_check_program(completion: str, test: str, entry_point: str) -> str:
    return f"{completion}\n\n{test}\n\ncheck({entry_point})\n"
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_official_humaneval.py -q`
Expected: PASS.

- [ ] **Step 5: Delegate `scorers/code.py`** — in `_run_case`, replace the inline pyfunc assembly with the official builder:

```python
from evaluator.official.humaneval_exec import build_check_program
# ... inside _run_case, the "entry_point" branch:
    if "entry_point" in tc:
        script = build_check_program(code, tc["test"], tc["entry_point"])
        result = runner(script)
        passed = result.status == "ok"
        return passed, {
            "kind": "pyfunc", "entry_point": tc["entry_point"],
            "status": result.status, "stderr": result.stderr[:500], "passed": passed,
        }
```

Leave the stdin branch and everything else unchanged.

- [ ] **Step 6: Run existing code scorer tests (must stay green)**

Run: `.venv/bin/pytest tests/eval/test_scorers_code.py tests/eval/test_official_humaneval.py -q`
Expected: PASS (behavior is identical; assembly is now shared).

- [ ] **Step 7: Commit**

```bash
git add evaluator/official/humaneval_exec.py evaluator/scorers/code.py tests/eval/test_official_humaneval.py
git commit -m "feat(eval): official HumanEval check-program assembly, delegate code scorer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Official 0-shot-CoT prompts + leakage-safe `build_prompt`

**Files:**
- Create: `evaluator/official/prompts.py`, `tests/eval/test_official_prompts.py`
- Modify: `evaluator/runner.py`, `tests/eval/test_runner.py`

**Interfaces:**
- Produces: `evaluator.official.prompts.build(task: Task) -> str` — the official 0-shot-CoT prompt for `task.source`, consuming only `task.problem`.
- Consumes: `Task`. `runner.build_prompt(task)` keeps its signature and delegates here.

**Design note:** These are the *official answer-format instructions* for chat/reasoning models — MMLU-Pro: "think step by step, then finish with 'The answer is (X).'"; MATH: "reason step by step and put your final answer within \boxed{}"; HumanEval: complete-the-function, single code block. `build_prompt` must still consume only `task.problem` so the existing leakage guarantee/test holds.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_official_prompts.py`

```python
from evaluator.suite.types import Task
from evaluator.official.prompts import build

def _t(source, problem="PROBLEM", answer="X"):
    return Task(id="q", source=source, problem=problem, answer=answer, tests=(), meta={})

def test_mmlu_prompt_official_format():
    p = build(_t("mmlu_pro"))
    assert "The answer is (X)" in p or "answer is (X)" in p
    assert "step by step" in p.lower()
    assert "PROBLEM" in p

def test_math_prompt_boxed():
    p = build(_t("math"))
    assert "\\boxed" in p
    assert "step by step" in p.lower()
    assert "PROBLEM" in p

def test_humaneval_prompt_codeblock():
    p = build(_t("humaneval"))
    assert "PROBLEM" in p

def test_prompt_never_contains_answer():
    # leakage: the gold answer must not be injected
    p = build(_t("math", problem="compute", answer="SECRET_GOLD"))
    assert "SECRET_GOLD" not in p
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_official_prompts.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/official/prompts.py`**

```python
"""Official 0-shot chain-of-thought prompt templates per benchmark source.

Answer-format instructions follow each benchmark's official evaluation
protocol for chat/reasoning models: MMLU-Pro's "The answer is (X)." sink,
MATH's \\boxed{} final answer, HumanEval's single-code-block completion.
Consumes ONLY task.problem — never task.answer or task.tests (leakage guard).
"""
from __future__ import annotations

from evaluator.suite.types import Task

_MMLU = ("The following is a multiple choice question. Think step by step, "
         "then finish your response with a single line 'The answer is (X).' "
         "where X is the correct option letter.\n\n{problem}")
_MATH = ("Solve the following math problem. Reason step by step, and put your "
         "final answer within \\boxed{{}}.\n\n{problem}")
_HUMANEVAL = ("Complete the following Python function. Respond with a single "
              "Python code block containing the full function definition.\n\n{problem}")
_DEFAULT = "Solve the following problem. Put your final answer clearly at the end.\n\n{problem}"

_TEMPLATES = {"mmlu_pro": _MMLU, "math": _MATH, "humaneval": _HUMANEVAL}


def build(task: Task) -> str:
    template = _TEMPLATES.get(task.source, _DEFAULT)
    return template.format(problem=task.problem)
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_official_prompts.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Delegate `runner.build_prompt`** — replace `_INSTRUCTIONS`/`_DEFAULT_INSTRUCTION`/`build_prompt` in `evaluator/runner.py` with:

```python
from evaluator.official.prompts import build as _build_official_prompt

def build_prompt(task: Task) -> str:
    """Build the prompt sent to the model.

    Delegates to the official 0-shot-CoT templates. Consumes ONLY task.problem
    (never task.answer/task.tests); the leakage guarantee is structural and
    verified by test_runner's leakage test.
    """
    return _build_official_prompt(task)
```

- [ ] **Step 6: Keep the leakage test green** — confirm `tests/eval/test_runner.py`'s leakage test still passes; if it asserted specific old instruction text, update it to assert the new invariant (answer/tests not in the prompt, problem present). Run:

Run: `.venv/bin/pytest tests/eval/test_runner.py tests/eval/test_official_prompts.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluator/official/prompts.py evaluator/runner.py tests/eval/test_official_prompts.py tests/eval/test_runner.py
git commit -m "feat(eval): official 0-shot-CoT prompts; build_prompt delegates (leakage-safe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Reference loader (side-channel, cached)

**Files:**
- Create: `evaluator/audit/__init__.py` (empty), `evaluator/audit/references.py`, `tests/eval/test_references.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: dataclass `Reference(task_id: str, source: str, gold: str | None, solution: str | None, canonical_solution: str | None, prompt: str | None, test: str | None, entry_point: str | None)`; `load_references(manifest, cache_path="runs/cache/references.json", fetch=None) -> dict[str, Reference]`; `build_reference_index(rows_by_source: dict) -> dict[str, Reference]` (pure, for tests).
- Consumes: `evaluator.suite.manifest.Manifest` (existing `load`), `datasets` (lazy, only in the real fetch path).

**Design note:** This must NOT go through `load_suite`/`content_sha`. It reads the **same pinned** `(hf_dataset, hf_revision, split)` as the manifest but pulls the *reference* fields the frozen records deliberately omit (`solution`, `canonical_solution`). The real fetch is injectable (`fetch=`) so unit tests never hit the network; a JSON cache makes repeat runs offline and deterministic.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_references.py`

```python
from evaluator.audit.references import Reference, build_reference_index

def test_build_index_math_and_humaneval():
    rows = {
        "math": [{"unique_id": "m1", "answer": "42", "solution": "... \\boxed{42}"}],
        "humaneval": [{"task_id": "HE/1", "prompt": "def f():\n", "test": "def check(c): assert c()==1",
                       "entry_point": "f", "canonical_solution": "    return 1"}],
        "mmlu_pro": [{"question_id": 7, "answer": "C"}],
    }
    idx = build_reference_index(rows)
    assert idx["m1"].gold == "42"
    assert "\\boxed{42}" in idx["m1"].solution
    assert idx["HE/1"].canonical_solution == "    return 1"
    assert idx["HE/1"].entry_point == "f"
    assert idx["7"].gold == "C"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_references.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/audit/references.py`**

```python
"""Side-channel loader for dataset reference answers, for the audit only.

Independent of `load_suite`: it reads the SAME pinned dataset revision as the
manifest but returns the reference fields the frozen suite records omit
(math `solution`, humaneval `canonical_solution`). It never alters or re-hashes
the frozen suite. Results are cached to JSON so re-runs are offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Reference:
    task_id: str
    source: str
    gold: str | None = None
    solution: str | None = None
    canonical_solution: str | None = None
    prompt: str | None = None
    test: str | None = None
    entry_point: str | None = None


def build_reference_index(rows_by_source: dict[str, list[dict]]) -> dict[str, Reference]:
    idx: dict[str, Reference] = {}
    for source, rows in rows_by_source.items():
        for row in rows:
            if source == "mmlu_pro":
                tid = str(row["question_id"])
                idx[tid] = Reference(tid, source, gold=str(row["answer"]))
            elif source == "math":
                tid = str(row["unique_id"])
                idx[tid] = Reference(tid, source, gold=str(row["answer"]),
                                     solution=row.get("solution"))
            elif source == "humaneval":
                tid = str(row["task_id"])
                idx[tid] = Reference(tid, source, prompt=row.get("prompt"),
                                     test=row.get("test"), entry_point=row.get("entry_point"),
                                     canonical_solution=row.get("canonical_solution"))
    return idx


def _fetch_rows(manifest, fetch) -> dict[str, list[dict]]:
    from datasets import load_dataset  # lazy: only the real path needs it
    fetch = fetch or (lambda ds, split, revision: list(load_dataset(ds, split=split, revision=revision)))
    rows_by_source: dict[str, list[dict]] = {}
    for s in manifest.sources:
        rows_by_source[s.name] = fetch(s.hf_dataset, s.split, s.hf_revision)
    return rows_by_source


def load_references(manifest, cache_path: str = "runs/cache/references.json",
                    fetch=None) -> dict[str, Reference]:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        return {tid: Reference(**rec) for tid, rec in data.items()}
    idx = build_reference_index(_fetch_rows(manifest, fetch))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({tid: asdict(r) for tid, r in idx.items()}, f)
    return idx
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_references.py -q`
Expected: PASS.

- [ ] **Step 5: gitignore the cache** — add to `.gitignore`:

```
runs/cache/
```

- [ ] **Step 6: Commit**

```bash
git add evaluator/audit/__init__.py evaluator/audit/references.py tests/eval/test_references.py .gitignore
git commit -m "feat(audit): side-channel reference loader (cached, off the frozen suite)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Reference-answer self-test (the audit weapon) + historical-bug regression

**Files:**
- Create: `evaluator/audit/reference_selftest.py`, `tests/eval/test_reference_selftest.py`

**Interfaces:**
- Produces: dataclass `SelfTestFailure(task_id, source, reason, detail)`; `selftest_one(task: Task, ref: Reference) -> SelfTestFailure | None`; `selftest(tasks: list[Task], refs: dict[str, Reference]) -> list[SelfTestFailure]`.
- Consumes: the delegated scorers (`evaluator.scorers.{mcq,math,code}.score`), `Reference` (Task 5), `Task`.

**Design note:** Construct each source's dataset-gold answer as a realistic model output, run the *actual* scorer, and require `correct is True`:
- **mmlu_pro:** synthesize `"The answer is ({gold})."` → mcq scorer must return correct.
- **math:** use the dataset `solution` (real prose ending in `\boxed{gold}`) as the output → math scorer must return correct; this exercises extraction + `math_equiv` on real solution text (the strong form of the self-test). If `solution` is missing, fall back to `"\\boxed{{{gold}}}"`.
- **humaneval:** run `prompt + canonical_solution` (the full reference function) as a fenced code block → code scorer must pass the official test.

A failure means the grader cannot recognize the benchmark's own correct answer — a definite scorer bug. The three historical bug formats are included as explicit regression cases (they must now pass or be correctly strict).

- [ ] **Step 1: Write failing tests** — `tests/eval/test_reference_selftest.py`

```python
from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.audit.reference_selftest import selftest_one

def _t(source, answer=None, tests=()):
    return Task(id="q", source=source, problem="P", answer=answer, tests=tests, meta={})

def test_mmlu_gold_recognized():
    t = _t("mmlu_pro", answer="C")
    ref = Reference("q", "mmlu_pro", gold="C")
    assert selftest_one(t, ref) is None

def test_math_gold_via_solution_recognized():
    t = _t("math", answer="42")
    ref = Reference("q", "math", gold="42", solution="Adding up we get \\boxed{42}.")
    assert selftest_one(t, ref) is None

def test_math_unit_answer_recognized():  # historical bug #3
    t = _t("math", answer="3")
    ref = Reference("q", "math", gold="3", solution="So \\boxed{3\\text{ treeks}}.")
    assert selftest_one(t, ref) is None

def test_humaneval_canonical_passes():
    tests = ({"kind": "pyfunc",
              "test": "def check(candidate):\n    assert candidate(2) == 3\n",
              "entry_point": "f"},)
    t = _t("humaneval", tests=tests)
    ref = Reference("q", "humaneval", prompt="def f(x):\n", canonical_solution="    return x + 1\n",
                    entry_point="f", test=tests[0]["test"])
    assert selftest_one(t, ref) is None

def test_detects_broken_gold():
    # a gold the scorer cannot match surfaces as a failure
    t = _t("mmlu_pro", answer="C")
    ref = Reference("q", "mmlu_pro", gold="Z")  # not A-J -> unparseable/incorrect
    assert selftest_one(t, ref) is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_reference_selftest.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/audit/reference_selftest.py`**

```python
"""Reference-answer self-test: the scorer MUST mark each dataset gold correct.

Constructs each task's own reference answer as a realistic model output and
asserts the delegated scorer returns correct. A failure is a definite scorer
bug (it cannot recognize the benchmark's own correct answer). Offline and
deterministic; intended both as a one-shot 1063-task audit and as a home for
the historical-bug regression cases.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.scorers import mcq, code
from evaluator.scorers import math as math_scorer


@dataclass(frozen=True)
class SelfTestFailure:
    task_id: str
    source: str
    reason: str
    detail: dict


def _synth_output(task: Task, ref: Reference) -> str:
    if task.source == "mmlu_pro":
        return f"Reasoning about the options. The answer is ({ref.gold})."
    if task.source == "math":
        if ref.solution:
            return ref.solution
        return f"Therefore the answer is \\boxed{{{ref.gold}}}."
    if task.source == "humaneval":
        body = (ref.prompt or "") + (ref.canonical_solution or "")
        return f"```python\n{body}\n```"
    return ref.gold or ""


_SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score}


def selftest_one(task: Task, ref: Reference) -> SelfTestFailure | None:
    output = _synth_output(task, ref)
    result = _SCORERS[task.source](task, output)
    if result.correct:
        return None
    return SelfTestFailure(task.id, task.source, "gold_not_recognized", dict(result.detail))


def selftest(tasks: list[Task], refs: dict[str, Reference]) -> list[SelfTestFailure]:
    failures: list[SelfTestFailure] = []
    for task in tasks:
        ref = refs.get(task.id)
        if ref is None:
            failures.append(SelfTestFailure(task.id, task.source, "missing_reference", {}))
            continue
        f = selftest_one(task, ref)
        if f is not None:
            failures.append(f)
    return failures
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_reference_selftest.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluator/audit/reference_selftest.py tests/eval/test_reference_selftest.py
git commit -m "feat(audit): reference-answer self-test + historical-bug regression cases

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Disagreement audit report

**Files:**
- Create: `evaluator/audit/disagreements.py`, `tests/eval/test_disagreements.py`

**Interfaces:**
- Produces: dataclass `DisagreementCase(task_id, source, model, kind, detail)`; `find_cases(rows: list[ResultRow], frozen_by_key: dict[tuple[str,str], str], scorers=...) -> list[DisagreementCase]`; `render(cases) -> str`.
- Consumes: `evaluator.report.ResultRow`; the frozen output text keyed by `(task_id, model)`.

**Design note:** No LLM judge. Surface two informative kinds for human eyeballing: (a) `wrong_but_others_right` — a task where this model was scored wrong while at least one other model was scored right (candidate false-negative); (b) `unparseable` — the scorer returned `method="none"` (candidate extraction miss). Rank so the highest-signal cases (most models-right-yet-this-wrong) come first.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_disagreements.py`

```python
from evaluator.report import ResultRow
from evaluator.audit.disagreements import find_cases, render

def test_flags_wrong_but_others_right():
    rows = [
        ResultRow("t1", "math", "A", False, 0.0),
        ResultRow("t1", "math", "B", True, 0.0),
        ResultRow("t2", "math", "A", True, 0.0),
        ResultRow("t2", "math", "B", True, 0.0),
    ]
    cases = find_cases(rows, frozen_by_key={})
    keys = {(c.task_id, c.model, c.kind) for c in cases}
    assert ("t1", "A", "wrong_but_others_right") in keys
    # t2: everyone right -> no case
    assert not any(c.task_id == "t2" for c in cases)

def test_render_nonempty():
    rows = [ResultRow("t1", "math", "A", False, 0.0), ResultRow("t1", "math", "B", True, 0.0)]
    out = render(find_cases(rows, frozen_by_key={}))
    assert "t1" in out and "wrong_but_others_right" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_disagreements.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/audit/disagreements.py`**

```python
"""Human-eyeball disagreement audit (no LLM judge).

Surfaces candidate scorer false-negatives for manual spot-check: tasks a model
got 'wrong' while others got them right, and outputs the scorer could not parse.
Ranks by how many other models were right (higher = more suspicious).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from evaluator.report import ResultRow


@dataclass(frozen=True)
class DisagreementCase:
    task_id: str
    source: str
    model: str
    kind: str
    detail: dict


def find_cases(rows: list[ResultRow], frozen_by_key: dict[tuple, str],
               scorers=None) -> list[DisagreementCase]:
    by_task: dict[str, list[ResultRow]] = defaultdict(list)
    for r in rows:
        by_task[r.task_id].append(r)

    cases: list[DisagreementCase] = []
    for task_id, task_rows in by_task.items():
        n_right = sum(1 for r in task_rows if r.correct)
        for r in task_rows:
            if not r.correct and n_right > 0:
                cases.append(DisagreementCase(
                    task_id, r.source, r.model, "wrong_but_others_right",
                    {"n_other_right": n_right,
                     "output": frozen_by_key.get((task_id, r.model), "")[:400]}))
    cases.sort(key=lambda c: (-c.detail.get("n_other_right", 0), c.task_id, c.model))
    return cases


def render(cases: list[DisagreementCase]) -> str:
    lines = ["## Disagreement audit (human spot-check)", ""]
    lines.append(f"total cases: {len(cases)}")
    lines.append("")
    lines.append("| task_id | source | model | kind | n_other_right |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in cases:
        lines.append(f"| {c.task_id} | {c.source} | {c.model} | {c.kind} | "
                     f"{c.detail.get('n_other_right', '')} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_disagreements.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluator/audit/disagreements.py tests/eval/test_disagreements.py
git commit -m "feat(audit): human-eyeball disagreement report (no LLM judge)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Budget-gated re-sample driver

**Files:**
- Create: `scripts/resample_official.py`, `tests/eval/test_resample_budget.py`

**Interfaces:**
- Produces: `estimate_cost(rows_est: list[tuple[str,int,int]], cost_fn) -> float`; `budget_gate(spent: float, next_cost: float, ceiling: float) -> str` returning `"ok"|"warn"|"stop"`; `main()` runnable entry.
- Consumes: `evaluator.pilot.run_pilot` (resumable driver), `evaluator.validate.MODELS`, `evaluator.pricing.cost`, `configs/pricing.toml`.

**Design note:** The paid run itself is a manual acceptance step (real API calls, isolated host), not a CI test. Only the pure budget logic is unit-tested. The driver re-samples the locked suite with the now-official prompts (Task 4 already changed `build_prompt`, so `run_pilot`/`sample` use official prompts automatically) over all feasible models into a NEW run dir; already-frozen pairs are skipped (resumable). Kimi is included only if a cheap probe call succeeds.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_resample_budget.py`

```python
from scripts.resample_official import budget_gate, estimate_cost

def test_budget_gate_states():
    assert budget_gate(spent=0.0, next_cost=1.0, ceiling=10.0) == "ok"
    assert budget_gate(spent=8.5, next_cost=0.1, ceiling=10.0) == "warn"   # >=80%
    assert budget_gate(spent=9.99, next_cost=0.1, ceiling=10.0) == "stop"  # would cross 100%

def test_estimate_cost_sums():
    cost_fn = lambda model, i, o: 0.001 * (i + o)
    rows = [("m", 100, 200), ("m", 0, 100)]
    assert abs(estimate_cost(rows, cost_fn) - (0.001*300 + 0.001*100)) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_resample_budget.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/resample_official.py`**

```python
"""Budget-gated official-pipeline re-sample over all feasible models.

Real API calls -> run manually on an isolated host with keys in
runs/secrets/.env. build_prompt already emits official 0-shot-CoT prompts, so
run_pilot/sample re-sample the locked suite through the official pipeline.
Writes a NEW run dir; already-frozen (task, model) pairs are skipped.
"""
from __future__ import annotations

FEASIBLE_MODELS = [
    "deepseek-chat", "claude-sonnet-5", "claude-opus-4-8",
    "gpt-5.6-sol", "gpt-5.5", "glm-5.2", "kimi-k2",
]
BUDGET_CEILING_USD = 35.0
WARN_FRACTION = 0.80


def estimate_cost(rows_est, cost_fn) -> float:
    return sum(cost_fn(m, i, o) for (m, i, o) in rows_est)


def budget_gate(spent: float, next_cost: float, ceiling: float) -> str:
    if spent + next_cost > ceiling:
        return "stop"
    if spent >= WARN_FRACTION * ceiling:
        return "warn"
    return "ok"


def main() -> None:
    import sys
    from datetime import datetime, timezone
    from evaluator import validate
    from evaluator.store import new_run_dir

    validate.load_secrets()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1063  # full locked suite
    models = [m for m in FEASIBLE_MODELS if m in validate.MODELS]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir("evaluator", "official_resample", ts)
    print(f"re-sampling {n} tasks x {len(models)} models -> {run_dir}")
    print(f"budget ceiling ${BUDGET_CEILING_USD}; warn at {int(WARN_FRACTION*100)}%")
    from evaluator.pilot import run_pilot
    run_pilot(n=n, run_dir=run_dir, model_names=models)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_resample_budget.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/resample_official.py tests/eval/test_resample_budget.py
git commit -m "feat(eval): budget-gated official-pipeline re-sample driver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6 (manual acceptance — real spend):** After the whole plan's code is merged, run the reference self-test first (Task 6, over the full suite via `load_references`), fix any grader gaps until it is 100% (or documented), THEN run `.venv/bin/python -m scripts.resample_official`. Monitor cost against the ceiling. This step produces the frozen outputs Task 9 consumes.

---

### Task 9: Final numbers + `BENCHMARK_REPORT.md` + revise `M3B_REPORT.md`

**Files:**
- Create: `scripts/final_numbers.py`, `tests/eval/test_final_numbers.py`, `docs/BENCHMARK_REPORT.md`
- Modify: `docs/M3B_REPORT.md`

**Interfaces:**
- Produces: `per_model_table(rows) -> dict[str, dict]` (delegates to `evaluator.report.aggregate`); `sota_verdict(points: dict[str, tuple[float,float]]) -> dict` returning `{"best_single": str, "best_acc": float}`; `main()` that scores a run dir and prints/writes the numbers.
- Consumes: `evaluator.sampler.sample` (offline scoring over frozen), `router.matrix.ResultMatrix.from_rows`, `router.learnability.per_model_cv_auc`+`gate`, `router.train.fit_oof`, `router.policy.sweep_lambda`, `router.cascade`, `router.pareto.static_points`+`envelopes`+`render_report`.

**Design note:** The computation core is unit-testable on a tiny synthetic matrix; the actual report is generated from the Task 8 run. Reuse the existing M3b pipeline exactly (GroupKFold OOF, λ-sweep, code cascade, envelope check) so the only thing that changed is the *labels* (now official). The report must state official-grader provenance, the one MMLU deviation, the self-test result, true per-model accuracy for every sampled model, and whether absolute-SOTA-over-pool (incl. sol) and Pareto-dominance hold on corrected numbers.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_final_numbers.py`

```python
from scripts.final_numbers import sota_verdict

def test_sota_verdict_picks_best_single():
    points = {"deepseek-chat": (0.85, 0.0001), "claude-sonnet-5": (0.87, 0.002),
              "gpt-5.6-sol": (0.83, 0.004)}
    v = sota_verdict(points)
    assert v["best_single"] == "claude-sonnet-5"
    assert abs(v["best_acc"] - 0.87) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_final_numbers.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/final_numbers.py`**

```python
"""Recompute true per-model + router numbers from an official-pipeline run.

Offline: scores frozen outputs with the (now official) scorers, builds the
result matrix, and reruns the exact M3b pipeline (learnability gate, OOF
classifiers, lambda-swept policy, code verify-cascade, Pareto envelope) on the
corrected labels. Prints the numbers for the benchmark report.
"""
from __future__ import annotations


def sota_verdict(points: dict[str, tuple[float, float]]) -> dict:
    best = max(points.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
    return {"best_single": best[0], "best_acc": best[1][0]}


def per_model_table(rows) -> dict:
    from evaluator.report import aggregate
    return aggregate(rows)


def main() -> None:
    import sys
    from evaluator import validate
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher
    from evaluator.sampler import sample
    from router.matrix import ResultMatrix
    from router.learnability import per_model_cv_auc, gate
    from router.train import fit_oof
    from router.policy import sweep_lambda
    from router import cascade
    from router.pareto import static_points, envelopes, render_report

    validate.load_secrets()
    run_dir = sys.argv[1]
    manifest = load("configs/suite.manifest.json")
    tasks = load_suite(manifest, {s.name: make_fetcher(s.name) for s in manifest.sources})
    models = {name: (lambda: None) for name in []}  # scoring only; no calls
    # score frozen outputs offline (sample() skips already-frozen pairs; with an
    # empty models dict it does no sampling and only scores what's frozen)
    rows = sample(models, tasks, run_dir)

    print("=== per-model (official scoring) ===")
    for m, a in sorted(per_model_table(rows).items()):
        print(f"  {m:18} acc={a['accuracy']:.4f} mean_cost=${a['mean_cost_usd']:.6f} n={a['n']}")

    matrix = ResultMatrix.from_rows(rows)
    task_ids = [t.id for t in tasks]
    non_code = [t.id for t in tasks if t.source != "humaneval"]
    code_ids = [t.id for t in tasks if t.source == "humaneval"]

    aucs = per_model_cv_auc([t for t in tasks if t.source != "humaneval"], matrix)
    print("learnability:", gate(aucs), aucs)

    oof = fit_oof([t for t in tasks if t.source != "humaneval"], matrix)
    dyn = sweep_lambda(oof, matrix, non_code, [0.0, 1.0, 3.0, 10.0, 1e6])
    sp = static_points(matrix, task_ids)
    print("SOTA verdict:", sota_verdict(sp))
    print(render_report(dyn, sp, envelopes(dyn, sp)))
    # code cascade over cheapest->dearest order (by mean cost)
    order = sorted(matrix.models, key=lambda m: sum(matrix.cost[m].values()))
    print("code cascade:", cascade.evaluate(code_ids, order, matrix))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run new tests to green**

Run: `.venv/bin/pytest tests/eval/test_final_numbers.py -q`
Expected: PASS.

- [ ] **Step 5: Create `docs/BENCHMARK_REPORT.md` skeleton** (numbers filled from the Task 8 run in Step 7):

```markdown
# M2c — Benchmark Official-Alignment Report

**Scoring:** official-benchmark judging, vendored under `evaluator/official/`:
MATH `is_equiv`/`_strip_string` (Hendrycks, MIT) + sympy fallback; MMLU-Pro
extraction chain (TIGER-Lab, MIT); HumanEval `check_correctness` assembly
(OpenAI human-eval, MIT). Prompts: official 0-shot CoT.

**One documented deviation:** MMLU-Pro official extraction falls back to a
random option on parse failure; for deterministic re-scoring we treat a parse
failure as incorrect. Affected items: <N> (from the disagreement audit).

**Reference self-test:** <PASS 1063/1063 | list exceptions>.

## True per-model accuracy (official scoring, official prompts)

| model | accuracy | mean cost/task |
|---|---|---|
| ... | ... | ... |

## Router on corrected labels

<paste render_report output: static baselines, lambda curve, envelope verdict>

## SOTA verdict

<does any strategy beat every single model incl. gpt-5.6-sol? Pareto dominance?>

## Disagreement audit

<count + notable human-checked cases>
```

- [ ] **Step 6: Revise `docs/M3B_REPORT.md`** — replace the headline and the "~2.6× lower cost / ~3×" claims with the recomputed numbers from Step 7, and add a top note: "Numbers regenerated under M2c official scoring (see BENCHMARK_REPORT.md); the earlier cost-savings figures were inflated by a math-scorer bug and are retracted." Keep the honest-attribution and DRACO-20 sections, updating any figures that changed.

- [ ] **Step 7 (manual, after Task 8 run):** Run `.venv/bin/python -m scripts.final_numbers <run_dir>` and the disagreement audit; paste the true numbers into `BENCHMARK_REPORT.md` and the revised `M3B_REPORT.md`.

- [ ] **Step 8: Commit**

```bash
git add scripts/final_numbers.py tests/eval/test_final_numbers.py docs/BENCHMARK_REPORT.md docs/M3B_REPORT.md
git commit -m "feat(eval): final-numbers recompute + benchmark report; revise M3B on official scoring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final gate (after all tasks)

- [ ] Run the whole eval + router suite: `.venv/bin/pytest tests/eval tests/router -q` — all green.
- [ ] Confirm isolation: `grep -rn "import gateway" evaluator/ router/` returns nothing.
- [ ] Confirm the suite hash still holds: `.venv/bin/python -c "from evaluator.suite.manifest import load; from evaluator.suite.loader import load_suite; from evaluator.hf_fetchers import make_fetcher; m=load('configs/suite.manifest.json'); load_suite(m, {s.name: make_fetcher(s.name) for s in m.sources}); print('suite hash OK')"`.
- [ ] Reference self-test 100% (or documented exceptions) recorded in `BENCHMARK_REPORT.md`.

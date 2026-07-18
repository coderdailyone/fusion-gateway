import re

from evaluator.suite.types import Task
from evaluator.scorers.base import Score
from evaluator.sandbox import run_code
from evaluator.official.humaneval_exec import build_check_program

# --- extraction -------------------------------------------------------

_PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)


def extract_code(output_text: str) -> str:
    """Return the contents of the last fenced code block in `output_text`.

    Prefers the last ```python ... ``` block; falls back to the last bare
    ``` ... ``` block; falls back to the whole (stripped) text if there are
    no fences at all.
    """
    python_matches = _PYTHON_FENCE.findall(output_text)
    if python_matches:
        return python_matches[-1].strip()

    bare_matches = _BARE_FENCE.findall(output_text)
    if bare_matches:
        return bare_matches[-1].strip()

    return output_text.strip()


# --- scoring -----------------------------------------------------------


def _run_case(code: str, tc: dict, runner) -> tuple[bool, dict]:
    """Run one test case. Two shapes are supported:

    - "pyfunc" (e.g. HumanEval): the case has `test` (defines check(candidate))
      and `entry_point`. We build `code + test + check(entry_point)` and run it;
      pass = the process exits cleanly (no AssertionError -> status "ok").
    - "stdin" (competitive style): the case has `stdin` / `expected_stdout`.
      We feed stdin and compare trimmed stdout.
    """
    if "entry_point" in tc:
        script = build_check_program(code, tc["test"], tc["entry_point"])
        result = runner(script)
        passed = result.status == "ok"
        return passed, {
            "kind": "pyfunc",
            "entry_point": tc["entry_point"],
            "status": result.status,
            "stderr": result.stderr[:500],
            "passed": passed,
        }

    result = runner(code, stdin=tc["stdin"])
    passed = result.status == "ok" and result.stdout.strip() == tc["expected_stdout"].strip()
    return passed, {
        "kind": "stdin",
        "stdin": tc["stdin"],
        "expected_stdout": tc["expected_stdout"],
        "status": result.status,
        "stdout": result.stdout,
        "passed": passed,
    }


def score(task: Task, output_text: str, runner=run_code) -> Score:
    if not task.tests:
        return Score(False, {"reason": "no_tests"})

    code = extract_code(output_text)

    cases = []
    first_failure = None
    all_pass = True

    for tc in task.tests:
        passed, case_detail = _run_case(code, tc, runner)
        cases.append(case_detail)
        if not passed:
            all_pass = False
            if first_failure is None:
                first_failure = case_detail

    detail = {"cases": cases}
    if first_failure is not None:
        detail["first_failure"] = first_failure

    return Score(all_pass, detail)

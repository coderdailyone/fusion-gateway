"""LiveCodeBench test-case normalization into the generic test shapes the
project's code scorer already executes.

Matches the evaluation protocol implemented in LiveCodeBench/LiveCodeBench
(https://github.com/LiveCodeBench/LiveCodeBench), specifically
`lcb_runner/evaluation/testing_util.py::grade_call_based` /
`grade_stdio` as observed at commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
(main HEAD, live-verified 2026-07-19), MIT License (live-verified via
https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/LICENSE
on the same date). This is NOT a verbatim vendored port: only the input-
parsing and check-assembly behavior is matched (`grade_call_based` parses
each functional-input line with `json.loads` and calls
`method(*gt_inp)` comparing `prediction == gt_out`; this module uses the
looser `ast.literal_eval`, a superset of JSON literals, for the same
newline-per-argument convention). Execution is delegated to this project's
isolated sandbox (`evaluator.scorers.code` / `evaluator.sandbox.run_code`),
not LCB's in-process `exec`/`signal.alarm` harness.

`row` is the pre-parsed intermediate shape produced by the fetcher (Task 2):
    {"test_type": "stdin"|"functional", "fn_name": <str|None>,
     "cases": [{"input": <str>, "output": <str>}, ...]}

Observed raw HF dataset schema (`livecodebench/code_generation_lite`,
`test.jsonl`, fetched directly 2026-07-19 — the packaged
`code_generation_lite.py` loading script no longer runs under current
`datasets` versions, so the fetcher must read the JSONL directly rather
than via `load_dataset(..., trust_remote_code=True)`):

    question_id      str, e.g. "2727" (LeetCode) or "1873_A" (Codeforces)
    question_title, question_content, platform, contest_id, contest_date
    difficulty        str
    starter_code      str; non-empty "class Solution: def <fn>(...):" stub
                      for functional problems, "" for stdin problems
    public_test_cases  JSON-*string* (needs a second json.loads) decoding to
                      list[{"input": str, "output": str, "testtype":
                      "stdin"|"functional"}]  -- note the raw key is
                      "testtype" (no underscore), not "test_type".
    private_test_cases  same logical shape as public_test_cases but wrapped
                      as base64(zlib(pickle.dumps(json_string))) -- observed
                      directly (decoded a real row: `pickle.loads(zlib
                      .decompress(base64.b64decode(...)))` yields the same
                      JSON string as public_test_cases's raw string).
    metadata          JSON-*string*; for functional problems contains
                      {"func_name": "<name>"} (raw key "func_name", not
                      "fn_name"); "{}" for stdin problems.

Task 2's fetcher must therefore rename `testtype` -> `test_type` and
`func_name` -> `fn_name` when building the intermediate row this module
consumes. Confirmed by inspecting real rows (e.g. question_id "2730",
`maximumOr(nums, k)`): functional multi-argument inputs are newline-joined
per-argument literals, e.g. `"[12, 9]\\n1"`, one line per positional arg in
declaration order -- matching this module's `_parse_args` line-splitting.
"""
from __future__ import annotations

import ast


def _parse_args(inp: str):
    """LCB functional inputs are newline-joined Python literals; parse each
    line with literal_eval, falling back to the raw line."""
    args = []
    for line in inp.split("\n"):
        line = line.strip()
        if line == "":
            continue
        try:
            args.append(repr(ast.literal_eval(line)))
        except Exception:
            args.append(repr(line))
    return ", ".join(args)


def build_functional_check(fn_name: str, cases: list[tuple[str, str]]) -> str:
    lines = ["def check(candidate):"]
    for inp, out in cases:
        args = _parse_args(inp)
        try:
            expected = repr(ast.literal_eval(out.strip()))
        except Exception:
            expected = repr(out.strip())
        lines.append(f"    assert candidate({args}) == {expected}")
    lines.append("    return True")
    return "\n".join(lines)


def functional_entry_point(fn_name: str) -> str:
    """The callable `check(candidate)` receives. LiveCodeBench functional
    problems are LeetCode-style: the completion defines `class Solution` and
    `fn_name` is a METHOD, so it must be called on an instance
    (`Solution().fn_name`), not as a bare top-level name (which was HumanEval's
    contract and NameErrors here). Falls back to a top-level function if no
    `Solution` class is defined, so both shapes work."""
    return f"(Solution().{fn_name} if 'Solution' in dir() else {fn_name})"


def normalize_tests(row: dict) -> tuple[dict, ...]:
    cases = row.get("cases", [])
    if row.get("test_type") == "functional":
        fn = row["fn_name"]
        check = build_functional_check(fn, [(c["input"], c["output"]) for c in cases])
        return ({"kind": "pyfunc", "test": check,
                 "entry_point": functional_entry_point(fn)},)
    return tuple(
        {"kind": "stdin", "stdin": c["input"], "expected_stdout": c["output"]}
        for c in cases
    )

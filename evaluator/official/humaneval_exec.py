"""Official HumanEval check-program assembly.

Ported from openai/human-eval `human_eval/execution.py::check_correctness`
(https://github.com/openai/human-eval , commit
`463c980b59e818ace59f6f9803cd92c749ceae61`, MIT License). At that commit,
`check_correctness` builds:

    check_program = (
        problem["prompt"] + completion + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    )

Our models are prompted to return the *full* function definition (not just a
body to append to `problem["prompt"]`), so the extracted `completion` already
contains the signature; the official-equivalent assembly for full-function
completions collapses to `completion + "\n" + test + "\n" +
f"check({entry_point})"`. Only the *program assembly* is vendored here.
Execution is delegated to this project's isolated sandbox
(`evaluator.sandbox.run_code`) rather than upstream's in-process
`exec(check_program, exec_globals)`, to preserve our subprocess isolation
discipline.
"""
from __future__ import annotations


def build_check_program(completion: str, test: str, entry_point: str) -> str:
    return f"{completion}\n\n{test}\n\ncheck({entry_point})\n"

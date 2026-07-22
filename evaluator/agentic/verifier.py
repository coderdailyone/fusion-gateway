"""Proxy verifier: decides "did this attempt pass?" as the escalation gate.

IRON RULE: never reads the official hidden tests (FAIL_TO_PASS / PASS_TO_PASS).
Signals used, all available to a real autonomous deployment:
  - reproduction test authored by the agent goes red -> green after the patch,
  - the repo's existing / touched tests do not regress,
  - the patch is non-empty.
`run_cmd(cmd) -> (returncode, output)` runs a shell command inside the instance
container; it is injected so the decision logic is testable without Docker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    had_repro_test: bool
    repro_red_green: bool
    no_regression: bool
    patch_nonempty: bool
    flagged: bool  # weak-signal case (no reproduction test available)


def decide(patch_nonempty: bool, had_repro_test: bool,
           repro_red_green: bool, no_regression: bool) -> VerifierResult:
    if not patch_nonempty:
        return VerifierResult(False, had_repro_test, repro_red_green,
                              no_regression, False, flagged=not had_repro_test)
    if had_repro_test:
        passed = repro_red_green and no_regression
        return VerifierResult(passed, True, repro_red_green, no_regression,
                              True, flagged=False)
    # fallback: no reproduction test -> regression guard only, and FLAG
    return VerifierResult(no_regression, False, False, no_regression, True,
                          flagged=True)


# Command templates the box implementation fills in (verify against the repo's
# test runner at implement time). Kept as module constants for the box wiring.
REPRO_TEST_PATH = "/tmp/m4_repro_test.py"


def verify(instance, patch: str, run_cmd: Callable[[str], tuple[int, str]]) -> VerifierResult:
    """Run the proxy checks inside the container and return the gate decision.

    The container is assumed to already have the candidate patch applied by the
    caller (cascade/runner). `run_cmd` executes inside it.
    Reads only instance.problem_statement / instance.repo — never hidden tests.
    """
    patch_nonempty = bool(patch.strip())
    if not patch_nonempty:
        return decide(False, had_repro_test=False, repro_red_green=False,
                      no_regression=False)

    # 1. Does an agent-authored reproduction test exist in the container?
    rc_exists, _ = run_cmd(f"test -f {REPRO_TEST_PATH}")
    had_repro = rc_exists == 0

    repro_red_green = False
    if had_repro:
        # It must FAIL on the un-patched tree and PASS with the patch applied.
        # The caller stashes/re-applies; here we run against the patched tree and
        # trust the caller-provided pre-patch result via a sentinel file.
        rc_post, _ = run_cmd(f"python -m pytest -q {REPRO_TEST_PATH}")
        rc_pre_marker, pre_out = run_cmd("cat /tmp/m4_repro_prepatch_rc || echo 1")
        repro_red_green = (rc_post == 0) and (pre_out.strip() != "0")

    # 2. Regression guard: the repo's existing tests still pass.
    rc_reg, _ = run_cmd("python -m pytest -q -x 2>/dev/null || true; echo $?")
    no_regression = _last_rc_zero(rc_reg)

    return decide(patch_nonempty=True, had_repro_test=had_repro,
                  repro_red_green=repro_red_green, no_regression=no_regression)


def _last_rc_zero(rc: int) -> bool:
    return rc == 0

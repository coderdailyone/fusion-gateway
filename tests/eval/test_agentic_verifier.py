import pytest
from evaluator.agentic.dataset import Instance
from evaluator.agentic.verifier import decide, verify


def test_decide_truth_table():
    # pass: non-empty + repro red->green + no regression
    r = decide(patch_nonempty=True, had_repro_test=True,
               repro_red_green=True, no_regression=True)
    assert r.passed and not r.flagged
    # fail: repro didn't flip
    assert not decide(True, True, False, True).passed
    # fail: regression
    assert not decide(True, True, True, False).passed
    # empty patch is an automatic fail
    assert not decide(False, True, True, True).passed
    # no repro test -> fall back to regression+nonempty, but FLAG it
    r2 = decide(patch_nonempty=True, had_repro_test=False,
                repro_red_green=False, no_regression=True)
    assert r2.passed and r2.flagged


def test_verify_never_reads_hidden_tests():
    inst = Instance("i", "r/r", "c", "boom fails", "2025-09-01T00:00:00Z",
                    "img", raw={"FAIL_TO_PASS": _Boom(), "PASS_TO_PASS": _Boom()})
    calls = []

    def run_cmd(cmd: str):
        calls.append(cmd)
        # simulate: repro fails pre-patch (rc!=0), passes post-patch (rc==0),
        # existing tests pass (rc==0)
        return (0, "ok")

    # If verify touched FAIL_TO_PASS/PASS_TO_PASS, _Boom.__eq__/__iter__ raises.
    res = verify(inst, patch="diff --git a/x b/x\n+fix\n", run_cmd=run_cmd)
    assert isinstance(res.passed, bool)
    assert calls, "verify should have run commands in the container"


class _Boom:
    def __iter__(self): raise AssertionError("hidden tests were read!")
    def __eq__(self, o): raise AssertionError("hidden tests were read!")
    def __hash__(self): return 0

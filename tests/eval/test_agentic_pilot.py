from evaluator.agentic.records import AgenticAttempt, read_attempts
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.dataset import Instance
from evaluator.agentic.pilot import run_pilot

PASS = VerifierResult(True, True, True, True, True, False)


def _inst(n):
    return Instance(f"i{n}", "r/r", "c", "p", "2025-09-01T00:00:00Z", "img", {})


def _mk(model, cost):
    return lambda i: AgenticAttempt(i.instance_id, model, "d", "t", 3, cost, "ok", None)


def test_resumes_and_respects_ceiling(tmp_path):
    insts = [_inst(0), _inst(1), _inst(2)]
    # each cheap run costs 0.40; ceiling 1.00 -> only 2 instances fit
    results = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                        strong_run=_mk("claude-opus-4-8", 5.0),
                        verify=lambda i, p: PASS, run_dir=tmp_path, ceiling=1.00)
    assert len(results) == 2                      # stopped before breaching ceiling
    assert len(read_attempts(tmp_path)) == 2      # frozen

    # resume: already-frozen instances are skipped, remaining one now fits
    more = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                     strong_run=_mk("claude-opus-4-8", 5.0),
                     verify=lambda i, p: PASS, run_dir=tmp_path, ceiling=1.00)
    assert [r.instance_id for r in more] == ["i2"]  # only the un-done one

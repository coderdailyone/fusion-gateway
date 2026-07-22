from evaluator.agentic.records import AgenticAttempt, read_attempts
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.dataset import Instance
from evaluator.agentic.pilot import run_pilot

PASS = VerifierResult(True, True, True, True, True, False)


def _inst(n):
    return Instance(f"i{n}", "r/r", "c", "p", "2025-09-01T00:00:00Z", "img", {})


def _mk(model, cost):
    return lambda i: AgenticAttempt(i.instance_id, model, "d", "t", 3, cost, "ok", None)


def test_ceiling_stops_before_breach(tmp_path):
    insts = [_inst(0), _inst(1), _inst(2)]
    results = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                        strong_run=_mk("claude-opus-4-8", 5.0),
                        verify=lambda i, p: PASS, run_dir=tmp_path,
                        ceiling=1.00, worst_case_per_instance=0.5)
    assert len(results) == 2                    # i0,i1 fit; i2 would breach -> stop
    assert len(read_attempts(tmp_path)) == 2    # frozen


def test_resume_counts_prior_spend_and_skips_done(tmp_path):
    insts = [_inst(0), _inst(1), _inst(2)]
    # first period fills i0,i1 (spends 0.80) under ceiling 1.00
    run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
              strong_run=_mk("claude-opus-4-8", 5.0),
              verify=lambda i, p: PASS, run_dir=tmp_path,
              ceiling=1.00, worst_case_per_instance=0.5)

    # resume under the SAME ceiling must REFUSE i2: prior $0.80 counted from disk,
    # so 0.80 + 0.5 margin > 1.00 -> no overspend, no new work (safe across restarts)
    same = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                     strong_run=_mk("claude-opus-4-8", 5.0),
                     verify=lambda i, p: PASS, run_dir=tmp_path,
                     ceiling=1.00, worst_case_per_instance=0.5)
    assert same == []

    # resume with a raised ceiling runs only the un-done i2 (i0,i1 skipped)
    more = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                     strong_run=_mk("claude-opus-4-8", 5.0),
                     verify=lambda i, p: PASS, run_dir=tmp_path,
                     ceiling=2.00, worst_case_per_instance=0.5)
    assert [r.instance_id for r in more] == ["i2"]

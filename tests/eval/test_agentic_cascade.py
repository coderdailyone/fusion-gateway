from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.cascade import run_cascade

PASS = VerifierResult(True, True, True, True, True, False)
FAIL = VerifierResult(False, True, False, True, True, False)


def _attempt(model, patch, cost):
    return AgenticAttempt("i", model, patch, "t.json", 5, cost, "ok", None)


def test_pass_keeps_cheap_and_does_not_call_strong():
    strong_calls = []
    res = run_cascade(
        instance=object(),
        cheap_run=lambda i: _attempt("deepseek-chat", "cheapdiff", 0.05),
        strong_run=lambda i: strong_calls.append(1) or _attempt("x", "y", 9.0),
        verify=lambda i, p: PASS)
    assert res.escalated is False
    assert res.accepted_patch == "cheapdiff"
    assert res.cost_usd == 0.05
    assert res.strong is None
    assert strong_calls == []  # strong never invoked


def test_fail_escalates_and_sums_cost():
    res = run_cascade(
        instance=object(),
        cheap_run=lambda i: _attempt("deepseek-chat", "cheapdiff", 0.05),
        strong_run=lambda i: _attempt("claude-opus-4-8", "opusdiff", 1.20),
        verify=lambda i, p: FAIL)
    assert res.escalated is True
    assert res.accepted_patch == "opusdiff"
    assert round(res.cost_usd, 4) == 1.25
    assert res.strong.model == "claude-opus-4-8"

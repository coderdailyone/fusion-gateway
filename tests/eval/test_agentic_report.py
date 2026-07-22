from scripts.agentic_report import metrics, verifier_agreement


def test_metrics_resolve_rate_and_cost_per_successful():
    resolved = {"i1": True, "i2": False, "i3": True, "i4": False}
    costs = {"i1": 0.10, "i2": 0.20, "i3": 0.30, "i4": 0.40}
    m = metrics(resolved, costs)
    assert m["n"] == 4
    assert m["resolved"] == 2
    assert abs(m["resolve_rate"] - 0.5) < 1e-9
    assert abs(m["total_cost"] - 1.0) < 1e-9
    assert abs(m["cost_per_successful"] - 0.5) < 1e-9  # 1.0 / 2
    assert 0.0 <= m["wilson_lo"] <= m["resolve_rate"] <= m["wilson_hi"] <= 1.0


def test_metrics_zero_resolved_is_inf_cost():
    m = metrics({"i1": False}, {"i1": 0.9})
    assert m["cost_per_successful"] == float("inf")


def test_verifier_agreement_precision_recall():
    # verifier passed i1,i2 ; officially resolved i1,i3
    vp = {"i1": True, "i2": True, "i3": False, "i4": False}
    res = {"i1": True, "i2": False, "i3": True, "i4": False}
    a = verifier_agreement(vp, res)
    assert abs(a["precision"] - 0.5) < 1e-9   # of {i1,i2} passed, 1 resolved
    assert abs(a["recall"] - 0.5) < 1e-9      # of {i1,i3} resolved, 1 caught

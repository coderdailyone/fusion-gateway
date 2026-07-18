from scripts.resample_official import budget_gate, estimate_cost


def test_budget_gate_states():
    assert budget_gate(spent=0.0, next_cost=1.0, ceiling=10.0) == "ok"
    assert budget_gate(spent=8.5, next_cost=0.1, ceiling=10.0) == "warn"   # >=80%
    assert budget_gate(spent=9.99, next_cost=0.1, ceiling=10.0) == "stop"  # would cross 100%


def test_estimate_cost_sums():
    cost_fn = lambda model, i, o: 0.001 * (i + o)
    rows = [("m", 100, 200), ("m", 0, 100)]
    assert abs(estimate_cost(rows, cost_fn) - (0.001*300 + 0.001*100)) < 1e-9

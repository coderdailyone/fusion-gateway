from evaluator.report import ResultRow, aggregate


def test_aggregate():
    rows = [
        ResultRow("t1", "math", "deepseek-chat", True, 0.001),
        ResultRow("t2", "math", "deepseek-chat", False, 0.002),
        ResultRow("t1", "math", "glm-4.6", True, 0.010),
    ]
    agg = aggregate(rows)
    assert agg["deepseek-chat"]["n"] == 2
    assert agg["deepseek-chat"]["accuracy"] == 0.5
    assert abs(agg["deepseek-chat"]["mean_cost_usd"] - 0.0015) < 1e-9
    assert agg["glm-4.6"]["accuracy"] == 1.0

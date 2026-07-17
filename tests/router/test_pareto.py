from evaluator.report import ResultRow
from router.matrix import ResultMatrix
from router.pareto import envelopes, render_report, static_points


def test_static_points_accuracy_and_mean_cost():
    rows = [
        ResultRow("t1", "math", "cheap", True, 0.01),
        ResultRow("t2", "math", "cheap", False, 0.02),
        ResultRow("t1", "math", "strong", True, 0.30),
        ResultRow("t2", "math", "strong", True, 0.34),
    ]
    m = ResultMatrix.from_rows(rows)
    pts = static_points(m, ["t1", "t2"])
    assert pts["cheap"] == (0.5, 0.015)
    assert pts["strong"] == (1.0, 0.32)


def test_static_points_restricted_to_given_task_ids():
    rows = [
        ResultRow("t1", "math", "cheap", True, 0.01),
        ResultRow("t2", "math", "cheap", False, 0.02),
        ResultRow("t3", "math", "cheap", True, 0.03),
    ]
    m = ResultMatrix.from_rows(rows)
    # Only score over t1, t2 -- t3 must not affect accuracy/mean_cost.
    pts = static_points(m, ["t1", "t2"])
    assert pts["cheap"] == (0.5, 0.015)


def test_envelope_true_when_dynamic_curve_dominates_every_static_point():
    # Each static point is dominated by *some* dynamic point:
    # cheap (0.86, 0.014) <= dyn[1] (0.86, 0.01):  acc 0.86>=0.86, cost 0.01<=0.014
    # strong (0.88, 0.317) <= dyn[0] (0.93, 0.05):  acc 0.93>=0.88, cost 0.05<=0.317
    dyn = [
        {"lambda": 0.0, "accuracy": 0.93, "mean_cost": 0.05},
        {"lambda": 1e6, "accuracy": 0.86, "mean_cost": 0.01},
    ]
    stat = {"cheap": (0.86, 0.014), "strong": (0.88, 0.317)}
    e = envelopes(dyn, stat)
    assert e["envelops_all"] is True
    assert set(e["dominated_static"]) == {"cheap", "strong"}
    assert e["undominated_static"] == []


def test_envelope_false_when_a_static_point_has_higher_accuracy_than_every_dynamic_point():
    # "strong" has accuracy 0.88, higher than every dynamic point's accuracy
    # (0.80), so no dynamic point can dominate it regardless of cost.
    dyn = [{"lambda": 0.0, "accuracy": 0.80, "mean_cost": 0.20}]
    stat = {"strong": (0.88, 0.317)}
    e = envelopes(dyn, stat)
    assert e["envelops_all"] is False
    assert e["dominated_static"] == []
    assert e["undominated_static"] == ["strong"]


def test_envelope_dominate_boundary_is_inclusive():
    # accuracy equal and cost equal -> still dominates (>= and <=, not strict).
    dyn = [{"lambda": 0.0, "accuracy": 0.9, "mean_cost": 0.1}]
    stat = {"tie": (0.9, 0.1)}
    e = envelopes(dyn, stat)
    assert e["envelops_all"] is True
    assert e["dominated_static"] == ["tie"]


def test_render_report_contains_verdict_and_is_nonempty():
    dyn = [{"lambda": 0.0, "accuracy": 0.93, "mean_cost": 0.05}]
    stat = {"cheap": (0.86, 0.014), "strong": (0.88, 0.317)}
    e = envelopes(dyn, stat)
    report = render_report(dyn, stat, e)
    assert isinstance(report, str)
    assert report.strip() != ""
    assert "envelops_all" in report or "Envelope" in report or "envelope" in report

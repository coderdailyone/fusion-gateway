from router.policy import route_task, sweep_lambda
from router.train import OOFPredictions
from evaluator.report import ResultRow
from router.matrix import ResultMatrix


def test_route_task_extremes():
    p = {"cheap": 0.6, "strong": 0.9}
    c = {"cheap": 0.001, "strong": 0.10}
    assert route_task(p, c, lam=0.0) == "strong"  # λ=0 -> max proba
    assert route_task(p, c, lam=1e6) == "cheap"  # huge λ -> min cost


def test_sweep_monotone_endpoints():
    rows = [
        ResultRow("t1", "math", "cheap", True, 0.001),
        ResultRow("t1", "math", "strong", True, 0.10),
        ResultRow("t2", "math", "cheap", False, 0.001),
        ResultRow("t2", "math", "strong", True, 0.10),
    ]
    m = ResultMatrix.from_rows(rows)
    oof = OOFPredictions(
        proba={"cheap": {"t1": 0.9, "t2": 0.2}, "strong": {"t1": 0.9, "t2": 0.9}},
        cv_auc={"cheap": 0.7, "strong": 0.7},
    )
    pts = sweep_lambda(oof, m, ["t1", "t2"], [0.0, 1e6])
    lo = next(p for p in pts if p["lambda"] == 1e6)
    hi = next(p for p in pts if p["lambda"] == 0.0)
    assert lo["mean_cost"] <= hi["mean_cost"]  # cheaper end costs no more

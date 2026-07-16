from evaluator.report import ResultRow
from router.matrix import ResultMatrix


def test_from_rows():
    rows = [
        ResultRow("t1", "math", "A", True, 0.001), ResultRow("t1", "math", "B", False, 0.010),
        ResultRow("t2", "math", "A", False, 0.001), ResultRow("t2", "math", "B", True, 0.010),
    ]
    m = ResultMatrix.from_rows(rows)
    assert m.tasks == ["t1", "t2"] and m.models == ["A", "B"]
    assert m.correct["A"]["t1"] is True and m.correct["B"]["t1"] is False
    assert m.cost["B"]["t2"] == 0.010 and m.source["t1"] == "math"

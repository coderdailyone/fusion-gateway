import pytest

from evaluator.suite.types import Task
from evaluator.report import ResultRow
from evaluator.pilot import stratified_subset, analyze


def tasks():
    out = []
    for i in range(80):
        out.append(Task(id=f"a{i}", source="mmlu_pro", problem="", answer="A", tests=(), meta={}))
    for i in range(20):
        out.append(Task(id=f"m{i}", source="math", problem="", answer="1", tests=(), meta={}))
    return out


def test_proportional_and_deterministic():
    a = stratified_subset(tasks(), 20, seed=3)
    b = stratified_subset(tasks(), 20, seed=3)
    assert [t.id for t in a] == [t.id for t in b]     # deterministic
    assert len(a) <= 20
    n_mmlu = sum(1 for t in a if t.source == "mmlu_pro")
    n_math = sum(1 for t in a if t.source == "math")
    assert n_mmlu > n_math                            # 80/20 split reflected
    assert n_math >= 1                                # min one per present source


def test_analyze_go_on_quality():
    rows = [
        ResultRow("t1", "math", "A", True,  0.001), ResultRow("t1", "math", "B", True,  0.010),
        ResultRow("t2", "math", "A", False, 0.001), ResultRow("t2", "math", "B", True,  0.010),
        ResultRow("t3", "math", "A", True,  0.001), ResultRow("t3", "math", "B", False, 0.010),
    ]
    a = analyze(rows)
    assert a["best_static"]["model"] == "A"                 # 2/3 acc, cheaper than B
    assert a["oracle_accuracy"] == 1.0
    assert abs(a["quality_headroom"] - (1.0 - 2/3)) < 1e-9  # ~0.333
    assert abs(a["disagreement_rate"] - 2/3) < 1e-9
    assert a["iso_quality_cost"] == pytest.approx(0.001 + 0.010 + 0.001)  # t2 needs B
    assert a["verdict"] == "GO" and "quality" in a["signals"]


def test_analyze_no_go_when_redundant():
    rows = [
        ResultRow("t1", "math", "A", True,  0.001), ResultRow("t1", "math", "B", True,  0.001),
        ResultRow("t2", "math", "A", False, 0.001), ResultRow("t2", "math", "B", False, 0.001),
    ]
    a = analyze(rows)
    assert a["oracle_accuracy"] == 0.5 and a["quality_headroom"] == 0.0
    assert a["disagreement_rate"] == 0.0
    assert a["verdict"] == "NO_GO"

from evaluator.report import ResultRow
from evaluator.audit.disagreements import find_cases, render

def test_flags_wrong_but_others_right():
    rows = [
        ResultRow("t1", "math", "A", False, 0.0),
        ResultRow("t1", "math", "B", True, 0.0),
        ResultRow("t2", "math", "A", True, 0.0),
        ResultRow("t2", "math", "B", True, 0.0),
    ]
    cases = find_cases(rows, frozen_by_key={})
    keys = {(c.task_id, c.model, c.kind) for c in cases}
    assert ("t1", "A", "wrong_but_others_right") in keys
    # t2: everyone right -> no case
    assert not any(c.task_id == "t2" for c in cases)

def test_render_nonempty():
    rows = [ResultRow("t1", "math", "A", False, 0.0), ResultRow("t1", "math", "B", True, 0.0)]
    out = render(find_cases(rows, frozen_by_key={}))
    assert "t1" in out and "wrong_but_others_right" in out

def test_orders_by_n_other_right_then_task_then_model():
    rows = [
        # t1: 1 other right (B), one wrong (A) -> n_other_right == 1
        ResultRow("t1", "math", "A", False, 0.0),
        ResultRow("t1", "math", "B", True, 0.0),
        # t2: 3 others right (X, Y, Z), one wrong (W) -> n_other_right == 3
        ResultRow("t2", "math", "W", False, 0.0),
        ResultRow("t2", "math", "X", True, 0.0),
        ResultRow("t2", "math", "Y", True, 0.0),
        ResultRow("t2", "math", "Z", True, 0.0),
        # t3: same n_other_right as t1 (1), two wrong models -> tie broken by model
        ResultRow("t3", "math", "M", False, 0.0),
        ResultRow("t3", "math", "N", False, 0.0),
        ResultRow("t3", "math", "P", True, 0.0),
    ]
    cases = find_cases(rows, frozen_by_key={})
    order = [(c.task_id, c.model, c.detail["n_other_right"]) for c in cases]
    # highest n_other_right first (t2/W with 3); then n_other_right==1 group
    # sorted by (task_id, model): t1/A, t3/M, t3/N.
    assert order == [
        ("t2", "W", 3),
        ("t1", "A", 1),
        ("t3", "M", 1),
        ("t3", "N", 1),
    ]

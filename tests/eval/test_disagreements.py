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

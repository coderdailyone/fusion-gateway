from evaluator.suite.types import Task
from evaluator.pilot import stratified_subset


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

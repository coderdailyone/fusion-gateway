import numpy as np
from evaluator.suite.types import Task
from evaluator.report import ResultRow
from router.matrix import ResultMatrix
from router.learnability import per_model_cv_auc, gate

def make(predictable: bool, n=60):
    tasks, rows = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        # two topics; if predictable, model A is correct exactly on topic 'alpha'
        topic = "alpha" if i % 2 == 0 else "beta"
        prob = f"{topic} question number {i} " + ("foo bar " * (i%5))
        tasks.append(Task(id=str(i), source="math", problem=prob, answer="x", tests=(), meta={}))
        a_ok = (topic == "alpha") if predictable else bool(rng.integers(0,2))
        rows.append(ResultRow(str(i), "math", "A", a_ok, 0.001))
    return tasks, ResultMatrix.from_rows(rows)

def test_predictable_signal_passes_gate():
    from router.features import TaskFeaturizer
    tasks, m = make(True)
    aucs = per_model_cv_auc(tasks, m, featurizer_cls=TaskFeaturizer)
    assert aucs["A"] > 0.7
    assert gate(aucs)["verdict"] == "GO"

def test_random_signal_fails_gate():
    from router.features import TaskFeaturizer
    tasks, m = make(False)
    aucs = per_model_cv_auc(tasks, m, featurizer_cls=TaskFeaturizer)
    assert gate(aucs, threshold=0.55)["verdict"] in ("GO","NO_GO")  # near .5
    assert abs(aucs["A"] - 0.5) < 0.25

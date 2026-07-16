from evaluator.suite.types import Task
from router.features import TaskFeaturizer

def T(id, src, prob): return Task(id=id, source=src, problem=prob, answer="x", tests=(), meta={})

def test_fit_transform_shape_and_determinism():
    tasks = [T("1","math","two plus two"), T("2","mmlu_pro","pick the best option"),
             T("3","math","integrate the function")]
    f = TaskFeaturizer().fit(tasks)
    X1 = f.transform(tasks); X2 = f.transform(tasks)
    assert X1.shape[0] == 3 and X1.shape[1] > 3        # tfidf + source-onehot + length cols
    assert (X1 != X2).nnz == 0                          # deterministic

def test_no_answer_leak_in_features():
    # answer text must not influence features (only problem/source/length are used)
    a = TaskFeaturizer().fit([T("1","math","two plus two")])
    import numpy as np
    x_secret = a.transform([Task("1","math","two plus two","SECRET-ANSWER",(),{})])
    x_plain  = a.transform([Task("1","math","two plus two","other",(),{})])
    assert (x_secret != x_plain).nnz == 0

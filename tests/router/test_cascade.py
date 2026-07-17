from router.cascade import cascade_task
from router.matrix import ResultMatrix
from evaluator.report import ResultRow


def m():
    return ResultMatrix.from_rows([
        ResultRow("t1","humaneval","cheap",True,0.001), ResultRow("t1","humaneval","strong",True,0.10),
        ResultRow("t2","humaneval","cheap",False,0.001), ResultRow("t2","humaneval","strong",True,0.10),
        ResultRow("t3","humaneval","cheap",False,0.001), ResultRow("t3","humaneval","strong",False,0.10)])


def test_cascade():
    M = m(); order = ["cheap","strong"]
    assert cascade_task("t1",order,M) == (True, 0.001)          # cheap passes -> stop
    assert cascade_task("t2",order,M) == (True, 0.101)          # cheap fails -> pay both, strong passes
    assert cascade_task("t3",order,M) == (False, 0.101)         # both fail

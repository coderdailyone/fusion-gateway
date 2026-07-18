from evaluator.official.humaneval_exec import build_check_program

def test_assembly_shape():
    prog = build_check_program("def f(x):\n    return x+1",
                               "def check(candidate):\n    assert candidate(1) == 2",
                               "f")
    assert prog.endswith("check(f)\n")
    assert "def f(x):" in prog
    assert "def check(candidate):" in prog
    # completion appears before the test, test before the check call
    assert prog.index("def f(x)") < prog.index("def check") < prog.index("check(f)")

from evaluator.official.livecodebench_exec import normalize_tests, build_functional_check


def test_stdin_case_shape():
    row = {"test_type": "stdin",
           "cases": [{"input": "3\n", "output": "9\n"}]}
    out = normalize_tests(row)
    assert out == ({"kind": "stdin", "stdin": "3\n", "expected_stdout": "9\n"},)


def test_functional_check_asserts_each_case():
    src = build_functional_check("sq", [("3", "9"), ("4", "16")])
    assert "def check(candidate):" in src
    assert "candidate(3) == 9" in src
    assert "candidate(4) == 16" in src


def test_functional_case_shape():
    row = {"test_type": "functional", "fn_name": "sq",
           "cases": [{"input": "3", "output": "9"}]}
    out = normalize_tests(row)
    assert len(out) == 1 and out[0]["kind"] == "pyfunc"
    assert out[0]["entry_point"] == "sq"
    assert "candidate(3) == 9" in out[0]["test"]

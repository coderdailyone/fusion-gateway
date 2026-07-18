from evaluator.audit.references import Reference, build_reference_index

def test_build_index_math_and_humaneval():
    rows = {
        "math": [{"unique_id": "m1", "answer": "42", "solution": "... \\boxed{42}"}],
        "humaneval": [{"task_id": "HE/1", "prompt": "def f():\n", "test": "def check(c): assert c()==1",
                       "entry_point": "f", "canonical_solution": "    return 1"}],
        "mmlu_pro": [{"question_id": 7, "answer": "C"}],
    }
    idx = build_reference_index(rows)
    assert idx["m1"].gold == "42"
    assert "\\boxed{42}" in idx["m1"].solution
    assert idx["HE/1"].canonical_solution == "    return 1"
    assert idx["HE/1"].entry_point == "f"
    assert idx["7"].gold == "C"

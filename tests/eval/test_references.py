from evaluator.audit.references import Reference, build_reference_index
from evaluator.hf_fetchers import extract

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


def test_build_index_gpqa_gold_matches_extract_shuffle():
    # CRITICAL: the reference gold letter MUST equal the shuffled letter
    # `extract` assigns for the SAME row -- otherwise the audit would grade
    # against a different letter than the manifest recorded. The reference
    # index reuses `extract` precisely so these can never drift; this test
    # is CI's guard on that guarantee.
    row = {"Record ID": "g1", "Question": "What is the capital?",
           "Correct Answer": "Paris",
           "Incorrect Answer 1": "London", "Incorrect Answer 2": "Berlin",
           "Incorrect Answer 3": "Rome", "Subdomain": "Geography"}
    tid, rec = extract("gpqa_diamond", row)
    idx = build_reference_index({"gpqa_diamond": [row]})
    assert idx[tid].gold == rec["answer"]
    # sanity: the gold is one of the shuffled option letters, not a raw
    # pass-through of some source field
    assert idx[tid].gold in ("A", "B", "C", "D")


def test_build_index_aime_capitalized_and_lowercase():
    rows = {"aime": [
        {"ID": "2024-II-4", "Problem": "P", "Answer": "204"},          # 2024 casing
        {"id": "2025-1", "problem": "P2", "answer": "17", "year": 2025},  # 2025 casing
    ]}
    idx = build_reference_index(rows)
    assert idx["2024-II-4"].gold == "204"
    assert idx["2025-1"].gold == "17"
    assert idx["2024-II-4"].solution is None  # AIME ships no worked solution


def test_build_index_math_l5_gold_from_boxed_solution():
    rows = {"math_l5": [
        {"unique_id": "ml1", "problem": "p", "solution": "work ... \\boxed{7}",
         "type": "algebra", "level": "Level 5"},
    ]}
    idx = build_reference_index(rows)
    assert idx["ml1"].gold == "7"                 # extracted from the boxed solution
    assert "\\boxed{7}" in idx["ml1"].solution    # solution carried through verbatim

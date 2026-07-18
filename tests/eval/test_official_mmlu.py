from evaluator.official.mmlu_extract import extract_answer

def test_answer_is_paren():
    assert extract_answer("... The answer is (C).") == "C"

def test_answer_colon():
    assert extract_answer("Reasoning...\nAnswer: B") == "B"

def test_last_letter_fallback():
    assert extract_answer("I think it is D") == "D"

def test_unparseable_returns_none():
    assert extract_answer("no letter here at all") is None

import base64
import json
import pickle
import zlib

from evaluator.hf_fetchers import extract, stratum
from evaluator.suite.loader import parse_task


def test_livecodebench_extract_and_parse():
    # a minimal LCB-shaped raw row (fields per the schema recorded in Task 1:
    # note the raw case key is "testtype", not "test_type", and public/private
    # test cases arrive as JSON *strings*).
    raw = {"question_id": "lcb_1", "question_content": "square it",
           "public_test_cases": '[{"input":"3","output":"9","testtype":"functional"}]',
           "private_test_cases": "[]", "metadata": '{"func_name":"sq"}',
           "starter_code": "", "difficulty": "easy", "contest_date": "2024-09-01"}
    tid, rec = extract("livecodebench", raw)
    assert tid == "lcb_1"
    assert rec["difficulty"] == "easy"
    assert rec["release_date"] == "2024-09-01"
    t = parse_task("livecodebench", rec)
    assert t.source == "livecodebench"
    assert t.tests and t.tests[0]["kind"] == "pyfunc"
    assert t.tests[0]["entry_point"] == "sq"


def test_livecodebench_stratum_is_difficulty():
    raw = {"difficulty": "hard"}
    assert stratum("livecodebench", raw) == "hard"


def test_livecodebench_decodes_pickled_private_tests():
    # Real LiveCodeBench rows encode private_test_cases as
    # base64(zlib(pickle.dumps(json_string))), NOT plain JSON (verified in
    # Task 1 against the live HF dataset). Build one here so the fetcher's
    # decode path (not just the public-only path) gets exercised.
    private_cases = json.dumps([{"input": "4", "output": "16", "testtype": "functional"}])
    encoded = base64.b64encode(zlib.compress(pickle.dumps(private_cases))).decode()
    raw = {"question_id": "lcb_2", "question_content": "square it",
           "public_test_cases": '[{"input":"3","output":"9","testtype":"functional"}]',
           "private_test_cases": encoded, "metadata": '{"func_name":"sq"}',
           "starter_code": "", "difficulty": "easy", "contest_date": "2024-09-01"}
    tid, rec = extract("livecodebench", raw)
    assert tid == "lcb_2"
    check_src = rec["tests"][0]["test"]
    # both the public case (3 -> 9) and the decoded private case (4 -> 16)
    # must be present in the assembled check
    assert "candidate(3) == 9" in check_src
    assert "candidate(4) == 16" in check_src


def test_livecodebench_falls_back_to_public_on_bad_private_encoding():
    raw = {"question_id": "lcb_3", "question_content": "square it",
           "public_test_cases": '[{"input":"3","output":"9","testtype":"functional"}]',
           "private_test_cases": "not-valid-base64!!", "metadata": '{"func_name":"sq"}',
           "starter_code": "", "difficulty": "easy", "contest_date": "2024-09-01"}
    tid, rec = extract("livecodebench", raw)
    assert tid == "lcb_3"
    assert "candidate(3) == 9" in rec["tests"][0]["test"]


def test_gpqa_seeded_shuffle_tracks_correct_letter():
    raw = {"Question": "Q?", "Correct Answer": "RIGHT",
           "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2",
           "Incorrect Answer 3": "w3", "Subdomain": "Physics",
           "Record ID": "gpqa_7"}
    tid, rec = extract("gpqa_diamond", raw)
    # the option at rec["answer"] letter must be the correct text
    letters = "ABCD"
    idx = letters.index(rec["answer"])
    assert rec["options"][idx] == "RIGHT"
    # deterministic: same row -> same letter
    _, rec2 = extract("gpqa_diamond", raw)
    assert rec2["answer"] == rec["answer"]
    t = parse_task("gpqa_diamond", rec)
    assert t.source == "gpqa_diamond" and t.answer == rec["answer"]
    assert "RIGHT" in t.problem


def test_gpqa_stratum_is_subdomain():
    raw = {"Subdomain": "Organic Chemistry"}
    assert stratum("gpqa_diamond", raw) == "Organic Chemistry"

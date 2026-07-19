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


def test_livecodebench_functional_flag_without_func_name_falls_back_to_stdin():
    # A row whose cases are flagged testtype="functional" but whose metadata
    # carries no func_name (a data anomaly, not expected in a real LCB row --
    # but must not be allowed to poison scoring if it ever occurs) must NOT
    # produce a {"kind":"pyfunc","entry_point":None} test: code.py's
    # `_run_case` treats any dict with an "entry_point" key as pyfunc no
    # matter its value, so entry_point=None would silently build a
    # `check(None)` call that always fails (None is not callable) instead of
    # surfacing the anomaly. It must fall back to a stdin-shaped test.
    raw = {"question_id": "lcb_4", "question_content": "square it",
           "public_test_cases": '[{"input":"3","output":"9","testtype":"functional"}]',
           "private_test_cases": "[]", "metadata": "{}",  # no func_name
           "starter_code": "", "difficulty": "easy", "contest_date": "2024-09-01"}
    tid, rec = extract("livecodebench", raw)
    assert tid == "lcb_4"
    assert len(rec["tests"]) == 1
    assert rec["tests"][0]["kind"] == "stdin"
    assert "entry_point" not in rec["tests"][0]


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


def test_aime_extract_parse():
    raw = {"ID": "2024-I-1", "Problem": "Find n.", "Answer": "204", "Year": "2024"}
    tid, rec = extract("aime", raw)
    t = parse_task("aime", rec)
    assert t.source == "aime" and t.answer == "204" and "Find n." in t.problem


def test_aime_year_derived_from_id_when_no_year_field():
    # Real Maxwell-Jia/AIME_2024 rows (live-verified in Step 0) have NO "Year"
    # field at all -- year lives only in the "ID" prefix (e.g. "2024-II-4").
    # Answer is also a real int in that dataset, not a string.
    raw = {"ID": "2024-II-4", "Problem": "Evaluate x.", "Answer": 33}
    tid, rec = extract("aime", raw)
    assert tid == "2024-II-4"
    assert rec["answer"] == "33"
    assert rec["year"] == "2024"
    assert stratum("aime", raw) == "2024"


def test_aime_stratum_prefers_explicit_year_field():
    raw = {"ID": "2024-I-1", "Answer": "1", "Problem": "p", "Year": "2025"}
    assert stratum("aime", raw) == "2025"


def test_math_l5_extract_parse():
    raw = {"problem": "Compute.", "solution": "... \\boxed{3}", "level": "Level 5",
           "type": "Algebra"}
    tid, rec = extract("math_l5", raw)
    t = parse_task("math_l5", rec)
    assert t.source == "math_l5" and "Compute." in t.problem


def test_math_l5_answer_is_boxed_content_of_solution():
    raw = {"problem": "Compute.", "solution": "blah \\boxed{42} done", "level": "Level 5",
           "type": "Algebra"}
    tid, rec = extract("math_l5", raw)
    assert rec["answer"] == "42"
    t = parse_task("math_l5", rec)
    assert t.answer == "42"


def test_math_l5_carries_level_for_build_time_filter():
    # Task 5's manifest builder filters level == "Level 5" on the *parsed*
    # record, so "level" must survive extract() and land in Task.meta
    # (parse_task only pops "problem"/"answer").
    raw = {"problem": "Compute.", "solution": "\\boxed{1}", "level": "Level 5",
           "type": "Geometry"}
    tid, rec = extract("math_l5", raw)
    assert rec["level"] == "Level 5"
    t = parse_task("math_l5", rec)
    assert t.meta["level"] == "Level 5"


def test_math_l5_stratum_is_subject_type():
    raw = {"type": "Number Theory"}
    assert stratum("math_l5", raw) == "Number Theory"


def test_math_l5_falls_back_to_answer_field_when_no_boxed_solution():
    raw = {"problem": "Compute.", "solution": "no boxed content here",
           "answer": "7", "level": "Level 5", "type": "Algebra"}
    tid, rec = extract("math_l5", raw)
    assert rec["answer"] == "7"

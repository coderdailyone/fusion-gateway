from evaluator.audit.references import Reference, build_reference_index, load_references
from evaluator.hf_fetchers import extract
from evaluator.suite.manifest import Manifest, SourceSpec

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


def test_fetch_rows_dispatches_hard_sources(tmp_path):
    # `_fetch_rows` must call `fetch(source_name, hf_dataset, revision, split)`
    # -- source_name FIRST -- so the real default (`hf_fetchers.raw_rows`) can
    # dispatch each of the 4 hard sources to its own loading path (gpqa
    # config, math_l5 7-config merge, livecodebench jsonl streaming, aime
    # "+"-merge) instead of the old config-blind `load_dataset(ds, split=,
    # revision=)` that only worked for the 3 standard sources. This test
    # drives the full `load_references` -> `_fetch_rows` -> `build_reference_index`
    # path with a FAKE `fetch` (no network) returning synthetic raw rows per
    # source, and asserts a `Reference` is produced for each of the 4 hard
    # sources -- proving the $0 reference self-test gate (before paid
    # sampling) would not crash on the hard manifest.
    manifest = Manifest(version=1, sources=(
        SourceSpec("gpqa_diamond", "Idavidrein/gpqa", "revA", "train",
                   ("g1",), "sha1"),
        SourceSpec("aime", "Maxwell-Jia/AIME_2024+yentinglin/aime_2025",
                   "r2024+r2025", "train+train", ("2024-I-1",), "sha2"),
        SourceSpec("math_l5", "EleutherAI/hendrycks_math", "r3", "test",
                   ("ml1",), "sha3"),
        SourceSpec("livecodebench", "livecodebench/code_generation_lite",
                   "r4", "test", ("lcb1",), "sha4"),
    ))

    fake_raw_rows = {
        "gpqa_diamond": [{"Record ID": "g1", "Question": "Q?",
                          "Correct Answer": "Right",
                          "Incorrect Answer 1": "a", "Incorrect Answer 2": "b",
                          "Incorrect Answer 3": "c", "Subdomain": "Physics"}],
        "aime": [{"ID": "2024-I-1", "Problem": "P", "Answer": "204",
                  "Year": "2024"}],
        "math_l5": [{"unique_id": "ml1", "problem": "p",
                     "solution": "work ... \\boxed{7}",
                     "type": "Algebra", "level": "Level 5"}],
        "livecodebench": [{"question_id": "lcb1",
                           "question_content": "square it"}],
    }

    def fake_fetch(source_name, hf_dataset, revision, split):
        # No network: dispatch purely off source_name, mirroring what a real
        # raw_rows(source_name, hf_dataset, revision, split) call signature
        # requires from _fetch_rows.
        return fake_raw_rows[source_name]

    cache_path = str(tmp_path / "references.json")
    idx = load_references(manifest, cache_path=cache_path, fetch=fake_fetch)

    assert idx["g1"].source == "gpqa_diamond"
    assert idx["g1"].gold in ("A", "B", "C", "D")
    assert idx["2024-I-1"].source == "aime"
    assert idx["2024-I-1"].gold == "204"
    assert idx["ml1"].source == "math_l5"
    assert idx["ml1"].gold == "7"
    assert idx["lcb1"].source == "livecodebench"

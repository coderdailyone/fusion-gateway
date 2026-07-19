"""HuggingFace fetchers: map raw dataset rows into the record shape that
`evaluator.suite.loader.parse_task` expects, and provide a `Fetcher` per source
for `load_suite`.

`datasets` is imported lazily inside functions so this module stays importable
without the `eval` extra installed (only the actual fetch needs it).
"""
from __future__ import annotations

import hashlib

from evaluator.suite.manifest import SourceSpec


def extract(source_name: str, row: dict) -> tuple[str, dict]:
    """Raw HF row -> (task_id, record) in parse_task's expected shape."""
    if source_name == "mmlu_pro":
        tid = str(row["question_id"])
        return tid, {"id": tid, "question": row["question"],
                     "options": list(row["options"]), "answer": row["answer"]}
    if source_name == "math":
        tid = str(row["unique_id"])
        return tid, {"id": tid, "problem": row["problem"], "answer": row["answer"]}
    if source_name == "humaneval":
        tid = str(row["task_id"])
        return tid, {"id": tid, "prompt": row["prompt"],
                     "test": row["test"], "entry_point": row["entry_point"]}
    if source_name == "livecodebench":
        import json
        from evaluator.official.livecodebench_exec import normalize_tests

        tid = str(row["question_id"])
        meta = json.loads(row.get("metadata") or "{}")
        fn = meta.get("func_name")
        raw_cases = json.loads(row.get("public_test_cases") or "[]")
        raw_cases += _decode_private_test_cases(row.get("private_test_cases"))
        is_functional = bool(fn) or any(
            c.get("testtype") == "functional" for c in raw_cases
        )
        test_type = "functional" if is_functional else "stdin"
        cases = [{"input": c["input"], "output": c["output"]} for c in raw_cases]
        tests = normalize_tests(
            {"test_type": test_type, "fn_name": fn, "cases": cases}
        )
        return tid, {
            "id": tid,
            "prompt": row["question_content"],
            "tests": [dict(t) for t in tests],
            "difficulty": row.get("difficulty", "?"),
            "release_date": str(row.get("contest_date", "")),
        }
    if source_name == "gpqa_diamond":
        import random as _random

        tid = str(row.get("Record ID") or row.get("id") or
                  hashlib.sha256(row["Question"].encode()).hexdigest()[:16])
        correct = row["Correct Answer"].strip()
        options = [correct,
                   row["Incorrect Answer 1"].strip(),
                   row["Incorrect Answer 2"].strip(),
                   row["Incorrect Answer 3"].strip()]
        # Seed derived from the STABLE question id (not global `random`) so
        # the manifest's content_sha pins the exact letter<->option mapping
        # and re-running extract on the same row is byte-identical.
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        order = list(range(4))
        _random.Random(seed).shuffle(order)
        shuffled = [options[i] for i in order]
        answer_letter = "ABCD"[shuffled.index(correct)]
        return tid, {"id": tid, "question": row["Question"],
                     "options": shuffled, "answer": answer_letter,
                     "subject": str(row.get("Subdomain", "?"))}
    if source_name == "aime":
        tid = str(row.get("ID") or row.get("id"))
        return tid, {"id": tid, "problem": row["Problem"],
                     "answer": str(row["Answer"]).strip(), "year": _aime_year(row)}
    if source_name == "math_l5":
        tid = str(row.get("unique_id") or
                  hashlib.sha256(row["problem"].encode()).hexdigest()[:16])
        # gold answer = boxed content of the solution (reuse the math extractor)
        from evaluator.scorers.math import _find_last_boxed
        gold = _find_last_boxed(row.get("solution", "")) or row.get("answer", "")
        return tid, {"id": tid, "problem": row["problem"],
                     "answer": (gold or "").strip(), "subject": str(row.get("type", "?")),
                     "level": str(row.get("level", ""))}  # carried for the build-time L5 filter
    raise ValueError(f"unknown source: {source_name!r}")


def _aime_year(row: dict) -> str:
    """AIME year: prefer an explicit "Year" field; real Maxwell-Jia/AIME_2024
    rows (live-verified) have NONE -- year lives only in the "ID" prefix
    (e.g. "2024-II-4" -> "2024")."""
    if row.get("Year"):
        return str(row["Year"])
    tid = str(row.get("ID") or row.get("id") or "")
    prefix = tid.split("-", 1)[0]
    return prefix if prefix.isdigit() else "?"


def _decode_private_test_cases(priv) -> list[dict]:
    """Decode LiveCodeBench's private test cases.

    Unlike `public_test_cases` (a plain JSON string), upstream encodes
    `private_test_cases` as base64(zlib(pickle.dumps(json_string))) --
    verified directly against the pinned, trusted
    `livecodebench/code_generation_lite` HF dataset in Task 1. `pickle.loads`
    below runs only here, at suite-build time on this isolated eval box, over
    that pinned trusted dataset -- never on model output or any other
    untrusted input -- so it does not carry pickle's usual untrusted-data risk.

    If a row's private cases fail to decode (encoding drift, truncation,
    etc.) we fall back to public-only tests for that row rather than
    aborting the whole build.
    """
    if not priv or priv == "[]":
        return []
    try:
        import base64
        import pickle
        import zlib

        decoded = pickle.loads(zlib.decompress(base64.b64decode(priv)))
        if isinstance(decoded, (bytes, str)):
            import json as _json

            decoded = _json.loads(decoded)
        return list(decoded)
    except Exception:
        return []


def stratum(source_name: str, row: dict) -> str:
    """The stratification label for a row (used for proportional sampling)."""
    if source_name == "mmlu_pro":
        return str(row.get("category", "?"))
    if source_name == "math":
        return str(row.get("level", "?"))
    if source_name == "livecodebench":
        return str(row.get("difficulty", "?"))
    if source_name == "gpqa_diamond":
        return str(row.get("Subdomain", "?"))
    if source_name == "aime":
        return _aime_year(row)
    if source_name == "math_l5":
        return str(row.get("type", "?"))
    return "all"  # humaneval: single stratum


def load_id_map(source_name: str, hf_dataset: str, revision: str, split: str):
    """Load the full split at a pinned revision; return (id->record, id->stratum)."""
    from datasets import load_dataset  # lazy: needs the eval extra
    ds = load_dataset(hf_dataset, split=split, revision=revision)
    id_map: dict[str, dict] = {}
    strata: dict[str, str] = {}
    for row in ds:
        tid, rec = extract(source_name, row)
        id_map[tid] = rec
        strata[tid] = stratum(source_name, row)
    return id_map, strata


def make_fetcher(source_name: str):
    """Return a `Fetcher` (SourceSpec -> list[dict]) for load_suite: it loads the
    pinned dataset and returns records for spec.task_ids, in that order."""
    def fetch(spec: SourceSpec) -> list[dict]:
        id_map, _ = load_id_map(source_name, spec.hf_dataset, spec.hf_revision, spec.split)
        return [id_map[tid] for tid in spec.task_ids]
    return fetch

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
        if is_functional and not fn:
            # A row whose cases are flagged testtype="functional" but whose
            # metadata carries no func_name (a data anomaly) would otherwise
            # make normalize_tests emit {"kind":"pyfunc","entry_point":None}.
            # code.py's `_run_case` treats ANY dict with an "entry_point" key
            # as pyfunc regardless of its value, so a None entry_point does
            # not raise -- it silently builds `check(None)` and always fails
            # (candidate is the None object, not callable), permanently and
            # confusingly, rather than surfacing the anomaly. Fall back to
            # stdin scoring instead, matching the private-test-case-decode
            # fallback's discipline (degrade gracefully, don't abort/crash).
            is_functional = False
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


# Per-source HF config name, for sources whose dataset repo requires one
# (e.g. GPQA ships gpqa_diamond/gpqa_main/gpqa_extended/gpqa_experts as
# separate configs of the same repo). Looked up ONLY inside `raw_rows`'s
# default branch, so a source's config only needs to be declared once here
# -- `build_hard` and `make_fetcher` no longer pass a `config` kwarg through
# `load_id_map` at all. Absent for standard-tier sources -> `HARD_CONFIG.get
# (name)` is None -> `raw_rows`'s default branch is byte-identical to the
# pre-hard-tier `load_dataset(hf_dataset, split=split, revision=revision)`.
HARD_CONFIG = {"gpqa_diamond": "gpqa_diamond"}

# EleutherAI/hendrycks_math (the live parquet mirror Task 5 pins in place of
# the dead hendrycks/competition_math) ships one config per subject, not a
# single flat split -- live-verified in Task 4.
_MATH_L5_CONFIGS = (
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
)


def raw_rows(source_name: str, hf_dataset: str, revision: str, split: str) -> list[dict]:
    """Return the RAW HF rows for a source at a pinned revision/split -- the
    ONE dataset-loading path shared by `load_id_map` (suite build/load time)
    and `evaluator.audit.references._fetch_rows` (the $0 reference
    side-channel, gate before paid sampling). Every source-specific loading
    quirk lives here so both callers agree on exactly what a "row" is.

    Standard-tier sources (mmlu_pro/math/humaneval) fall through to the
    default branch with `HARD_CONFIG.get(name)` -> None, so the call there is
    byte-identical to the pre-hard-tier `load_dataset(hf_dataset, split=split,
    revision=revision)` -- the standard tier is unaffected by everything
    added here.

    Hard-tier special cases (live-verified while implementing Tasks 1-5 of
    the M2d hard tier; `datasets.load_dataset(..., trust_remote_code=True)`
    is dead for loading-script datasets on the current `datasets` release):
    - "gpqa_diamond": needs the "gpqa_diamond" config (`HARD_CONFIG`), passed
      to `load_dataset` in the default branch below.
    - "livecodebench": streams the hub's raw jsonl shards directly instead
      of `load_dataset` (see `_livecodebench_raw_rows`).
    - "math_l5": `EleutherAI/hendrycks_math` has no single flat split --
      iterates its 7 subject configs and merges them (see
      `_math_l5_raw_rows`).
    - "aime": `hf_dataset`/`revision`/`split` may each be a "+"-joined pair
      (built by `build_hard`) to fold AIME 2025 into the same "aime" source
      as AIME 2024 (see `_aime_merged_raw_rows`) -- kept as one source, not
      a 5th HARD_SOURCES entry, since the hard-tier name set is fixed at 4
      names.
    """
    if source_name == "livecodebench":
        return _livecodebench_raw_rows(hf_dataset, revision, split)
    if source_name == "math_l5":
        return _math_l5_raw_rows(hf_dataset, revision, split)
    if source_name == "aime" and "+" in hf_dataset:
        return _aime_merged_raw_rows(hf_dataset, revision, split)

    from datasets import load_dataset  # lazy: needs the eval extra
    config = HARD_CONFIG.get(source_name)
    kwargs = {"split": split, "revision": revision}
    ds = load_dataset(hf_dataset, config, **kwargs) if config is not None else load_dataset(hf_dataset, **kwargs)
    return list(ds)


def _math_l5_raw_rows(hf_dataset: str, revision: str, split: str) -> list[dict]:
    """`EleutherAI/hendrycks_math` ships one parquet config per subject
    (no single flat split) -- iterate all 7 known configs and merge. The
    Level-5 filter itself is applied later, by `build_hard`, over the
    merged (unfiltered) rows this returns."""
    from datasets import load_dataset  # lazy: needs the eval extra
    rows: list[dict] = []
    for cfg in _MATH_L5_CONFIGS:
        ds = load_dataset(hf_dataset, cfg, split=split, revision=revision)
        rows.extend(list(ds))
    return rows


def _aime_merged_raw_rows(hf_dataset: str, revision: str, split: str) -> list[dict]:
    """"+"-joined composite for the "aime" source: `hf_dataset`, `revision`
    and `split` are each two "+"-joined halves, in (2024, 2025) order --
    built by `build_hard`. Folds AIME 2025 into the same source as AIME
    2024 rather than adding a 5th HARD_SOURCES entry.

    `yentinglin/aime_2025`'s columns are lowercase (`id`/`problem`/`answer`/
    `year`), unlike `Maxwell-Jia/AIME_2024`'s capitalized `ID`/`Problem`/
    `Answer` -- live-verified in Task 5. `extract("aime", ...)` only reads
    the capitalized keys (Task 4's contract), so the 2025 rows are
    normalized to that shape here rather than teaching `extract` two
    casings for one source.
    """
    hf_2024, hf_2025 = hf_dataset.split("+")
    rev_2024, rev_2025 = revision.split("+")
    split_2024, split_2025 = split.split("+")
    rows = list(raw_rows("aime", hf_2024, rev_2024, split_2024))

    from datasets import load_dataset  # lazy: needs the eval extra
    ds_2025 = load_dataset(hf_2025, "default", split=split_2025, revision=rev_2025)
    for row in ds_2025:
        rows.append({"ID": f"2025-{row.get('id')}", "Problem": row["problem"],
                      "Answer": row["answer"], "Year": str(row.get("year") or "2025")})
    return rows


def _livecodebench_raw_rows(hf_dataset: str, revision: str, split: str) -> list[dict]:
    """`load_dataset(..., trust_remote_code=True)` fails on the current
    `datasets` release (loading-script support removed). Streams the raw
    dataset instead: `HfApi.list_repo_files` (live-verified in Task 1/5)
    shows upstream ships `livecodebench/code_generation_lite`'s "test"
    split as N chronological jsonl shards (`test.jsonl`, `test2.jsonl`, ...,
    `test6.jsonl`), not one file, each covering a later contest-date range
    than the last -- so shards are discovered by pattern (not hardcoded to
    6) and concatenated via `hf_hub_download` + line-by-line JSON parsing.

    Note: these shards are individually 100MB-1.2GB (the corpus embeds
    base64+zlib+pickle-encoded private test cases per row) -- a full build
    downloads several GB total. This is expected to be slow/best-effort in
    a bandwidth-constrained environment; see the M2d Task 5 report.
    """
    import json
    import re

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(hf_dataset, repo_type="dataset", revision=revision)
    pattern = re.compile(rf"^{re.escape(split)}(\d*)\.jsonl$")
    matches = [(f, pattern.match(f)) for f in files]
    shards = sorted((f for f, m in matches if m),
                     key=lambda f: int(pattern.match(f).group(1) or 0))
    if not shards:
        raise FileNotFoundError(f"no {split}*.jsonl shard in {hf_dataset}@{revision}")

    rows: list[dict] = []
    for shard in shards:
        path = hf_hub_download(repo_id=hf_dataset, filename=shard,
                                repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def load_id_map(source_name: str, hf_dataset: str, revision: str, split: str):
    """Load the full split at a pinned revision; return (id->record, id->stratum).

    Thin wrapper around `raw_rows` + `extract`/`stratum` per row -- the
    dataset-loading quirks (gpqa config, LCB jsonl streaming, math_l5
    multi-config merge, aime "+"-merge) all live in `raw_rows` so this stays
    the single place `extract`/`stratum` are applied, for every source.
    """
    id_map: dict[str, dict] = {}
    strata: dict[str, str] = {}
    for row in raw_rows(source_name, hf_dataset, revision, split):
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

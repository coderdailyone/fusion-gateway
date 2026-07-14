"""HuggingFace fetchers: map raw dataset rows into the record shape that
`evaluator.suite.loader.parse_task` expects, and provide a `Fetcher` per source
for `load_suite`.

`datasets` is imported lazily inside functions so this module stays importable
without the `eval` extra installed (only the actual fetch needs it).
"""
from __future__ import annotations

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
    raise ValueError(f"unknown source: {source_name!r}")


def stratum(source_name: str, row: dict) -> str:
    """The stratification label for a row (used for proportional sampling)."""
    if source_name == "mmlu_pro":
        return str(row.get("category", "?"))
    if source_name == "math":
        return str(row.get("level", "?"))
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

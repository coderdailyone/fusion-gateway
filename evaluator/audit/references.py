"""Side-channel loader for dataset reference answers, for the audit only.

Independent of `load_suite`: it reads the SAME pinned dataset revision as the
manifest but returns the reference fields the frozen suite records omit
(math `solution`, humaneval `canonical_solution`). It never alters or re-hashes
the frozen suite. Results are cached to JSON so re-runs are offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Reference:
    task_id: str
    source: str
    gold: str | None = None
    solution: str | None = None
    canonical_solution: str | None = None
    prompt: str | None = None
    test: str | None = None
    entry_point: str | None = None


def build_reference_index(rows_by_source: dict[str, list[dict]]) -> dict[str, Reference]:
    idx: dict[str, Reference] = {}
    for source, rows in rows_by_source.items():
        for row in rows:
            if source == "mmlu_pro":
                tid = str(row["question_id"])
                idx[tid] = Reference(tid, source, gold=str(row["answer"]))
            elif source == "math":
                tid = str(row["unique_id"])
                idx[tid] = Reference(tid, source, gold=str(row["answer"]),
                                     solution=row.get("solution"))
            elif source == "humaneval":
                tid = str(row["task_id"])
                idx[tid] = Reference(tid, source, prompt=row.get("prompt"),
                                     test=row.get("test"), entry_point=row.get("entry_point"),
                                     canonical_solution=row.get("canonical_solution"))
    return idx


def _fetch_rows(manifest, fetch) -> dict[str, list[dict]]:
    if fetch is None:
        from datasets import load_dataset  # lazy: only the real path needs it
        fetch = lambda ds, split, revision: list(load_dataset(ds, split=split, revision=revision))
    rows_by_source: dict[str, list[dict]] = {}
    for s in manifest.sources:
        rows_by_source[s.name] = fetch(s.hf_dataset, s.split, s.hf_revision)
    return rows_by_source


def load_references(manifest, cache_path: str = "runs/cache/references.json",
                    fetch=None) -> dict[str, Reference]:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        return {tid: Reference(**rec) for tid, rec in data.items()}
    idx = build_reference_index(_fetch_rows(manifest, fetch))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({tid: asdict(r) for tid, r in idx.items()}, f)
    return idx

"""Side-channel loader for dataset reference answers, for the audit only.

Independent of `load_suite`: it reads the SAME pinned dataset revision as the
manifest but returns the reference fields the frozen suite records omit
(math `solution`, humaneval `canonical_solution`). It never alters or re-hashes
the frozen suite. Results are cached to JSON so re-runs are offline.
"""
from __future__ import annotations

import hashlib
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
            elif source == "gpqa_diamond":
                # Gold here is the SHUFFLED answer letter, a function of the
                # raw options + a seed derived from the (stable) task id --
                # not a plain raw-field pass-through like mmlu_pro/math. Reuse
                # `hf_fetchers.extract`'s own gpqa_diamond branch rather than
                # re-deriving the shuffle here, so this side-channel index
                # always agrees with whatever the manifest itself recorded
                # (a hand-duplicated shuffle formula could silently drift).
                from evaluator.hf_fetchers import extract
                tid, rec = extract(source, row)
                idx[tid] = Reference(tid, source, gold=str(rec["answer"]))
            elif source == "aime":
                # Maxwell-Jia/AIME_2024's raw columns are "ID"/"Answer"
                # (capitalized); yentinglin/aime_2025's are lowercase
                # "id"/"answer" (live-verified, Task 5) -- the merged "aime"
                # source draws rows from both, so accept either casing here.
                # No `solution` field exists in either dataset (AIME ships
                # bare final answers, no worked-solution text), so
                # `_synth_output`'s math-like branch always falls back to
                # `\\boxed{gold}` for this source.
                tid = str(row.get("ID") or row.get("id"))
                idx[tid] = Reference(tid, source,
                                     gold=str(row.get("Answer") or row.get("answer") or "").strip())
            elif source == "math_l5":
                # EleutherAI/hendrycks_math has no ready-made "answer"
                # column for every row; gold = the last \boxed{...} in the
                # solution (falling back to a raw "answer" field if
                # present) -- mirrors `hf_fetchers.extract`'s math_l5
                # branch exactly, reusing the same boxed-extraction helper
                # the "math" scorer itself uses.
                from evaluator.scorers.math import _find_last_boxed
                tid = str(row.get("unique_id") or
                          hashlib.sha256(row["problem"].encode()).hexdigest()[:16])
                gold = _find_last_boxed(row.get("solution", "")) or row.get("answer", "")
                idx[tid] = Reference(tid, source, gold=str(gold).strip(),
                                     solution=row.get("solution"))
            elif source == "livecodebench":
                # `code_generation_lite` ships no gold/canonical completion
                # at all -- only the problem statement and test cases --
                # so `canonical_solution` is left unset here (default
                # None). `_synth_output`'s livecodebench branch documents
                # that this makes its reference self-test a best-effort
                # smoke rather than a "gold recognized" guarantee.
                tid = str(row["question_id"])
                idx[tid] = Reference(tid, source, prompt=row.get("question_content"))
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

"""Build the locked benchmark manifest by sampling HuggingFace datasets.

Run once (network required) to produce `configs/suite.manifest.json`:
    .venv/bin/python -m evaluator.build_manifest

Sampling is seeded and stratified (proportional per subject/difficulty), so the
manifest is reproducible. Each source records the dataset's pinned commit sha,
the selected task ids, and a content SHA-256 over the exact records — which
`load_suite` re-verifies at load time.

`datasets` / `huggingface_hub` are imported lazily so this module stays
importable (and `stratified_sample` unit-testable) without the eval extra.
"""
from __future__ import annotations

import random
from collections import defaultdict

from evaluator.suite.manifest import SourceSpec, Manifest, content_sha, save
from evaluator.hf_fetchers import load_id_map

# (source_name, hf_dataset, split, sample_size)
SOURCES = [
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "test", 600),
    ("math", "HuggingFaceH4/MATH-500", "test", 300),
    ("humaneval", "openai/openai_humaneval", "test", 164),
]
DEFAULT_SEED = 1234


def stratified_sample(ids: list[str], strata: dict[str, str], n: int, seed: int) -> list[str]:
    """Pick ~n ids, allocated proportionally across strata; deterministic for a seed.

    Pure function (no network) so it can be unit-tested. Returns a sorted list.
    """
    if n <= 0 or not ids:
        return []
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for tid in ids:
        groups[strata[tid]].append(tid)
    total = len(ids)
    picked: list[str] = []
    for stratum_key in sorted(groups):
        members = sorted(groups[stratum_key])
        rng.shuffle(members)
        k = max(1, round(n * len(members) / total))
        picked.extend(members[:k])
    picked = sorted(set(picked))
    if len(picked) > n:
        rng2 = random.Random(seed + 1)
        rng2.shuffle(picked)
        picked = sorted(picked[:n])
    return picked


def build(seed: int = DEFAULT_SEED) -> Manifest:
    from huggingface_hub import dataset_info  # lazy: needs the eval extra
    sources: list[SourceSpec] = []
    for name, hf, split, n in SOURCES:
        revision = dataset_info(hf).sha
        id_map, strata = load_id_map(name, hf, revision, split)
        all_ids = list(id_map)
        sampled = stratified_sample(all_ids, strata, min(n, len(all_ids)), seed)
        records = [id_map[tid] for tid in sampled]
        sha = content_sha(records)
        sources.append(SourceSpec(name, hf, revision, split, tuple(sampled), sha))
    return Manifest(version=1, sources=tuple(sources))


def main() -> None:
    manifest = build()
    save(manifest, "configs/suite.manifest.json")
    total = sum(len(s.task_ids) for s in manifest.sources)
    print(f"wrote configs/suite.manifest.json ({total} tasks)")
    for s in manifest.sources:
        print(f"  {s.name}: {len(s.task_ids)} tasks @ {s.hf_revision[:12]} sha={s.content_sha[:12]}")


if __name__ == "__main__":
    main()

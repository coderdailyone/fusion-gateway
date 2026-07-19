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


# --- hard tier (M2d) ----------------------------------------------------
#
# (source_name, hf_dataset, split, sample_size, filter_predicate|None)
# filter_predicate: (PARSED record from extract) -> bool, applied before
# sampling. `load_id_map` runs extract, so predicates read the record
# fields extract carries (livecodebench -> "release_date"; math_l5 ->
# "level").
#
# Dataset ids below are the LIVE-VERIFIED replacements found while
# implementing Tasks 1-4 (M2d) -- the original plan's ids were written
# from memory and are wrong/dead for 3 of these 4 sources:
#   - livecodebench: `load_dataset(..., trust_remote_code=True)` is dead
#     (loading-script support removed from the current `datasets`
#     release); `load_id_map` streams the hub's raw jsonl shards directly
#     for this source instead (see evaluator.hf_fetchers).
#   - gpqa_diamond: `Idavidrein/gpqa` is a GATED dataset -- building this
#     source needs an HF_TOKEN in the environment. The "gpqa_diamond"
#     config is passed automatically via `hf_fetchers.HARD_CONFIG`.
#   - math_l5: `hendrycks/competition_math` is dead (same loading-script
#     issue); pinned to `EleutherAI/hendrycks_math` instead, a 7-subject-
#     config parquet mirror with identical `problem`/`solution`/`level`/
#     `type` fields -- `load_id_map` iterates all 7 configs internally.
#   - aime: `Maxwell-Jia/AIME_2024` (as planned, 30 rows) + AIME 2025
#     (`yentinglin/aime_2025`, config "default", split "train"; live-
#     verified 30 rows, fields `id`/`problem`/`answer`/`year`) merged into
#     this one source by `build_hard` below -- kept as a single "aime"
#     entry rather than a 5th HARD_SOURCES name, since the hard-tier name
#     set is fixed at exactly 4 (`test_hard_sources_declared`).
HARD_SOURCES = [
    ("livecodebench", "livecodebench/code_generation_lite", "test", 150,
     lambda rec: str(rec.get("release_date", "")) >= "2024-08-01"),
    ("gpqa_diamond", "Idavidrein/gpqa", "train", 198, None),
    ("aime", "Maxwell-Jia/AIME_2024", "train", 60, None),  # + AIME 2025, merged below
    ("math_l5", "EleutherAI/hendrycks_math", "test", 250,
     lambda rec: rec.get("level") == "Level 5"),
]

# AIME 2025 half of the merged "aime" source (see HARD_SOURCES note above
# and `hf_fetchers._load_aime_merged_id_map`).
AIME_2025_HF = "yentinglin/aime_2025"
AIME_2025_SPLIT = "train"


def build_hard(seed: int = DEFAULT_SEED) -> Manifest:
    from huggingface_hub import dataset_info  # lazy: needs the eval extra

    from evaluator.hf_fetchers import HARD_CONFIG

    sources: list[SourceSpec] = []
    for name, hf, split, n, pred in HARD_SOURCES:
        revision = dataset_info(hf).sha
        if name == "aime":
            revision_2025 = dataset_info(AIME_2025_HF).sha
            hf = f"{hf}+{AIME_2025_HF}"
            revision = f"{revision}+{revision_2025}"
            split = f"{split}+{AIME_2025_SPLIT}"
            id_map, strata = load_id_map(name, hf, revision, split)
        else:
            id_map, strata = load_id_map(name, hf, revision, split, config=HARD_CONFIG.get(name))
        ids = [tid for tid in id_map if pred is None or pred(id_map[tid])]
        sampled = stratified_sample(ids, strata, min(n, len(ids)), seed)
        records = [id_map[tid] for tid in sampled]
        sources.append(SourceSpec(name, hf, revision, split, tuple(sampled),
                                  content_sha(records)))
    return Manifest(version=1, sources=tuple(sources))


def main() -> None:
    import sys
    if "--hard" in sys.argv:
        manifest, path = build_hard(), "configs/suite.hard.manifest.json"
    else:
        manifest, path = build(), "configs/suite.manifest.json"
    save(manifest, path)
    total = sum(len(s.task_ids) for s in manifest.sources)
    print(f"wrote {path} ({total} tasks)")
    for s in manifest.sources:
        print(f"  {s.name}: {len(s.task_ids)} tasks @ {s.hf_revision[:12]} sha={s.content_sha[:12]}")


if __name__ == "__main__":
    main()

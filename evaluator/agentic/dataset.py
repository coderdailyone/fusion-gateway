"""Load a pinned SWE-bench-Live snapshot, filter for de-contamination, sample.

The hidden-test fields (FAIL_TO_PASS/PASS_TO_PASS) are carried untouched inside
Instance.raw for the official grader ONLY. Nothing in the agentic pipeline other
than grade.py reads them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Pin at implement time (see implementer note in the plan).
SNAPSHOT_REVISION = "REPLACE_WITH_PINNED_REVISION"
# A date safely after every pool model's training cutoff (verify current cutoffs).
DEFAULT_MIN_CREATED = "2025-06-01"


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    created_at: str
    image_ref: str
    raw: dict


def filter_by_date(rows: list[dict], min_created: str) -> list[dict]:
    """Keep only rows created strictly after min_created (ISO date compare)."""
    return [r for r in rows if r["created_at"][:10] > min_created[:10]]


def load_instances(rows: list[dict]) -> list[Instance]:
    return [
        Instance(
            instance_id=r["instance_id"], repo=r["repo"],
            base_commit=r["base_commit"], problem_statement=r["problem_statement"],
            created_at=r["created_at"], image_ref=r["image_ref"], raw=r)
        for r in rows
    ]


def _key(seed: str, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def stratified_sample(instances: list[Instance], k: int, seed: str) -> list[Instance]:
    """Deterministic round-robin over repos, so a single repo can't dominate.

    Within each repo, instances are ordered by a seeded hash of their id; then
    one instance is picked per repo, cycling through repos in stable (sorted)
    order, until k instances are chosen or every repo's pool is exhausted.
    """
    by_repo: dict[str, list[Instance]] = {}
    for inst in sorted(instances, key=lambda i: _key(seed, i.instance_id)):
        by_repo.setdefault(inst.repo, []).append(inst)

    queues = {repo: iter(pool) for repo, pool in by_repo.items()}
    repos = sorted(by_repo)

    picked: list[Instance] = []
    remaining_repos = list(repos)
    while remaining_repos and len(picked) < k:
        next_round = []
        for repo in remaining_repos:
            nxt = next(queues[repo], None)
            if nxt is None:
                continue
            picked.append(nxt)
            next_round.append(repo)
            if len(picked) >= k:
                break
        remaining_repos = next_round

    return picked[:k]

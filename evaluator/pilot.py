import random
from collections import defaultdict
from evaluator.suite.types import Task


def stratified_subset(tasks: list[Task], n: int, seed: int) -> list[Task]:
    """
    Select a stratified subset of tasks proportionally by source.

    Groups tasks by source, allocates roughly proportionally (at least 1 per source),
    shuffles each group with a seeded RNG, and returns a deterministically ordered result.

    Args:
        tasks: List of Task objects to sample from.
        n: Target size of subset.
        seed: Random seed for deterministic shuffling.

    Returns:
        List of at most n tasks, stratified by source, sorted by (source, id).
    """
    if not tasks:
        return []

    # Group tasks by source
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.source].append(task)

    total = len(tasks)

    # Calculate allocations: proportional with at least 1 per source
    allocations: dict[str, int] = {}
    for source in sorted(groups.keys()):
        # Proportional allocation, at least 1, not more than available
        alloc = round(n * len(groups[source]) / total)
        alloc = max(1, min(alloc, len(groups[source])))
        allocations[source] = alloc

    # Shuffle each group with seeded RNG and take allocations
    rng = random.Random(seed)
    result = []

    for source in sorted(groups.keys()):
        group = groups[source]
        shuffled = group.copy()
        rng.shuffle(shuffled)
        result.extend(shuffled[: allocations[source]])

    # Trim to at most n
    result = result[:n]

    # Sort deterministically by (source, id)
    result.sort(key=lambda t: (t.source, t.id))

    return result

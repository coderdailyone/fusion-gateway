import random
from collections import defaultdict
from evaluator.suite.types import Task
from evaluator.report import ResultRow

QUALITY_HEADROOM_MIN = 0.05
COST_SAVINGS_MIN = 0.15


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


def analyze(rows: list[ResultRow]) -> dict:
    """
    Compute routing-signal metrics and a go/no-go verdict from pilot results.

    Groups rows by model (for per-model accuracy/cost) and by task_id (for
    oracle/disagreement/iso-quality-cost, since each task has one row per
    model). See module-level QUALITY_HEADROOM_MIN / COST_SAVINGS_MIN for the
    verdict thresholds.

    Args:
        rows: List of ResultRow objects, one per (task, model) pair.

    Returns:
        Dict with keys: per_model, best_static, oracle_accuracy,
        quality_headroom, disagreement_rate, iso_quality_cost, cost_savings,
        verdict, signals.
    """
    if not rows:
        return {
            "per_model": {},
            "best_static": {"model": None, "accuracy": 0.0, "total_cost_usd": 0.0},
            "oracle_accuracy": 0.0,
            "quality_headroom": 0.0,
            "disagreement_rate": 0.0,
            "iso_quality_cost": 0.0,
            "cost_savings": 0.0,
            "verdict": "NO_GO",
            "signals": [],
        }

    by_model: dict[str, list[ResultRow]] = defaultdict(list)
    by_task: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        by_model[row.model].append(row)
        by_task[row.task_id].append(row)

    # Per-model n / accuracy / total cost.
    per_model: dict[str, dict] = {}
    for model, model_rows in by_model.items():
        n = len(model_rows)
        correct_count = sum(1 for r in model_rows if r.correct)
        total_cost = sum(r.cost_usd for r in model_rows)
        per_model[model] = {
            "n": n,
            "accuracy": correct_count / n,
            "total_cost_usd": total_cost,
        }

    # Best static model: highest accuracy, tie-break lowest total cost,
    # then model name for full determinism.
    best_model = min(
        per_model.items(),
        key=lambda item: (-item[1]["accuracy"], item[1]["total_cost_usd"], item[0]),
    )[0]
    best_static = {
        "model": best_model,
        "accuracy": per_model[best_model]["accuracy"],
        "total_cost_usd": per_model[best_model]["total_cost_usd"],
    }

    # Per-task signals: oracle correctness, disagreement, iso-quality cost.
    n_tasks = len(by_task)
    oracle_correct = 0
    disagreement_count = 0
    iso_quality_cost = 0.0

    for task_rows in by_task.values():
        corrects = [r.correct for r in task_rows]
        any_correct = any(corrects)
        if any_correct:
            oracle_correct += 1
        if any_correct and not all(corrects):
            disagreement_count += 1

        correct_costs = [r.cost_usd for r in task_rows if r.correct]
        if correct_costs:
            iso_quality_cost += min(correct_costs)
        else:
            iso_quality_cost += min(r.cost_usd for r in task_rows)

    oracle_accuracy = oracle_correct / n_tasks
    disagreement_rate = disagreement_count / n_tasks
    quality_headroom = oracle_accuracy - best_static["accuracy"]

    if best_static["total_cost_usd"] == 0:
        cost_savings = 0.0
    else:
        cost_savings = 1 - iso_quality_cost / best_static["total_cost_usd"]

    signals = []
    if quality_headroom >= QUALITY_HEADROOM_MIN:
        signals.append("quality")
    if cost_savings >= COST_SAVINGS_MIN:
        signals.append("cost")

    verdict = "GO" if signals else "NO_GO"

    return {
        "per_model": per_model,
        "best_static": best_static,
        "oracle_accuracy": oracle_accuracy,
        "quality_headroom": quality_headroom,
        "disagreement_rate": disagreement_rate,
        "iso_quality_cost": iso_quality_cost,
        "cost_savings": cost_savings,
        "verdict": verdict,
        "signals": signals,
    }

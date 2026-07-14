"""Sample-level report aggregation for evaluator results."""

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ResultRow:
    """A single evaluation result row."""
    task_id: str
    source: str
    model: str
    correct: bool
    cost_usd: float


def aggregate(rows: list[ResultRow]) -> dict:
    """Aggregate results by model.

    Args:
        rows: List of ResultRow objects.

    Returns:
        Dictionary mapping model name to aggregated stats:
        {model: {"n": int, "accuracy": float, "mean_cost_usd": float, "total_cost_usd": float}}
    """
    stats = defaultdict(lambda: {
        "n": 0,
        "correct_count": 0,
        "total_cost_usd": 0.0,
    })

    for row in rows:
        stats[row.model]["n"] += 1
        if row.correct:
            stats[row.model]["correct_count"] += 1
        stats[row.model]["total_cost_usd"] += row.cost_usd

    result = {}
    for model, model_stats in stats.items():
        n = model_stats["n"]
        if n > 0:
            correct_count = model_stats["correct_count"]
            total_cost = model_stats["total_cost_usd"]
            result[model] = {
                "n": n,
                "accuracy": correct_count / n,
                "mean_cost_usd": total_cost / n,
                "total_cost_usd": total_cost,
            }

    return result

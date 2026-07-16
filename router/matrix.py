"""Cost-aware routing result matrix."""

from dataclasses import dataclass, field
from collections import defaultdict

from evaluator.report import ResultRow


@dataclass
class ResultMatrix:
    """Matrix of (success, cost) results indexed by model and task."""
    tasks: list[str]
    models: list[str]
    correct: dict[str, dict[str, bool]]
    cost: dict[str, dict[str, float]]
    source: dict[str, str]

    @classmethod
    def from_rows(cls, rows: list[ResultRow]) -> "ResultMatrix":
        """Build a ResultMatrix from a list of result rows.

        Args:
            rows: List of ResultRow objects, each containing task_id, source, model,
                  correct (bool), and cost_usd (float).

        Returns:
            ResultMatrix with sorted tasks, sorted models, and lookup dicts.
        """
        # Extract unique tasks and models, then sort them
        tasks = sorted(set(row.task_id for row in rows))
        models = sorted(set(row.model for row in rows))

        # Build lookup dicts
        correct_dict = defaultdict(dict)
        cost_dict = defaultdict(dict)
        source_dict = {}

        for row in rows:
            correct_dict[row.model][row.task_id] = row.correct
            cost_dict[row.model][row.task_id] = row.cost_usd
            source_dict[row.task_id] = row.source

        return cls(
            tasks=tasks,
            models=models,
            correct=dict(correct_dict),
            cost=dict(cost_dict),
            source=source_dict,
        )

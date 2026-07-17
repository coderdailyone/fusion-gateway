"""Deterministic code verify-cascade: walk models cheapest→dearest, stop at first pass."""

from router.matrix import ResultMatrix


def cascade_task(task_id: str, order: list[str], matrix: ResultMatrix) -> tuple[bool, float]:
    """Walk models in order (cheapest→dearest); pay each tried; stop at first pass.

    Args:
        task_id: The task identifier to evaluate.
        order: List of model names in order (cheapest to dearest).
        matrix: ResultMatrix with correct[model][task_id] and cost[model][task_id].

    Returns:
        Tuple of (correct: bool, cost: float).
        - correct: True if any model in order passed tests for this task.
        - cost: Sum of costs of all models tried up to and including the first pass,
                or sum of all models if none pass.
    """
    total_cost = 0.0

    for model in order:
        model_cost = matrix.cost[model][task_id]
        total_cost += model_cost

        if matrix.correct[model][task_id]:
            return (True, total_cost)

    # No model passed
    return (False, total_cost)


def evaluate(code_task_ids: list[str], order: list[str], matrix: ResultMatrix) -> tuple[float, float]:
    """Evaluate cascade strategy across multiple tasks.

    Args:
        code_task_ids: List of task identifiers to evaluate.
        order: List of model names in order (cheapest to dearest).
        matrix: ResultMatrix with results for all tasks and models.

    Returns:
        Tuple of (accuracy: float, mean_cost: float).
        - accuracy: Fraction of tasks where cascade found a passing model.
        - mean_cost: Mean total cost per task.
    """
    if not code_task_ids:
        return (0.0, 0.0)

    total_correct = 0
    total_cost = 0.0

    for task_id in code_task_ids:
        correct, cost = cascade_task(task_id, order, matrix)
        if correct:
            total_correct += 1
        total_cost += cost

    accuracy = total_correct / len(code_task_ids)
    mean_cost = total_cost / len(code_task_ids)

    return (accuracy, mean_cost)

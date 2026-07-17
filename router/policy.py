"""Lambda-swept cost-aware routing policy.

`route_task` turns one task's per-model (P(correct), cost) pair into a
routing decision: pick the model maximizing `proba - lam * cost`. At
`lam=0` this collapses to "pick the most likely to succeed"; as `lam` grows
the cost term dominates and the policy converges to "pick the cheapest".

`evaluate` scores a fixed `lam` against a set of tasks using out-of-fold
success probabilities (`OOFPredictions`, so the routing decision cannot
peek at the label it's trying to predict) and the REALIZED outcome/cost
from the `ResultMatrix` for whichever model got chosen.

`sweep_lambda` runs `evaluate` across a list of `lam` values, tracing out
a cost-quality curve: each point trades an accuracy loss for a cost saving
as `lam` increases from 0 toward "route by cost alone".
"""

from __future__ import annotations

from router.matrix import ResultMatrix
from router.train import OOFPredictions


def route_task(
    proba_at_task: dict[str, float], cost_at_task: dict[str, float], lam: float
) -> str:
    """Pick the model maximizing `proba - lam*cost` among models with both a
    proba and a cost entry for this task.

    Ties are broken by lowest cost, then by model name, so the choice is
    deterministic regardless of dict iteration order.
    """
    candidates = sorted(set(proba_at_task) & set(cost_at_task))
    if not candidates:
        raise ValueError("no model has both a proba and a cost for this task")

    def key(model: str) -> tuple[float, float, str]:
        score = proba_at_task[model] - lam * cost_at_task[model]
        return (-score, cost_at_task[model], model)

    return min(candidates, key=key)


def evaluate(
    oof: OOFPredictions,
    matrix: ResultMatrix,
    lam: float,
    task_ids: list[str],
) -> tuple[float, float]:
    """Route every task in `task_ids` at a fixed `lam` and score the
    REALIZED outcome/cost of the chosen model.

    Returns (accuracy, mean_cost) = (fraction of tasks where the chosen
    model's `matrix.correct` is True, mean of the chosen model's
    `matrix.cost`) over task_ids.
    """
    n_correct = 0
    total_cost = 0.0

    for task_id in task_ids:
        proba_at_task = {
            model: probas[task_id]
            for model, probas in oof.proba.items()
            if task_id in probas
        }
        cost_at_task = {
            model: costs[task_id]
            for model, costs in matrix.cost.items()
            if task_id in costs
        }
        chosen = route_task(proba_at_task, cost_at_task, lam)

        if matrix.correct[chosen][task_id]:
            n_correct += 1
        total_cost += matrix.cost[chosen][task_id]

    n = len(task_ids)
    accuracy = n_correct / n if n else float("nan")
    mean_cost = total_cost / n if n else float("nan")
    return accuracy, mean_cost


def sweep_lambda(
    oof: OOFPredictions,
    matrix: ResultMatrix,
    task_ids: list[str],
    lambdas: list[float],
) -> list[dict]:
    """Trace the cost-quality curve: one `evaluate` point per lambda."""
    points = []
    for lam in lambdas:
        accuracy, mean_cost = evaluate(oof, matrix, lam, task_ids)
        points.append({"lambda": lam, "accuracy": accuracy, "mean_cost": mean_cost})
    return points

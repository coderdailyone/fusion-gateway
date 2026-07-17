"""Pareto envelope check + report rendering.

Acceptance logic for M3b: does the dynamic (lambda-swept) routing curve
"envelope" the static single-model baselines? A dynamic point dominates a
static point when it is at least as accurate AND no more costly -- i.e. the
static baseline offers no reason to prefer it over some point on the
dynamic curve.

`static_points` scores each single model as if it alone handled every task
in `task_ids` (the same shape of (accuracy, mean_cost) that `router.policy`
produces for the dynamic curve, so the two are directly comparable).

`envelopes` checks domination for each static point against the whole
dynamic curve. `render_report` turns both into a markdown block for
inclusion in the M3b writeup.
"""

from __future__ import annotations

from router.matrix import ResultMatrix


def static_points(
    matrix: ResultMatrix, task_ids: list[str]
) -> dict[str, tuple[float, float]]:
    """Score each model as a single-model baseline over `task_ids`.

    Returns {model: (accuracy, mean_cost)} where accuracy is the fraction
    of task_ids the model gets correct and mean_cost is its mean cost,
    both restricted to `task_ids` (tasks outside that set are ignored).
    """
    n = len(task_ids)
    points: dict[str, tuple[float, float]] = {}
    for model in matrix.models:
        correct = matrix.correct.get(model, {})
        cost = matrix.cost.get(model, {})
        n_correct = sum(1 for t in task_ids if correct.get(t))
        total_cost = sum(cost[t] for t in task_ids if t in cost)
        accuracy = n_correct / n if n else float("nan")
        mean_cost = total_cost / n if n else float("nan")
        points[model] = (accuracy, mean_cost)
    return points


def _dominated_by_any(
    static_point: tuple[float, float], dynamic_points: list[dict]
) -> bool:
    a_s, c_s = static_point
    return any(
        p["accuracy"] >= a_s and p["mean_cost"] <= c_s for p in dynamic_points
    )


def envelopes(
    dynamic_points: list[dict], static_points: dict[str, tuple[float, float]]
) -> dict:
    """Check whether the dynamic curve dominates every static baseline.

    A dynamic point {"accuracy": a, "mean_cost": c, ...} dominates a static
    point (a_s, c_s) iff a >= a_s AND c <= c_s.

    Returns {"envelops_all": bool, "dominated_static": [...], "undominated_static": [...]}.
    """
    dominated = []
    undominated = []
    for model, point in static_points.items():
        if _dominated_by_any(point, dynamic_points):
            dominated.append(model)
        else:
            undominated.append(model)
    return {
        "envelops_all": len(undominated) == 0,
        "dominated_static": dominated,
        "undominated_static": undominated,
    }


def render_report(
    dynamic_points: list[dict],
    static_points: dict[str, tuple[float, float]],
    envelope_result: dict,
    extra: dict | None = None,
) -> str:
    """Render a markdown block: static baselines table, dynamic curve
    table, and the envelope verdict line."""
    lines = ["## Pareto envelope check", ""]

    lines.append("### Static single-model baselines")
    lines.append("")
    lines.append("| model | accuracy | mean_cost |")
    lines.append("| --- | --- | --- |")
    for model in sorted(static_points):
        accuracy, mean_cost = static_points[model]
        lines.append(f"| {model} | {accuracy:.4f} | {mean_cost:.6f} |")
    lines.append("")

    lines.append("### Dynamic (lambda-swept) curve")
    lines.append("")
    lines.append("| lambda | accuracy | mean_cost |")
    lines.append("| --- | --- | --- |")
    for point in dynamic_points:
        lines.append(
            f"| {point.get('lambda')} | {point['accuracy']:.4f} | "
            f"{point['mean_cost']:.6f} |"
        )
    lines.append("")

    envelops_all = envelope_result["envelops_all"]
    verdict = "PASS" if envelops_all else "FAIL"
    lines.append(
        f"**Verdict: {verdict}** -- envelops_all={envelops_all}. "
        f"dominated_static={envelope_result['dominated_static']}, "
        f"undominated_static={envelope_result['undominated_static']}."
    )

    if extra:
        lines.append("")
        for key, value in extra.items():
            lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"

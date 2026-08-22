"""Small reporting metrics shared by offline scripts."""
from __future__ import annotations

import math


def linear_percentile(sorted_values: list[float], percentile: float) -> float:
    """Return a linear-interpolation percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return (
        sorted_values[low] * (high - position)
        + sorted_values[high] * (position - low)
    )

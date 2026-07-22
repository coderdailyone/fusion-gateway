"""Compute M4 agentic-tier metrics and render docs/M4_AGENTIC_TIER_REPORT.md.

Reuses scripts/hard_report.wilson_ci. Pure functions here are unit-tested; the
main() render reads frozen attempts + the harness resolved map.
"""
from __future__ import annotations

from scripts.hard_report import wilson_ci


def metrics(resolved: dict, costs: dict) -> dict:
    n = len(resolved)
    k = sum(1 for v in resolved.values() if v)
    total = sum(costs.get(i, 0.0) for i in resolved)
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    return {
        "n": n, "resolved": k,
        "resolve_rate": (k / n) if n else 0.0,
        "wilson_lo": lo, "wilson_hi": hi,
        "total_cost": total,
        "cost_per_successful": (total / k) if k else float("inf"),
    }


def verifier_agreement(verifier_pass: dict, resolved: dict) -> dict:
    ids = [i for i in verifier_pass if i in resolved]
    passed = [i for i in ids if verifier_pass[i]]
    truly = [i for i in ids if resolved[i]]
    tp = sum(1 for i in passed if resolved[i])
    precision = (tp / len(passed)) if passed else float("nan")
    recall = (tp / len(truly)) if truly else float("nan")
    return {"precision": precision, "recall": recall, "n": len(ids)}

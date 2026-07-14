#!/usr/bin/env python3
"""Daily rollup over a gateway SQLite DB.

Usage:
    python scripts/rollup.py <db_path>

Opens the DB read-only (never writes) and prints, per day (grouped by
date(created_at)):
    - requests grouped by status
    - total cost (SUM over ledger settled/preflight/orphaned states of
      COALESCE(actual_cost_usd, est_cost_usd))
    - P50/P95 latency_ms over settled ledger rows
    - fallback count (events with kind='call.failed')
Plus the current state of every row in the budgets table (not day-bucketed;
budgets carry no history, only a current cap/state).
"""
from __future__ import annotations

import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Ledger states whose (actual or estimated) cost counts as "consumed" budget.
# Mirrors gateway.ledger.CONSUMING_STATES.
COST_STATES = ("settled", "preflight", "orphaned")


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same convention as numpy.percentile)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def requests_by_day_status(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """One query: per-day count of requests grouped by status."""
    rows = conn.execute(
        "SELECT date(created_at) AS day, status, COUNT(*) AS c "
        "FROM requests WHERE id != 'admin' GROUP BY day, status"
    ).fetchall()
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        out[row["day"]][row["status"]] = row["c"]
    return dict(out)


def cost_by_day(conn: sqlite3.Connection) -> dict[str, float]:
    """One query: per-day total cost across settled/preflight/orphaned ledger rows."""
    rows = conn.execute(
        "SELECT date(created_at) AS day, "
        "SUM(COALESCE(actual_cost_usd, est_cost_usd)) AS cost "
        "FROM ledger WHERE state IN (?, ?, ?) GROUP BY day",
        COST_STATES,
    ).fetchall()
    return {row["day"]: (row["cost"] or 0.0) for row in rows}


def latency_by_day(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """One query: raw settled latency_ms per day; P50/P95 computed in Python
    (sqlite3 in this environment has no built-in percentile function)."""
    rows = conn.execute(
        "SELECT date(created_at) AS day, latency_ms FROM ledger "
        "WHERE state = 'settled' AND latency_ms IS NOT NULL"
    ).fetchall()
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["day"]].append(row["latency_ms"])
    out: dict[str, dict[str, float]] = {}
    for day, values in grouped.items():
        values.sort()
        out[day] = {"p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95)}
    return out


def fallback_count_by_day(conn: sqlite3.Connection) -> dict[str, int]:
    """One query: per-day count of events with kind='call.failed'."""
    rows = conn.execute(
        "SELECT date(created_at) AS day, COUNT(*) AS c FROM events "
        "WHERE kind = 'call.failed' GROUP BY day"
    ).fetchall()
    return {row["day"]: row["c"] for row in rows}


def budget_state(conn: sqlite3.Connection) -> list[dict]:
    """One query: current state of every budget (no per-day history)."""
    rows = conn.execute(
        "SELECT name, cap_usd, state, updated_at FROM budgets ORDER BY name"
    ).fetchall()
    return [dict(row) for row in rows]


def rollup(db_path: str | Path) -> dict:
    """Compute the full daily rollup for db_path. Read-only, no side effects."""
    conn = _connect_readonly(db_path)
    try:
        by_status = requests_by_day_status(conn)
        by_cost = cost_by_day(conn)
        by_latency = latency_by_day(conn)
        by_fallback = fallback_count_by_day(conn)
        budgets = budget_state(conn)
    finally:
        conn.close()

    days = sorted(set(by_status) | set(by_cost) | set(by_latency) | set(by_fallback))
    result_days = {}
    for day in days:
        latency = by_latency.get(day, {})
        result_days[day] = {
            "requests_by_status": by_status.get(day, {}),
            "cost_usd": by_cost.get(day, 0.0),
            "latency_p50_ms": latency.get("p50", 0.0),
            "latency_p95_ms": latency.get("p95", 0.0),
            "fallback_count": by_fallback.get(day, 0),
        }

    return {"days": result_days, "budgets": budgets}


def format_report(data: dict) -> str:
    lines: list[str] = []
    for day in sorted(data["days"]):
        d = data["days"][day]
        status_str = ", ".join(
            f"{k}={v}" for k, v in sorted(d["requests_by_status"].items())
        ) or "(none)"
        lines.append(f"== {day} ==")
        lines.append(f"  requests: {status_str}")
        lines.append(f"  cost_usd: {d['cost_usd']:.6f}")
        lines.append(
            f"  latency_ms: p50={d['latency_p50_ms']:.1f} p95={d['latency_p95_ms']:.1f}"
        )
        lines.append(f"  fallback_count: {d['fallback_count']}")
    if not data["days"]:
        lines.append("(no requests)")
    lines.append("== budgets ==")
    if not data["budgets"]:
        lines.append("  (none)")
    for b in data["budgets"]:
        lines.append(
            f"  {b['name']}: state={b['state']} cap_usd={b['cap_usd']} "
            f"updated_at={b['updated_at']}"
        )
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: rollup.py <db_path>", file=sys.stderr)
        return 1
    data = rollup(sys.argv[1])
    print(format_report(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())

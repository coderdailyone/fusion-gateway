#!/usr/bin/env python3
"""Render a deterministic Markdown summary of a frozen benchmark run.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/benchmark_summary.py <RUN_DIR>

The command only reads <RUN_DIR>/frozen.jsonl. It does not load a benchmark
suite, download datasets, score answers, or call a model.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.metrics import linear_percentile
except ModuleNotFoundError:  # direct ``python scripts/benchmark_summary.py``
    from metrics import linear_percentile

REQUIRED_FIELDS = {
    "task_id",
    "source",
    "model",
    "prompt",
    "output_text",
    "in_tokens",
    "out_tokens",
    "latency_ms",
    "status",
    "error",
}
NON_EMPTY_STRING_FIELDS = ("task_id", "source", "model", "status")
ERROR_SUMMARY_LIMIT = 5
ERROR_SUMMARY_LENGTH = 160


class SummaryError(ValueError):
    """An invalid run directory or frozen-output file."""


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_row(row: object, line_number: int) -> dict:
    if not isinstance(row, dict):
        raise SummaryError(f"line {line_number}: expected a JSON object")

    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise SummaryError(
            f"line {line_number}: missing required field(s): {', '.join(missing)}"
        )

    for field in NON_EMPTY_STRING_FIELDS:
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            raise SummaryError(
                f"line {line_number}: {field} must be a non-empty string"
            )

    if not isinstance(row["output_text"], str):
        raise SummaryError(f"line {line_number}: output_text must be a string")

    for field in ("in_tokens", "out_tokens"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SummaryError(
                f"line {line_number}: {field} must be a non-negative integer"
            )

    latency = row["latency_ms"]
    if not _is_number(latency) or latency < 0:
        raise SummaryError(
            f"line {line_number}: latency_ms must be a non-negative number"
        )

    error = row["error"]
    if error is not None and not isinstance(error, str):
        raise SummaryError(f"line {line_number}: error must be a string or null")

    if "cost_usd" in row:
        cost = row["cost_usd"]
        if not _is_number(cost) or cost < 0:
            raise SummaryError(
                f"line {line_number}: cost_usd must be a non-negative number"
            )

    return row


def load_rows(run_dir: str | Path) -> list[dict]:
    """Load and validate every non-blank record in RUN_DIR/frozen.jsonl."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise SummaryError(f"run directory does not exist: {run_path}")
    if not run_path.is_dir():
        raise SummaryError(f"run path is not a directory: {run_path}")

    frozen_path = run_path / "frozen.jsonl"
    if not frozen_path.exists():
        raise SummaryError(f"missing frozen.jsonl: {frozen_path}")
    if not frozen_path.is_file():
        raise SummaryError(f"frozen.jsonl is not a file: {frozen_path}")

    rows: list[dict] = []
    try:
        with frozen_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SummaryError(
                        f"invalid JSON in {frozen_path} at line {line_number}: {exc.msg}"
                    ) from None
                rows.append(_validate_row(row, line_number))
    except UnicodeDecodeError as exc:
        raise SummaryError(
            f"frozen.jsonl is not valid UTF-8: {frozen_path} (byte {exc.start})"
        ) from None
    except OSError as exc:
        raise SummaryError(f"could not read {frozen_path}: {exc}") from None

    if not rows:
        raise SummaryError(f"frozen.jsonl contains no records: {frozen_path}")
    return rows


def _token_totals(rows: list[dict], group_field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "input": 0, "output": 0}
    )
    for row in rows:
        totals = grouped[row[group_field]]
        totals["rows"] += 1
        totals["input"] += row["in_tokens"]
        totals["output"] += row["out_tokens"]
    return {key: grouped[key] for key in sorted(grouped)}


def _error_summary(value: str | None) -> str:
    text = " ".join((value or "(no error message)").split())
    if len(text) > ERROR_SUMMARY_LENGTH:
        return text[: ERROR_SUMMARY_LENGTH - 3] + "..."
    return text


def summarize(run_dir: str | Path, rows: list[dict]) -> dict:
    """Aggregate validated frozen rows without deduplicating them."""
    pair_counts = Counter((row["task_id"], row["model"]) for row in rows)
    duplicates = [
        {
            "task_id": task_id,
            "model": model,
            "occurrences": count,
            "duplicate_rows": count - 1,
        }
        for (task_id, model), count in sorted(pair_counts.items())
        if count > 1
    ]

    status_counts = Counter(row["status"] for row in rows)
    error_rows = sorted(
        (row for row in rows if row["status"] == "error"),
        key=lambda row: (row["task_id"], row["model"], row["error"] or ""),
    )
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    costs = [float(row["cost_usd"]) for row in rows if "cost_usd" in row]
    total_cost = math.fsum(costs) if costs else None
    tokens_by_model = _token_totals(rows, "model")
    tokens_by_source = _token_totals(rows, "source")

    return {
        "run_dir": str(Path(run_dir)),
        "raw_rows": len(rows),
        "unique_pairs": len(pair_counts),
        "duplicate_rows": len(rows) - len(pair_counts),
        "duplicates": duplicates,
        "statuses": dict(sorted(status_counts.items())),
        "ok_empty_outputs": sum(
            row["status"] == "ok" and not row["output_text"].strip()
            for row in rows
        ),
        "error_samples": [
            {
                "task_id": row["task_id"],
                "model": row["model"],
                "error": _error_summary(row["error"]),
            }
            for row in error_rows[:ERROR_SUMMARY_LIMIT]
        ],
        "tokens": {
            "input": sum(row["in_tokens"] for row in rows),
            "output": sum(row["out_tokens"] for row in rows),
            "by_model": tokens_by_model,
            "by_source": tokens_by_source,
        },
        "latency_ms": {
            "p50": linear_percentile(latencies, 0.50),
            "p95": linear_percentile(latencies, 0.95),
            "max": latencies[-1],
        },
        "cost": {
            "records": len(costs),
            "total": total_cost,
            "mean": (total_cost / len(costs)) if costs else None,
        },
    }


def _markdown_cell(value: object) -> str:
    text = " ".join(str(value).split())
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
    )


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _render_token_table(grouped: dict[str, dict[str, int]], label: str) -> list[str]:
    rows = [
        (key, values["rows"], values["input"], values["output"])
        for key, values in grouped.items()
    ]
    return _table((label, "Rows", "Input tokens", "Output tokens"), rows)


def render_markdown(summary: dict) -> str:
    """Render a stable Markdown report from summarize()."""
    lines = ["# Benchmark Run Summary", "", "## Basic information", ""]
    lines.extend(
        _table(
            ("Metric", "Value"),
            [
                ("Run directory", summary["run_dir"]),
                ("Raw rows", summary["raw_rows"]),
                ("Unique `(task_id, model)` pairs", summary["unique_pairs"]),
                ("Duplicate records", summary["duplicate_rows"]),
            ],
        )
    )

    lines.extend(["", "### Models", ""])
    lines.extend(
        _table(
            ("Model", "Rows"),
            [
                (model, totals["rows"])
                for model, totals in summary["tokens"]["by_model"].items()
            ],
        )
    )
    lines.extend(["", "### Sources", ""])
    lines.extend(
        _table(
            ("Source", "Rows"),
            [
                (source, totals["rows"])
                for source, totals in summary["tokens"]["by_source"].items()
            ],
        )
    )

    lines.extend(["", "## Run health", "", "### Status distribution", ""])
    lines.extend(
        _table(
            ("Status", "Rows"),
            [(status, count) for status, count in summary["statuses"].items()],
        )
    )
    error_count = summary["statuses"].get("error", 0)
    lines.extend(
        [
            "",
            f"- `status=ok` with empty output: **{summary['ok_empty_outputs']}**",
            f"- `status=error` records: **{error_count}**",
        ]
    )

    lines.extend(["", "## Resource usage", ""])
    resource_rows: list[tuple[object, ...]] = [
        ("Total input tokens", summary["tokens"]["input"]),
        ("Total output tokens", summary["tokens"]["output"]),
        ("Latency p50 (ms)", f"{summary['latency_ms']['p50']:.2f}"),
        ("Latency p95 (ms)", f"{summary['latency_ms']['p95']:.2f}"),
        ("Latency max (ms)", f"{summary['latency_ms']['max']:.2f}"),
    ]
    cost = summary["cost"]
    if cost["records"]:
        resource_rows.extend(
            [
                ("Records with `cost_usd`", f"{cost['records']} / {summary['raw_rows']}"),
                ("Total cost (USD)", f"${cost['total']:.8f}"),
                ("Mean cost (USD)", f"${cost['mean']:.8f}"),
            ]
        )
    else:
        resource_rows.append(("Cost", "Not available (`cost_usd` absent)"))
    lines.extend(_table(("Metric", "Value"), resource_rows))
    if 0 < cost["records"] < summary["raw_rows"]:
        lines.extend(
            [
                "",
                "> **Warning:** `cost_usd` is present on only "
                f"{cost['records']} of {summary['raw_rows']} records; cost totals "
                "cover available values only.",
            ]
        )

    lines.extend(["", "### Tokens by model", ""])
    lines.extend(_render_token_table(summary["tokens"]["by_model"], "Model"))
    lines.extend(["", "### Tokens by source", ""])
    lines.extend(_render_token_table(summary["tokens"]["by_source"], "Source"))

    lines.extend(["", "## Duplicate records", ""])
    if summary["duplicates"]:
        lines.append(
            "> **Warning:** duplicate pairs are included in all row-based totals; "
            "they were not silently discarded."
        )
        lines.append("")
        lines.extend(
            _table(
                ("Task ID", "Model", "Occurrences", "Duplicate rows"),
                [
                    (
                        item["task_id"],
                        item["model"],
                        item["occurrences"],
                        item["duplicate_rows"],
                    )
                    for item in summary["duplicates"]
                ],
            )
        )
    else:
        lines.append("No duplicate `(task_id, model)` pairs found.")

    lines.extend(["", "## Error samples", ""])
    if summary["error_samples"]:
        lines.append(
            f"Showing {len(summary['error_samples'])} of "
            f"{summary['statuses'].get('error', 0)} `status=error` records "
            f"(maximum {ERROR_SUMMARY_LIMIT})."
        )
        lines.append("")
        lines.extend(
            _table(
                ("Task ID", "Model", "Error summary"),
                [
                    (item["task_id"], item["model"], item["error"])
                    for item in summary["error_samples"]
                ],
            )
        )
    else:
        lines.append("No `status=error` records found.")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: benchmark_summary.py <RUN_DIR>", file=sys.stderr)
        return 2

    try:
        rows = load_rows(args[0])
        report = render_markdown(summarize(args[0], rows))
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

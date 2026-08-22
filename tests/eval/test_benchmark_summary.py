import json
import re

import pytest

from scripts.benchmark_summary import (
    SummaryError,
    load_rows,
    main,
    render_markdown,
    summarize,
)


def _row(task_id="t1", model="model-b", source="math", **overrides):
    row = {
        "task_id": task_id,
        "source": source,
        "model": model,
        "prompt": "A synthetic prompt",
        "output_text": "A synthetic answer",
        "in_tokens": 10,
        "out_tokens": 5,
        "cost_usd": 0.001,
        "latency_ms": 100,
        "status": "ok",
        "error": None,
    }
    row.update(overrides)
    return row


def _write_rows(run_dir, rows):
    run_dir.mkdir()
    frozen = run_dir / "frozen.jsonl"
    frozen.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return frozen


def test_normal_input_reports_stable_counts_tokens_latency_and_cost(tmp_path):
    run_dir = tmp_path / "run"
    rows = [
        _row("t2", model="model-b", source="math", in_tokens=20,
             out_tokens=7, cost_usd=0.003, latency_ms=300),
        _row("t1", model="model-a", source="humaneval", in_tokens=10,
             out_tokens=3, cost_usd=0.001, latency_ms=100),
        _row("t3", model="model-a", source="math", in_tokens=30,
             out_tokens=9, cost_usd=0.002, latency_ms=200),
    ]
    _write_rows(run_dir, rows)

    loaded = load_rows(run_dir)
    summary = summarize(run_dir, loaded)
    report = render_markdown(summary)

    assert summary["raw_rows"] == 3
    assert summary["unique_pairs"] == 3
    assert summary["duplicate_rows"] == 0
    assert {
        model: totals["rows"]
        for model, totals in summary["tokens"]["by_model"].items()
    } == {"model-a": 2, "model-b": 1}
    assert {
        source: totals["rows"]
        for source, totals in summary["tokens"]["by_source"].items()
    } == {"humaneval": 1, "math": 2}
    assert summary["tokens"]["input"] == 60
    assert summary["tokens"]["output"] == 19
    assert summary["tokens"]["by_model"]["model-a"] == {
        "rows": 2, "input": 40, "output": 12,
    }
    assert summary["tokens"]["by_source"]["math"] == {
        "rows": 2, "input": 50, "output": 16,
    }
    assert summary["latency_ms"] == {"p50": 200.0, "p95": 290.0, "max": 300.0}
    assert summary["cost"]["total"] == pytest.approx(0.006)
    assert summary["cost"]["mean"] == pytest.approx(0.002)
    assert report.index("| model-a | 2 |") < report.index("| model-b | 1 |")
    assert "No duplicate `(task_id, model)` pairs found." in report
    assert render_markdown(summary) == report


def test_error_rows_are_counted_sorted_truncated_and_markdown_escaped(tmp_path):
    run_dir = tmp_path / "run"
    rows = []
    for number in range(7, 0, -1):
        message = (
            f"upstream | failure <{number}>\n" + "x" * 200
            if number == 1
            else f"upstream failure {number}"
        )
        rows.append(
            _row(
                f"t{number}",
                status="error",
                output_text="",
                in_tokens=0,
                out_tokens=0,
                cost_usd=0.0,
                error=message,
            )
        )
    _write_rows(run_dir, rows)

    summary = summarize(run_dir, load_rows(run_dir))
    report = render_markdown(summary)

    assert summary["statuses"] == {"error": 7}
    assert [item["task_id"] for item in summary["error_samples"]] == [
        "t1", "t2", "t3", "t4", "t5",
    ]
    assert len(summary["error_samples"][0]["error"]) == 160
    assert "upstream \\| failure &lt;1&gt;" in report
    assert "Showing 5 of 7 `status=error` records (maximum 5)." in report
    assert "t6" not in report and "t7" not in report


def test_duplicate_pairs_are_reported_and_remain_in_resource_totals(tmp_path):
    run_dir = tmp_path / "run"
    _write_rows(
        run_dir,
        [
            _row("same", in_tokens=10, out_tokens=1),
            _row("same", in_tokens=20, out_tokens=2),
            _row("same", in_tokens=30, out_tokens=3),
            _row("other", in_tokens=40, out_tokens=4),
        ],
    )

    summary = summarize(run_dir, load_rows(run_dir))
    report = render_markdown(summary)

    assert summary["raw_rows"] == 4
    assert summary["unique_pairs"] == 2
    assert summary["duplicate_rows"] == 2
    assert summary["tokens"]["input"] == 100
    assert summary["duplicates"] == [
        {
            "task_id": "same",
            "model": "model-b",
            "occurrences": 3,
            "duplicate_rows": 2,
        }
    ]
    assert "duplicate pairs are included in all row-based totals" in report
    assert "| same | model-b | 3 | 2 |" in report


def test_missing_frozen_file_has_clear_error(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(SummaryError, match=r"missing frozen\.jsonl: .*frozen\.jsonl"):
        load_rows(run_dir)


def test_invalid_json_reports_file_and_line_number(tmp_path):
    run_dir = tmp_path / "run"
    frozen = _write_rows(run_dir, [_row()])
    with frozen.open("a", encoding="utf-8") as stream:
        stream.write('{"task_id":\n')

    with pytest.raises(SummaryError) as exc_info:
        load_rows(run_dir)

    message = str(exc_info.value)
    assert str(frozen) in message
    assert "line 2" in message
    assert "invalid JSON" in message


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing_dir", "run directory does not exist"),
        ("not_dir", "run path is not a directory"),
        ("empty", "frozen.jsonl contains no records"),
        ("blank", "frozen.jsonl contains no records"),
        ("not_object", "line 1: expected a JSON object"),
        ("missing_field", "line 1: missing required field(s): latency_ms"),
        ("missing_prompt", "line 1: missing required field(s): prompt"),
    ],
)
def test_invalid_run_shapes_have_clear_errors(tmp_path, setup, message):
    run_path = tmp_path / setup
    if setup == "not_dir":
        run_path.write_text("not a directory", encoding="utf-8")
    elif setup in {
        "empty", "blank", "not_object", "missing_field", "missing_prompt",
    }:
        run_path.mkdir()
        frozen = run_path / "frozen.jsonl"
        if setup == "empty":
            frozen.write_text("", encoding="utf-8")
        elif setup == "blank":
            frozen.write_text("\n   \n", encoding="utf-8")
        elif setup == "not_object":
            frozen.write_text("[]\n", encoding="utf-8")
        else:
            row = _row()
            del row["prompt" if setup == "missing_prompt" else "latency_ms"]
            frozen.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SummaryError, match=re.escape(message)):
        load_rows(run_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", "", "task_id must be a non-empty string"),
        ("output_text", None, "output_text must be a string"),
        ("in_tokens", -1, "in_tokens must be a non-negative integer"),
        ("out_tokens", 1.5, "out_tokens must be a non-negative integer"),
        ("latency_ms", -1, "latency_ms must be a non-negative number"),
        ("error", {"message": "bad"}, "error must be a string or null"),
        ("cost_usd", float("inf"), "cost_usd must be a non-negative number"),
    ],
)
def test_invalid_field_values_are_rejected(tmp_path, field, value, message):
    run_dir = tmp_path / "run"
    _write_rows(run_dir, [_row(**{field: value})])

    with pytest.raises(SummaryError, match=message):
        load_rows(run_dir)


def test_ok_whitespace_output_is_unhealthy(tmp_path):
    run_dir = tmp_path / "run"
    _write_rows(run_dir, [_row(output_text=" \n\t ")])

    summary = summarize(run_dir, load_rows(run_dir))

    assert summary["ok_empty_outputs"] == 1


def test_absent_and_partial_cost_are_explicit(tmp_path):
    no_cost_dir = tmp_path / "no-cost"
    no_cost = _row()
    del no_cost["cost_usd"]
    _write_rows(no_cost_dir, [no_cost])
    no_cost_summary = summarize(no_cost_dir, load_rows(no_cost_dir))
    no_cost_report = render_markdown(no_cost_summary)
    assert no_cost_summary["cost"] == {"records": 0, "total": None, "mean": None}
    assert "Not available (`cost_usd` absent)" in no_cost_report

    partial_dir = tmp_path / "partial"
    _write_rows(partial_dir, [no_cost, _row("t2", cost_usd=0.25)])
    partial_summary = summarize(partial_dir, load_rows(partial_dir))
    partial_report = render_markdown(partial_summary)
    assert partial_summary["cost"]["records"] == 1
    assert partial_summary["cost"]["total"] == 0.25
    assert partial_summary["cost"]["mean"] == 0.25
    assert "present on only 1 of 2 records" in partial_report


def test_blank_lines_are_not_counted_as_raw_records(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frozen = run_dir / "frozen.jsonl"
    frozen.write_text("\n" + json.dumps(_row()) + "\n \n", encoding="utf-8")

    assert len(load_rows(run_dir)) == 1


def test_cli_success_and_failure_contract(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_rows(run_dir, [_row()])

    assert main([str(run_dir)]) == 0
    success = capsys.readouterr()
    assert success.out.startswith("# Benchmark Run Summary\n")
    assert success.err == ""

    assert main([str(tmp_path / "missing")]) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert failure.err.startswith("error: run directory does not exist:")

    assert main([]) == 2
    usage = capsys.readouterr()
    assert usage.out == ""
    assert usage.err == "usage: benchmark_summary.py <RUN_DIR>\n"

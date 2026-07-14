import json

import pytest

from gateway.db import connect
from scripts.rollup import rollup

DAY = "2026-07-14"


def _seed(db_path):
    conn = connect(db_path)

    # r1: single model, succeeded straight away.
    conn.execute(
        "INSERT INTO requests VALUES "
        "('r1','2026-07-14T00:00:00+00:00','prism','deepseek-chat','succeeded',"
        "'2026-07-14T00:00:01+00:00')"
    )
    conn.execute(
        "INSERT INTO ledger "
        "(request_id, provider, model, state, est_cost_usd, actual_cost_usd, "
        "usage_source, in_tokens, out_tokens, latency_ms, created_at, settled_at) "
        "VALUES ('r1','deepseek','deepseek-chat','settled',0.001,0.0012,'reported',"
        "100,20,800,'2026-07-14T00:00:00+00:00','2026-07-14T00:00:01+00:00')"
    )

    # r2: first model fails (call.failed event), falls back to a second
    # model which succeeds. Overall request status is 'succeeded'.
    conn.execute(
        "INSERT INTO requests VALUES "
        "('r2','2026-07-14T00:05:00+00:00','prism','deepseek-chat','succeeded',"
        "'2026-07-14T00:05:02+00:00')"
    )
    conn.execute(
        "INSERT INTO ledger "
        "(request_id, provider, model, state, est_cost_usd, actual_cost_usd, "
        "usage_source, in_tokens, out_tokens, latency_ms, created_at, settled_at) "
        "VALUES ('r2','deepseek','deepseek-chat','failed',0.0009,NULL,NULL,"
        "NULL,NULL,NULL,'2026-07-14T00:05:00+00:00',NULL)"
    )
    conn.execute(
        "INSERT INTO ledger "
        "(request_id, provider, model, state, est_cost_usd, actual_cost_usd, "
        "usage_source, in_tokens, out_tokens, latency_ms, created_at, settled_at) "
        "VALUES ('r2','glm','glm-4.7','settled',0.002,0.0021,'reported',"
        "150,30,1200,'2026-07-14T00:05:01+00:00','2026-07-14T00:05:02+00:00')"
    )
    conn.execute(
        "INSERT INTO events (request_id, parent_seq, created_at, kind, payload) "
        "VALUES ('r2', NULL, '2026-07-14T00:05:00+00:00', 'call.failed', ?)",
        (json.dumps({"model": "deepseek-chat", "status": 500}),),
    )

    # r3: only model fails, no fallback available -> request fails.
    conn.execute(
        "INSERT INTO requests VALUES "
        "('r3','2026-07-14T00:10:00+00:00','prism','deepseek-chat','failed',"
        "'2026-07-14T00:10:01+00:00')"
    )
    conn.execute(
        "INSERT INTO ledger "
        "(request_id, provider, model, state, est_cost_usd, actual_cost_usd, "
        "usage_source, in_tokens, out_tokens, latency_ms, created_at, settled_at) "
        "VALUES ('r3','deepseek','deepseek-chat','failed',0.0011,NULL,NULL,"
        "NULL,NULL,NULL,'2026-07-14T00:10:00+00:00',NULL)"
    )
    conn.execute(
        "INSERT INTO events (request_id, parent_seq, created_at, kind, payload) "
        "VALUES ('r3', NULL, '2026-07-14T00:10:00+00:00', 'call.failed', ?)",
        (json.dumps({"model": "deepseek-chat", "status": 500}),),
    )

    conn.execute(
        "INSERT INTO budgets VALUES ('M1', 5.0, 'active', '2026-07-14T00:10:01+00:00')"
    )

    conn.commit()
    conn.close()


def test_rollup_computes_per_day_metrics(tmp_path):
    db_path = tmp_path / "g.sqlite"
    _seed(db_path)

    data = rollup(db_path)

    assert list(data["days"]) == [DAY]
    day = data["days"][DAY]

    # requests grouped by status: 2 succeeded (r1, r2 via fallback), 1 failed (r3).
    assert day["requests_by_status"] == {"succeeded": 2, "failed": 1}

    # cost sums only settled/preflight/orphaned ledger rows: r1 (0.0012) +
    # r2's settled fallback row (0.0021). The two 'failed' ledger rows
    # (r2's first attempt, r3) must be excluded even though they carry an
    # est_cost_usd.
    assert day["cost_usd"] == pytest.approx(0.0012 + 0.0021)

    # P50/P95 over settled latency_ms only: [800, 1200].
    assert day["latency_p50_ms"] == pytest.approx(1000.0)
    assert day["latency_p95_ms"] == pytest.approx(1180.0)

    # fallback count = events with kind='call.failed' (r2's first attempt, r3).
    assert day["fallback_count"] == 2

    assert data["budgets"] == [
        {"name": "M1", "cap_usd": 5.0, "state": "active",
         "updated_at": "2026-07-14T00:10:01+00:00"}
    ]


def test_rollup_empty_db(tmp_path):
    db_path = tmp_path / "empty.sqlite"
    connect(db_path).close()

    data = rollup(db_path)

    assert data["days"] == {}
    assert data["budgets"] == []

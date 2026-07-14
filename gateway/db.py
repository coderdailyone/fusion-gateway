from __future__ import annotations
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    client TEXT,
    requested_model TEXT,
    status TEXT CHECK (status IN ('open','succeeded','failed','orphaned')),
    finished_at TEXT
);

CREATE TABLE events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT REFERENCES requests(id),
    parent_seq INTEGER,
    created_at TEXT,
    kind TEXT,
    payload TEXT
);

CREATE TABLE decisions (
    request_id TEXT REFERENCES requests(id),
    policy_version TEXT,
    action TEXT,
    features TEXT,
    degraded INTEGER DEFAULT 0
);

CREATE TABLE ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT REFERENCES requests(id),
    provider TEXT,
    model TEXT,
    state TEXT CHECK (state IN ('preflight','settled','failed','orphaned')),
    est_cost_usd REAL,
    actual_cost_usd REAL,
    usage_source TEXT,
    in_tokens INTEGER,
    out_tokens INTEGER,
    latency_ms INTEGER,
    created_at TEXT,
    settled_at TEXT
);

CREATE TABLE budgets (
    name TEXT PRIMARY KEY,
    cap_usd REAL,
    state TEXT CHECK (state IN ('active','tripped')),
    updated_at TEXT
);

CREATE TABLE schema_version (
    version INTEGER
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    applied = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if applied is None:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.commit()

    return conn


@dataclass
class Store:
    conn: sqlite3.Connection
    lock: threading.Lock = field(default_factory=threading.Lock)

from gateway.db import connect

def test_schema_idempotent(tmp_path):
    p = tmp_path / "g.sqlite"
    c1 = connect(p); c1.close()
    c2 = connect(p)
    v = c2.execute("SELECT version FROM schema_version").fetchone()[0]
    assert v == 1
    tables = {r[0] for r in c2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"requests","events","decisions","ledger","budgets"} <= tables

def test_orphan_status_constraint(tmp_path):
    c = connect(tmp_path / "g.sqlite")
    c.execute("INSERT INTO requests VALUES ('r1','t','prism','auto','open',NULL)")
    import sqlite3, pytest
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("UPDATE requests SET status='bogus' WHERE id='r1'")

import pytest
from gateway.db import connect, Store
from gateway.ledger import Ledger, BudgetTripped, estimate_tokens
from tests.helpers import FakeClock

def make_ledger(tmp_path, cap=1.0, cb=None):
    store = Store(connect(tmp_path / "g.sqlite"))
    store.conn.execute("INSERT INTO requests VALUES ('r1','t','prism','auto','open',NULL)")
    return Ledger(store, FakeClock(), cap_usd=cap, budget_name="T", alert_cb=cb)

def test_preflight_settle_and_drift(tmp_path):
    led = make_ledger(tmp_path)
    eid = led.preflight("r1", "deepseek", "deepseek-chat", 1000, 1000, 0.14, 0.28)
    drift = led.settle(eid, 1000, 500, "reported", 800, 0.14, 0.28)
    assert 0 <= drift < 1
    assert led.status()["consumed_usd"] == pytest.approx((1000*0.14 + 500*0.28) / 1e6)

def test_trip_at_cap_and_explicit_release(tmp_path):
    led = make_ledger(tmp_path, cap=0.0001)
    with pytest.raises(BudgetTripped):
        led.preflight("r1", "deepseek", "deepseek-chat", 10000, 10000, 10.0, 10.0)
    assert led.status()["state"] == "tripped"
    with pytest.raises(BudgetTripped):   # stays tripped even for tiny call
        led.preflight("r1", "deepseek", "deepseek-chat", 1, 1, 0.01, 0.01)
    led.release()
    assert led.status()["state"] == "active"

def test_alert_fires_once_at_80pct(tmp_path):
    hits = []
    led = make_ledger(tmp_path, cap=1.0, cb=lambda c, cap: hits.append(c))
    led.preflight("r1", "p", "m", 3_000_000, 0, 0.29, 0.0)   # ~0.87
    led.preflight("r1", "p", "m", 100, 0, 0.29, 0.0)
    assert len(hits) == 1

def test_failed_rows_do_not_consume(tmp_path):
    led = make_ledger(tmp_path)
    eid = led.preflight("r1", "p", "m", 1000, 1000, 1.0, 1.0)
    before = led.status()["consumed_usd"]; assert before > 0
    led.fail(eid)
    assert led.status()["consumed_usd"] == 0

def test_estimate_tokens():
    i, o = estimate_tokens([{"role":"user","content":"x"*400}], None)
    assert i == 100 and o == 1024

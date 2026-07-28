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


# -- no-cap budgets (cap_usd omitted from the config) ---------------------

def test_no_cap_never_trips_however_much_is_spent(tmp_path):
    led = make_ledger(tmp_path, cap=None)
    for i in range(5):                    # each call is ~$200 of estimate
        led.preflight("r1", "deepseek", "deepseek-chat", 10_000_000, 10_000_000, 10.0, 10.0)
    assert led.status()["state"] == "active"
    assert led.status()["cap_usd"] is None
    assert led.status()["consumed_usd"] > 500

def test_no_cap_never_fires_the_alert(tmp_path):
    hits = []
    led = make_ledger(tmp_path, cap=None, cb=lambda c, cap: hits.append(c))
    led.preflight("r1", "deepseek", "deepseek-chat", 10_000_000, 10_000_000, 10.0, 10.0)
    assert hits == []                     # 80% of "no cap" is not a number

def test_no_cap_still_honours_a_manual_trip(tmp_path):
    # Removing the ceiling must not remove the operator's ability to stop
    # spending: without this, an unbounded budget would have no brake at all.
    led = make_ledger(tmp_path, cap=None)
    led.trip()
    with pytest.raises(BudgetTripped):
        led.preflight("r1", "deepseek", "deepseek-chat", 1, 1, 0.01, 0.01)
    led.release()
    led.preflight("r1", "deepseek", "deepseek-chat", 1, 1, 0.01, 0.01)


def test_config_cap_is_authoritative_over_an_existing_row(tmp_path):
    # The budgets row used to be written once and never reconciled, so editing
    # cap_usd in gateway.toml silently did nothing on any host that had already
    # run. An operator raising the cap would still be tripping at the old one.
    store = Store(connect(tmp_path / "g.sqlite"))
    store.conn.execute("INSERT INTO requests VALUES ('r1','t','prism','auto','open',NULL)")
    Ledger(store, FakeClock(), cap_usd=5.0, budget_name="T")
    reopened = Ledger(store, FakeClock(), cap_usd=None, budget_name="T")
    assert reopened.status()["cap_usd"] is None
    again = Ledger(store, FakeClock(), cap_usd=50.0, budget_name="T")
    assert again.status()["cap_usd"] == 50.0

def test_reconciling_the_cap_does_not_un_trip_a_tripped_budget(tmp_path):
    # Only an explicit release may clear a trip; a restart must not.
    store = Store(connect(tmp_path / "g.sqlite"))
    store.conn.execute("INSERT INTO requests VALUES ('r1','t','prism','auto','open',NULL)")
    led = Ledger(store, FakeClock(), cap_usd=0.0001, budget_name="T")
    with pytest.raises(BudgetTripped):
        led.preflight("r1", "deepseek", "deepseek-chat", 10000, 10000, 10.0, 10.0)
    restarted = Ledger(store, FakeClock(), cap_usd=None, budget_name="T")
    assert restarted.status()["state"] == "tripped"
    with pytest.raises(BudgetTripped):
        restarted.preflight("r1", "deepseek", "deepseek-chat", 1, 1, 0.01, 0.01)

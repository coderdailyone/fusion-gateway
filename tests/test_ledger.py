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


# -- usage_for_request: the client-facing token total for a fanned-out request --

def test_usage_for_request_sums_every_call_not_just_the_last(tmp_path):
    """A fusion request's usage is the panel total. Reporting only the final
    leg is the specific bug this exists to prevent."""
    led = make_ledger(tmp_path)
    for in_tok, out_tok in ((100, 200), (300, 400), (7, 9)):
        eid = led.preflight("r1", "p", "m", in_tok, out_tok, 1.0, 1.0)
        led.settle(eid, in_tok, out_tok, "reported", 10, 1.0, 1.0)
    assert led.usage_for_request("r1") == (407, 609)


def test_usage_for_request_ignores_failed_calls(tmp_path):
    """'failed' is the state for a call that never reached an upstream, so it
    owes no tokens -- unlike a cancelled call, which settles as 'estimated'."""
    led = make_ledger(tmp_path)
    ok = led.preflight("r1", "p", "m", 10, 20, 1.0, 1.0)
    led.settle(ok, 10, 20, "reported", 1, 1.0, 1.0)
    led.fail(led.preflight("r1", "p", "m", 999, 999, 1.0, 1.0))
    assert led.usage_for_request("r1") == (10, 20)


def test_usage_for_request_ignores_an_unsettled_preflight_row(tmp_path):
    """A 'preflight' row has NULL token columns until it settles. SQLite's SUM
    skips NULLs, so the completed calls still total correctly."""
    led = make_ledger(tmp_path)
    eid = led.preflight("r1", "p", "m", 10, 20, 1.0, 1.0)
    led.settle(eid, 10, 20, "reported", 1, 1.0, 1.0)
    led.preflight("r1", "p", "m", 5, 5, 1.0, 1.0)   # left in flight
    assert led.usage_for_request("r1") == (10, 20)


def test_usage_for_request_returns_zero_when_no_call_ever_settled(tmp_path):
    """SQLite's SUM skips NULLs but returns NULL when EVERY value is NULL --
    and over no rows at all. Without COALESCE this is int(None), a TypeError
    raised mid-response. It is reachable: a panel whose every call failed
    leaves no row in a consuming state, and the streaming fallback still asks
    for a usage body before serving a candidate."""
    led = make_ledger(tmp_path)
    assert led.usage_for_request("r1") == (0, 0)          # no rows at all
    led.preflight("r1", "p", "m", 5, 5, 1.0, 1.0)          # only NULL columns
    assert led.usage_for_request("r1") == (0, 0)
    led.fail(led.preflight("r1", "p", "m", 9, 9, 1.0, 1.0))
    assert led.usage_for_request("r1") == (0, 0)


def test_usage_for_request_is_scoped_to_its_own_request(tmp_path):
    led = make_ledger(tmp_path)
    led.store.conn.execute(
        "INSERT INTO requests VALUES ('r2','t','prism','auto','open',NULL)")
    for rid, toks in (("r1", (10, 20)), ("r2", (500, 600))):
        eid = led.preflight(rid, "p", "m", *toks, 1.0, 1.0)
        led.settle(eid, *toks, "reported", 1, 1.0, 1.0)
    assert led.usage_for_request("r1") == (10, 20)
    assert led.usage_for_request("r2") == (500, 600)


# -- observed_out_rate: pricing a cancelled leg from evidence, not from the cap --

def _settle(led, model, out_tokens, latency_ms, source="reported"):
    eid = led.preflight("r1", "p", model, 10, out_tokens, 1.0, 1.0)
    led.settle(eid, 10, out_tokens, source, latency_ms, 1.0, 1.0)


def test_observed_out_rate_is_none_without_evidence(tmp_path):
    """No history -> no estimate. The caller must fall back to max_tokens,
    because over-estimating is the safe direction when nothing is known."""
    assert make_ledger(tmp_path).observed_out_rate("m") is None


def test_observed_out_rate_measures_tokens_per_second(tmp_path):
    led = make_ledger(tmp_path)
    _settle(led, "m", out_tokens=100, latency_ms=2000)     # 50 tok/s
    assert led.observed_out_rate("m") == pytest.approx(50.0)


def test_observed_out_rate_takes_the_median_not_the_mean(tmp_path):
    """One pathological call -- a reasoning model that thought for 40s and
    emitted 12 tokens -- must not drag the estimate for every cancellation
    after it. Mean here would be ~33.4; median is 50."""
    led = make_ledger(tmp_path)
    _settle(led, "m", out_tokens=100, latency_ms=2000)     # 50 tok/s
    _settle(led, "m", out_tokens=50, latency_ms=1000)      # 50 tok/s
    _settle(led, "m", out_tokens=12, latency_ms=40000)     # 0.3 tok/s
    assert led.observed_out_rate("m") == pytest.approx(50.0)


def test_observed_out_rate_ignores_estimated_rows(tmp_path):
    """An estimated row is this estimator's OWN output. Counting it would let
    each cancellation teach the next one a number no upstream confirmed."""
    led = make_ledger(tmp_path)
    _settle(led, "m", out_tokens=100, latency_ms=2000)                    # 50 tok/s
    _settle(led, "m", out_tokens=8192, latency_ms=1000, source="estimated")
    assert led.observed_out_rate("m") == pytest.approx(50.0)


def test_observed_out_rate_is_per_model(tmp_path):
    led = make_ledger(tmp_path)
    _settle(led, "fast", out_tokens=100, latency_ms=1000)   # 100 tok/s
    _settle(led, "slow", out_tokens=10, latency_ms=1000)    # 10 tok/s
    assert led.observed_out_rate("fast") == pytest.approx(100.0)
    assert led.observed_out_rate("slow") == pytest.approx(10.0)

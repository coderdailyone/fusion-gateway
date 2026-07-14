from gateway.db import connect, Store
from gateway.events import EventLog, replay_actions
from tests.helpers import FakeClock

def make_log(tmp_path):
    store = Store(connect(tmp_path / "g.sqlite"))
    store.conn.execute("INSERT INTO requests VALUES ('r1','t','prism','auto','open',NULL)")
    return EventLog(store, FakeClock())

def test_append_and_trace_ordered(tmp_path):
    log = make_log(tmp_path)
    s1 = log.append("r1", "request.received", {})
    s2 = log.append("r1", "call.attempt", {"model": "deepseek-chat"}, parent_seq=s1)
    tr = log.trace("r1")
    assert [e.seq for e in tr] == [s1, s2]
    assert tr[1].parent_seq == s1 and tr[1].payload["model"] == "deepseek-chat"

def test_replay_is_deterministic(tmp_path):
    log = make_log(tmp_path)
    log.append("r1", "request.received", {})
    log.append("r1", "call.attempt", {"model": "m"})
    log.append("r1", "call.succeeded", {"model": "m"})
    a = replay_actions(log.trace("r1"))
    b = replay_actions(log.trace("r1"))
    assert a == b == ["request.received:-", "call.attempt:m", "call.succeeded:m"]

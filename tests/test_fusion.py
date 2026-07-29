import asyncio
import time
import pytest
from gateway.config import FusionCfg, ModelCfg, ProviderCfg
from gateway.db import connect, Store
from gateway.events import EventLog
from gateway.ledger import Ledger
from gateway.fusion_prompts import Verdict
from gateway.fusion import (
    PanelResult, gather_panel, is_consensus, fuser_body, best_candidate,
    openai_response,
)
from gateway.providers import ProviderError
from tests.helpers import FakeClock


@pytest.fixture
def anyio_backend():
    return "asyncio"


FCFG = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                 reviewers=("a", "b"), fuser="b",
                 review_max_tokens=512, stage_timeout_s=5)


class FakeCfg:
    """Minimal stand-in for gateway.config.Config."""
    def __init__(self):
        self.models = {
            n: ModelCfg(name=n, provider="p", upstream_model=n, in_usd_per_mtok=1.0,
                        out_usd_per_mtok=1.0, fallback=())
            for n in ("a", "b", "s")
        }


class FakeAdapter:
    """Returns scripted text per upstream model; `delay` models a slow leg."""
    def __init__(self, script, delays=None, errors=()):
        self.script = script          # upstream_model -> text (or callable)
        self.delays = delays or {}
        self.errors = set(errors)
        self.calls = []

    async def chat(self, upstream_model, payload):
        self.calls.append((upstream_model, payload))
        await asyncio.sleep(self.delays.get(upstream_model, 0))
        if upstream_model in self.errors:
            raise ProviderError("p", "http", status=500)
        text = self.script.get(upstream_model, "")
        if callable(text):
            text = text(payload)
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


def make_env(tmp_path, adapter, fcfg=FCFG):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(connect(tmp_path / "g.sqlite"))
    clock = FakeClock()
    store.conn.execute(
        "INSERT INTO requests VALUES ('r1','t','prism','fusion','open',NULL)")
    store.conn.commit()
    return dict(fcfg=fcfg, cfg=FakeCfg(), adapters={"p": adapter},
                ledger=Ledger(store, clock, cap_usd=None, budget_name="T"),
                events=EventLog(store, clock), clock=clock,
                request_id="r1"), store


BODY = {"messages": [{"role": "user", "content": "2+2?"}]}


def agree_script(text="4"):
    """a and b answer; both review the other as correct; s is slow."""
    def review_or_answer(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} correct fine" for t in targets)
        return text
    return {"a": review_or_answer, "b": review_or_answer, "s": review_or_answer}


@pytest.mark.anyio
async def test_quorum_agreement_short_circuits_and_cancels_the_slow_leg(tmp_path):
    ad = FakeAdapter(agree_script(), delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.path == "quorum"
    assert set(panel.candidates) == {"a", "b"}     # s never contributed
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger ORDER BY id").fetchall()
    states = {(r["model"], r["state"]) for r in rows}
    # The cancelled leg is SETTLED with an estimate, never failed (that would
    # post $0 for work the upstream may bill) and never left in preflight.
    assert ("s", "settled") in states
    assert [r["usage_source"] for r in rows if r["model"] == "s"] == ["estimated"]
    assert not any(r["state"] == "preflight" for r in rows)


@pytest.mark.anyio
async def test_disagreement_waits_for_the_slow_leg(tmp_path):
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong nope" for t in targets)
        return "an answer"
    ad = FakeAdapter({"a": script, "b": script, "s": script}, delays={"s": 0.05})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.path == "full"
    assert set(panel.candidates) == {"a", "b", "s"}


@pytest.mark.anyio
async def test_a_wrong_verdict_forces_the_full_path_even_if_answers_match(tmp_path):
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong disputed" for t in targets)
        return "identical"
    ad = FakeAdapter({"a": script, "b": script, "s": script})
    env, _ = make_env(tmp_path, ad)
    assert (await gather_panel(body=BODY, **env)).path == "full"


@pytest.mark.anyio
async def test_slow_leg_starts_at_t0_not_after_the_quorum(tmp_path):
    # The whole latency argument rests on this. If the slow leg were launched
    # after the quorum decided, the full path would cost quorum + slow instead
    # of max(quorum, slow).
    ad = FakeAdapter(agree_script(), delays={"a": 0.2, "b": 0.2, "s": 0.2})
    env, _ = make_env(tmp_path, ad)
    started = time.monotonic()
    await gather_panel(body=BODY, **env)
    # a, b and s all ran concurrently, then one review round: ~0.4s, not ~0.6s.
    assert time.monotonic() - started < 0.55


@pytest.mark.anyio
async def test_timed_out_quorum_member_leaves_no_running_task(tmp_path):
    """A quorum member that blows through stage_timeout_s is skipped by the
    collect loop (`continue`) -- but per the resolved ambiguity in the task
    brief, gather_panel must never return with that member's task still
    running. `asyncio.wait_for(asyncio.shield(...))` only cancels the wait
    wrapper, not the shielded task itself, so without an unconditional
    cleanup at the end of gather_panel the slow member's call_model
    coroutine would keep the upstream connection open past the request.
    """
    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                     reviewers=("a", "b"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.05)
    # "a" never finishes within stage_timeout_s; b and s are fast.
    ad = FakeAdapter(agree_script(), delays={"a": 10})
    env, store = make_env(tmp_path, ad, fcfg=fcfg)

    before = asyncio.all_tasks() - {asyncio.current_task()}
    await gather_panel(body=BODY, **env)
    after = asyncio.all_tasks() - {asyncio.current_task()}
    leftover = after - before

    assert not any(not t.done() for t in leftover), (
        "gather_panel returned with a panel task still running")

    # The timed-out leg still settles (with an estimate), never left dangling
    # in preflight -- same invariant the explicit-cancellation path relies on.
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger ORDER BY id").fetchall()
    a_rows = [r for r in rows if r["model"] == "a"]
    assert len(a_rows) == 1
    assert a_rows[0]["state"] == "settled"
    assert a_rows[0]["usage_source"] == "estimated"


def test_is_consensus_requires_every_pairwise_correct():
    c = {"a": "x", "b": "x"}
    assert is_consensus(c, {"a": {"b": Verdict("correct", "")},
                            "b": {"a": Verdict("correct", "")}})
    # a missing review is not agreement -- absence of evidence is not evidence
    assert not is_consensus(c, {"a": {"b": Verdict("correct", "")}})
    assert not is_consensus(c, {"a": {"b": Verdict("unsure", "")},
                                "b": {"a": Verdict("correct", "")}})
    assert not is_consensus({"a": "x"}, {})          # fewer than 2 candidates
    assert not is_consensus(c, {})


def test_fuser_body_drops_client_tools_and_messages():
    panel = PanelResult("Q", {"a": "x", "b": "y"}, {}, "quorum", False)
    body = {"messages": [{"role": "user", "content": "Q"}],
            "tools": [{"type": "function"}], "max_tokens": 99,
            "temperature": 0.3, "stream": True, "user": "someone"}
    out = fuser_body(FCFG, panel, body)
    assert "tools" not in out and "user" not in out
    assert len(out["messages"]) == 1 and out["messages"][0]["role"] == "user"
    assert "Candidate a" in out["messages"][0]["content"]
    assert out["max_tokens"] == 99 and out["temperature"] == 0.3


def test_best_candidate_prefers_panel_order():
    panel = PanelResult("Q", {"b": "second", "a": "first"}, {}, "full", False)
    assert best_candidate(FCFG, panel) == ("a", "first")
    assert best_candidate(FCFG, PanelResult("Q", {}, {}, "full", True)) is None


def test_openai_response_is_well_formed():
    r = openai_response("hi", "fusion", {"path": "quorum"})
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["model"] == "fusion" and r["fusion"]["path"] == "quorum"

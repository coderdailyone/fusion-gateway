import asyncio
import time
import pytest
from gateway.config import FusionCfg, ModelCfg, ProviderCfg
from gateway.db import connect, Store
from gateway.events import EventLog
from gateway.ledger import BudgetTripped, Ledger, estimate_tokens
from gateway.fusion_prompts import Verdict, build_review_prompt, render_conversation
from gateway.fusion import (
    Candidate, PanelResult, gather_panel, is_consensus, fuser_body,
    best_candidate, openai_response, call_model, _extract_text,
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
        # M9 Task 5: a script can return a full response dict (e.g. a
        # tool-call shape with its own "choices"/"usage") instead of plain
        # text, when a candidate needs to answer with `tool_calls` rather
        # than `content`. Passed through unchanged rather than wrapped a
        # second time as `{"content": <dict>}`, which every plain-string
        # script (every test predating M9 Task 5) still gets exactly as
        # before.
        if isinstance(text, dict):
            return text
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
    # Fix round 2 (coordinator review), finding 2: nothing anywhere asserted
    # `degraded is False` on a healthy run -- both quorum members answered
    # and agreed, so this is the one case that must NOT be flagged degraded.
    # Mutation-tested: hardcoding `degraded=True` at both PanelResult
    # construction sites in gather_panel passed all 23 pre-fix tests; this
    # assertion is what catches it (see task-4-report.md, Fix round 1).
    assert not panel.degraded
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
    # Fix round 1, finding 4: the original version of this test used
    # agree_script(), which takes the quorum ("consensus") path. On that path
    # the slow leg's task is cancelled without ever being awaited, so WHEN it
    # was launched cannot affect wall-clock either way -- the test could not
    # tell a correct t=0 launch apart from a deliberately serial
    # implementation that creates the slow task only after the quorum
    # decides. Measured directly (3 runs each, agree_script, 0.2s delay):
    #   real:   0.4493s, 0.4522s, 0.4531s
    #   serial: 0.4369s, 0.4357s, 0.4350s
    # Both comfortably pass the old `< 0.55` assertion, and serial is if
    # anything FASTER (it skips launching the slow leg's task at all on this
    # path) -- proof the old test could not tell a correct implementation
    # from a broken one.
    #
    # This version uses a DISAGREEING script instead, so the full path runs
    # and the slow leg is genuinely folded in and awaited -- only here does
    # its start time matter. Measured in this environment, 5 runs each, same
    # 0.2s delay on every panel member:
    #   real   (all three launched at t=0):              0.6656s - 0.6706s
    #   serial (slow leg launched only after the quorum
    #           decides, i.e. candidates + review, THEN
    #           slow leg + a second review round):        0.8626s - 0.8729s
    # The ~0.2s gap is the whole latency argument for this milestone. The
    # threshold below sits in the middle of it with margin both ways: a
    # regression back to "launch the slow leg only when needed" would push
    # this comfortably past 0.8s.
    def disagree_script():
        def script(payload):
            prompt = payload["messages"][0]["content"]
            if "VERDICT" in prompt:
                targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
                return "\n".join(f"VERDICT {t} wrong nope" for t in targets)
            return "an answer"
        return {"a": script, "b": script, "s": script}

    ad = FakeAdapter(disagree_script(), delays={"a": 0.2, "b": 0.2, "s": 0.2})
    env, _ = make_env(tmp_path, ad)
    started = time.monotonic()
    panel = await gather_panel(body=BODY, **env)
    elapsed = time.monotonic() - started
    assert panel.path == "full"          # disagreement -- the slow leg was actually used
    assert elapsed < 0.8, f"took {elapsed:.3f}s; see comment above for measured real/serial"


class _ReviewOnlyDelayAdapter:
    """Candidate calls (no "VERDICT" in the prompt) resolve instantly; review
    calls sleep `delay`. Isolates the review stage's own timeout behaviour
    from the candidate stage's, which FakeAdapter's flat per-model delay
    cannot do (a reviewer's candidate call and its review call share an
    upstream_model)."""
    def __init__(self, delay):
        self.delay = delay
        self.calls = []

    async def chat(self, upstream_model, payload):
        self.calls.append((upstream_model, payload))
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            await asyncio.sleep(self.delay)
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            text = "\n".join(f"VERDICT {t} wrong nope" for t in targets)
        else:
            text = "an answer"
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


@pytest.mark.anyio
async def test_review_stage_is_bounded_by_stage_timeout_s(tmp_path):
    """Fix round 1, finding 2. The review stage previously had no timeout at
    all -- `_cross_review`'s bare `asyncio.gather(...)` just awaited however
    long the reviewers' upstream calls took, bounded only by httpx's 120s
    default, paid TWICE on the full path (once for the quorum's review round,
    once more after the slow leg joins).

    Measured in this environment with stage_timeout_s=0.05 and a 1.0s review
    upstream (candidates resolve instantly; the disagreeing script forces
    both review rounds to run), 3 runs each:
      before this fix: 2.0748s, 2.0809s, 2.0767s  (~2x the 1.0s review delay,
                                                     i.e. fully unbounded)
      after:           0.1718s, 0.1670s, 0.1639s  (~2x the 0.05s budget, as
                                                     expected for two rounds)
    """
    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                     reviewers=("a", "b"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.05)
    ad = _ReviewOnlyDelayAdapter(delay=1.0)
    env, _ = make_env(tmp_path, ad, fcfg=fcfg)
    started = time.monotonic()
    panel = await gather_panel(body=BODY, **env)
    elapsed = time.monotonic() - started
    assert panel.path == "full"
    assert elapsed < 0.5, (
        f"took {elapsed:.3f}s; review stage should be bounded near "
        f"2x stage_timeout_s (0.05s), not the review upstream's 1.0s delay")


@pytest.mark.anyio
async def test_stage_timeout_bounds_the_whole_stage_not_each_member(tmp_path):
    """Fix round 1, finding 3. `collect()` used to give each member its own
    fresh `stage_timeout_s`, so the deadline accumulated: three hanging
    quorum members under one `collect()` call cost ~3x the configured
    budget, not ~1x.

    Measured in this environment with stage_timeout_s=0.3 and three quorum
    members that never respond (5s each), 1 run each (both are
    deterministic -- the whole point is that hung members contribute nothing
    but the timeout itself):
      before this fix: 0.930s  (~3.1x the 0.3s budget)
      after:           0.327s  (~1.1x the 0.3s budget)
    """
    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b", "s"),
                     reviewers=("a", "b", "s"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.3)
    ad = FakeAdapter(agree_script(), delays={"a": 5, "b": 5, "s": 5})
    env, _ = make_env(tmp_path, ad, fcfg=fcfg)
    started = time.monotonic()
    panel = await gather_panel(body=BODY, **env)
    elapsed = time.monotonic() - started
    assert panel.degraded and not panel.candidates    # all three hung
    assert elapsed < 0.6, (
        f"took {elapsed:.3f}s; stage_timeout_s=0.3 should bound the WHOLE "
        f"collect() call, not restart per member (which would cost ~0.9s "
        f"for three hanging members)")


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


@pytest.mark.anyio
async def test_budget_trip_mid_review_leaves_no_running_task_or_preflight_row(tmp_path):
    """Fix round 1, finding 1 (CRITICAL).

    The old `_cross_review` fanned reviewers out with a bare
    `asyncio.gather(*(one(r) for r in reviewers))`, default
    `return_exceptions=False`. `ledger.preflight` can raise `BudgetTripped`
    for ANY reviewer independently (each reviewer's call is its own ledger
    row). If a sibling reviewer is still in flight when one of them trips the
    budget, gather() propagates immediately without cancelling or awaiting
    that sibling -- its task is invisible to gather_panel's own `tasks` dict
    (that only tracks panel-level CANDIDATE tasks, never review tasks), so it
    is orphaned: still running, holding an upstream connection, its ledger
    row stuck in 'preflight' (a CONSUMING_STATE) for as long as the upstream
    call takes -- exactly when the budget accounting matters most.

    This test uses the REAL `Ledger` (not a stub) with a cap computed to fit
    every candidate call plus exactly one review call's preflight, but not a
    second: reviewer "a" is deliberately slow so its review task is still in
    flight when reviewer "b"'s preflight is checked immediately afterward and
    trips. Costs are computed with the same `estimate_tokens`/
    `build_review_prompt` calls production code uses, rather than hardcoded,
    so the cap tracks the real formula.
    """
    text = "4"
    tight_body = {"messages": [{"role": "user", "content": "2+2?"}],
                  "max_tokens": 50}  # keeps candidate estimates comparable to
                                     # review estimates (no default 1024 out)

    # Cost of one candidate call once SETTLED (FakeAdapter always reports a
    # fixed usage of 3 prompt / 4 completion tokens, price $1/Mtok each way).
    candidate_settled_cost = 3 * 1.0 / 1e6 + 4 * 1.0 / 1e6
    # All three panel members (a, b, s) launch at t=0 and settle, whether or
    # not they end up in the quorum's `candidates` dict.
    three_candidates_settled = 3 * candidate_settled_cost

    # Cost of one reviewer's preflight ESTIMATE -- identical for "a" and "b"
    # here since each reviews exactly one same-length candidate ("4").
    conversation = render_conversation(tight_body["messages"])
    candidates_preview = {"a": text, "b": text}
    review_costs = set()
    for reviewer in ("a", "b"):
        prompt = build_review_prompt(conversation, candidates_preview, reviewer)
        in_tok, out_tok = estimate_tokens(
            [{"role": "user", "content": prompt}], FCFG.review_max_tokens)
        review_costs.add(in_tok * 1.0 / 1e6 + out_tok * 1.0 / 1e6)
    assert len(review_costs) == 1, "test assumption: both reviewers cost the same"
    review_cost = review_costs.pop()

    # Comfortably survives the candidate stage (settled cost is tiny), then
    # fits exactly 1.5 review calls: the first reviewer's preflight succeeds,
    # the second's does not.
    cap = three_candidates_settled + 1.5 * review_cost

    ad = FakeAdapter(agree_script(text), delays={"a": 0.3})
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(connect(tmp_path / "g.sqlite"))
    clock = FakeClock()
    store.conn.execute(
        "INSERT INTO requests VALUES ('r1','t','prism','fusion','open',NULL)")
    store.conn.commit()
    real_ledger = Ledger(store, clock, cap_usd=cap, budget_name="T")
    env = dict(fcfg=FCFG, cfg=FakeCfg(), adapters={"p": ad},
               ledger=real_ledger, events=EventLog(store, clock), clock=clock,
               request_id="r1")

    before = asyncio.all_tasks() - {asyncio.current_task()}
    with pytest.raises(BudgetTripped):
        await gather_panel(body=tight_body, **env)
    after = asyncio.all_tasks() - {asyncio.current_task()}
    leftover = after - before

    assert not any(not t.done() for t in leftover), (
        "gather_panel raised BudgetTripped with a review task still running")
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert not any(r["state"] == "preflight" for r in rows), (
        f"a ledger row was left in preflight: {[dict(r) for r in rows]}")


# -- Final whole-branch review, finding 3: the review stage ran twice even
# when the slow leg contributed nothing -- same reviewers, same candidates,
# same prompts, charged again for evidence that cannot change anything.

@pytest.mark.anyio
async def test_no_second_review_round_when_the_slow_leg_adds_nothing(tmp_path):
    """kimi-k3 is 403ing in production right now, so every disagreeing
    request was paying for this duplicate round today. Reproduced with the
    slow member failing outright (matches production): only ONE review round
    (2 calls: a reviews b, b reviews a) should ever run."""
    review_calls = []

    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            review_calls.append(prompt)
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong nope" for t in targets)
        return "an answer"

    ad = FakeAdapter({"a": script, "b": script}, errors={"s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.path == "full"
    assert set(panel.candidates) == {"a", "b"}      # s failed, never joined
    assert len(review_calls) == 2, (
        f"expected exactly 2 review calls (one round), got {len(review_calls)} "
        f"-- the slow leg added no candidate, so a second round must not run")


# -- Final whole-branch review, finding 4: the second review round REPLACED
# the first instead of merging into it. If round 2 wholly fails, already-paid
# round-1 verdicts were discarded and the fuser prompt fell back to "(no
# reviews available)" even though round 1 succeeded and was billed.

@pytest.mark.anyio
async def test_round_2_review_failure_does_not_discard_round_1_verdicts(tmp_path):
    calls = {"a": 0, "b": 0}

    def candidate_or_review(model):
        def fn(payload):
            prompt = payload["messages"][0]["content"]
            if "VERDICT" not in prompt:
                return f"answer-{model}"       # disagreeing candidate text
            calls[model] += 1
            if calls[model] > 1:
                raise RuntimeError("round 2 review blew up")
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong round1-verdict" for t in targets)
        return fn

    ad = FakeAdapter({"a": candidate_or_review("a"), "b": candidate_or_review("b"),
                      "s": lambda payload: "answer-s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)

    assert panel.path == "full"
    assert set(panel.candidates) == {"a", "b", "s"}   # slow leg joined -> round 2 ran
    # Round 1's verdicts must survive even though round 2 raised for both
    # reviewers -- a bare `reviews = round2_result` would leave this {}.
    assert panel.reviews.get("a", {}).get("b") is not None
    assert panel.reviews.get("b", {}).get("a") is not None


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
    # Fix round 1, finding 9: a leaked `stream` key would make the fuser call
    # itself streaming, which nothing downstream of gather_panel expects.
    assert "stream" not in out
    assert len(out["messages"]) == 1 and out["messages"][0]["role"] == "user"
    assert "Candidate a" in out["messages"][0]["content"]
    assert out["max_tokens"] == 99 and out["temperature"] == 0.3


# -- Final whole-branch review, finding 6: response_format/stop reached the
# candidates but were silently dropped for the fuser -- a client in JSON mode
# got prose back on the default (fusion) path, and `stop` was ignored by the
# answer the client actually received.

def test_fuser_body_passes_response_format_and_stop_but_never_n():
    panel = PanelResult("Q", {"a": "x", "b": "y"}, {}, "quorum", False)
    body = {"messages": [{"role": "user", "content": "Q"}],
            "response_format": {"type": "json_object"}, "stop": ["END"], "n": 2,
            "max_tokens": 99, "temperature": 0.3}
    out = fuser_body(FCFG, panel, body)
    assert out["response_format"] == {"type": "json_object"}
    assert out["stop"] == ["END"]
    # The fuser must return exactly one answer -- passing `n` through would
    # ask the upstream for several, and _extract_text only ever reads
    # choices[0], silently discarding the rest while still paying for them.
    assert "n" not in out


def test_best_candidate_prefers_panel_order():
    panel = PanelResult("Q", {"b": "second", "a": "first"}, {}, "full", False)
    assert best_candidate(FCFG, panel) == ("a", "first")
    assert best_candidate(FCFG, PanelResult("Q", {}, {}, "full", True)) is None


def test_best_candidate_skips_falsy_entries_in_the_sorted_fallback():
    # Fix round 1, finding 8. Neither key is in FCFG.panel, so the first loop
    # (configured panel order) finds nothing and falls through to the second
    # loop (sorted candidate names). That second loop used to `return m,
    # panel.candidates[m]` unconditionally, unlike the first -- so it could
    # hand back an empty answer. "x" sorts before "y"; without the fix this
    # would incorrectly return ("x", "").
    panel = PanelResult("Q", {"x": "", "y": "answer"}, {}, "full", True)
    assert best_candidate(FCFG, panel) == ("y", "answer")


def test_openai_response_is_well_formed():
    # M9 Task 5 step 5: openai_response used to accept a bare str too (the
    # fuser leg still returned one via _extract_text, coerced here with an
    # isinstance check) -- this test originally called it with "hi" directly
    # to pin that. Once the fuser leg was fixed to return a Candidate like
    # every other caller, the coercion was deleted on purpose (see
    # openai_response's docstring), so a bare str now raises AttributeError
    # instead of being silently accepted. Updated to pass a Candidate, which
    # is what every real caller has always done since Task 3.
    r = openai_response(Candidate("hi"), "fusion", {"path": "quorum"})
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["model"] == "fusion" and r["fusion"]["path"] == "quorum"


# -- Fix round 1, finding 7: the money paths in call_model had no coverage --

@pytest.mark.anyio
async def test_call_model_provider_error_fails_never_settles(tmp_path):
    """A ProviderError (upstream 4xx/5xx, timeout, network) must `fail` the
    ledger row, not settle it -- no usage was billed by the upstream, so
    settling would fabricate a cost."""
    ad = FakeAdapter({}, errors={"a"})
    env, store = make_env(tmp_path, ad)
    text = await call_model(model_name="a", body=BODY, cfg=env["cfg"],
                            adapters=env["adapters"], ledger=env["ledger"],
                            events=env["events"], clock=env["clock"],
                            request_id=env["request_id"], kind="candidate")
    assert text is None
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert [(r["model"], r["state"]) for r in rows] == [("a", "failed")]

    kinds = [e.kind for e in env["events"].trace("r1")]
    assert "call.failed" in kinds


class _BrokenAdapter:
    """Raises something that is neither ProviderError nor CancelledError --
    e.g. a translator bug, a JSON-decode error on a malformed 200, or any
    other upstream-adjacent surprise call_model's generic `except Exception`
    net is there to catch."""
    async def chat(self, upstream_model, payload):
        raise RuntimeError("translator exploded")


@pytest.mark.anyio
async def test_call_model_unexpected_exception_fails_not_500s(tmp_path):
    env, store = make_env(tmp_path, FakeAdapter({}))
    env["adapters"] = {"p": _BrokenAdapter()}
    text = await call_model(model_name="a", body=BODY, cfg=env["cfg"],
                            adapters=env["adapters"], ledger=env["ledger"],
                            events=env["events"], clock=env["clock"],
                            request_id=env["request_id"], kind="candidate")
    assert text is None
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert [(r["model"], r["state"]) for r in rows] == [("a", "failed")]


def test_extract_text_is_defensive_about_malformed_upstream_responses():
    """The response is upstream-controlled; a raw index/key access would let
    a malformed 200 500 the gateway. Every shape here must degrade to ''."""
    assert _extract_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    assert _extract_text(None) == ""                       # TypeError
    assert _extract_text("not even a dict") == ""           # TypeError
    assert _extract_text({}) == ""                          # KeyError: choices
    assert _extract_text({"choices": []}) == ""             # IndexError
    assert _extract_text({"choices": [{}]}) == ""           # KeyError: message
    assert _extract_text({"choices": [{"message": {}}]}) == ""       # KeyError: content
    assert _extract_text({"choices": [{"message": {"content": 123}}]}) == ""  # not str/list
    # OpenAI content-parts form, with a stray non-dict entry that must be skipped.
    parts = [{"type": "text", "text": "hi"}, "skip-me",
             {"type": "text", "text": "there"}]
    assert _extract_text({"choices": [{"message": {"content": parts}}]}) == "hi\nthere"


# -- Task 4: the degradation ladder -----------------------------------------

@pytest.mark.anyio
async def test_one_dead_panel_member_does_not_stop_fusion(tmp_path):
    ad = FakeAdapter(agree_script(), errors={"a"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert "a" not in panel.candidates and panel.degraded


@pytest.mark.anyio
async def test_a_single_surviving_candidate_yields_no_reviews(tmp_path):
    # Fewer than 2 candidates: there is nothing to cross-review and nothing to
    # fuse. app.py returns the survivor verbatim (Task 5).
    ad = FakeAdapter(agree_script(), errors={"a", "s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert set(panel.candidates) == {"b"} and panel.reviews == {}
    assert panel.degraded


@pytest.mark.anyio
async def test_zero_candidates_returns_an_empty_panel(tmp_path):
    ad = FakeAdapter(agree_script(), errors={"a", "b", "s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.candidates == {} and panel.degraded


@pytest.mark.anyio
async def test_all_reviews_failing_still_fuses(tmp_path):
    # Reviews are evidence, not a precondition. The fusion prompt already
    # renders "(no reviews available)".
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            raise RuntimeError("reviewer exploded")
        return "an answer"

    ad = FakeAdapter({"a": script, "b": script, "s": script})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.reviews == {} and panel.path == "full"
    assert len(panel.candidates) >= 2


@pytest.mark.anyio
async def test_a_slow_candidate_past_the_stage_timeout_is_dropped(tmp_path):
    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                     reviewers=("a", "b"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.05)
    ad = FakeAdapter(agree_script(), delays={"a": 1.0})
    env, store = make_env(tmp_path, ad)
    env["fcfg"] = fcfg
    panel = await gather_panel(body=BODY, **env)
    assert "a" not in panel.candidates
    # Fix round 2 (coordinator review), finding 3: the dropped candidate's
    # ledger row must still settle correctly -- not left in 'preflight', not
    # wrongly `fail`ed (that would post $0 for work the upstream may already
    # be billing). This is the same invariant
    # `test_timed_out_quorum_member_leaves_no_running_task` pins from the
    # "no leftover running task" angle; this asserts the ledger-row angle
    # directly for the same scenario.
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger WHERE model='a'").fetchall()
    assert len(rows) == 1
    assert rows[0]["state"] == "settled"
    assert rows[0]["usage_source"] == "estimated"


@pytest.mark.anyio
async def test_budget_tripped_propagates_and_strands_nothing(tmp_path):
    """Fix round 2 (coordinator review), finding 1 (CRITICAL).

    The original version of this test called `env["ledger"].trip()` BEFORE
    `gather_panel` ran at all, so EVERY panel member's `ledger.preflight`
    raises on its very first call -- no ledger row is ever inserted, for
    anyone. `assert not any(state == 'preflight')` was then vacuously true
    whether or not `gather_panel`'s `finally: await cancel_remaining()`
    cleanup ran, since there was never a row (or a running task) to strand
    in the first place. It only proved `BudgetTripped` propagates, which is
    trivial -- `call_model` never catches it. Mutation-tested: deleting the
    `finally: await cancel_remaining()` block from `gather_panel` left this
    test green (see task-4-report.md, Fix round 1).

    This version trips the budget genuinely MID-FLIGHT, the same shape as
    `test_budget_trip_mid_review_leaves_no_running_task_or_preflight_row`
    but at the CANDIDATE stage instead of the review stage: "a" is
    deliberately slow (10s, far past `stage_timeout_s`), so `collect()`'s
    `asyncio.wait_for(asyncio.shield(tasks["a"]), ...)` abandons waiting on
    it via a genuine TIMEOUT -- `shield` means task "a" itself keeps
    running in the background exactly as the module docstring describes.
    `collect()` then moves on to quorum member "b"; the cap is sized to
    admit exactly one candidate's preflight estimate (all three panel
    members preflight the same `BODY`, so their estimates are identical),
    so "b"'s preflight trips the budget while "a" is still asleep in the
    background -- a genuine sibling-in-flight scenario, not a vacuous one.
    """
    in_tok, out_tok = estimate_tokens(BODY["messages"], BODY.get("max_tokens"))
    candidate_cost = in_tok * 1.0 / 1e6 + out_tok * 1.0 / 1e6
    cap = 1.5 * candidate_cost  # room for exactly one candidate's preflight

    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                     reviewers=("a", "b"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.05)
    ad = FakeAdapter(agree_script(), delays={"a": 10})
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(connect(tmp_path / "g.sqlite"))
    clock = FakeClock()
    store.conn.execute(
        "INSERT INTO requests VALUES ('r1','t','prism','fusion','open',NULL)")
    store.conn.commit()
    real_ledger = Ledger(store, clock, cap_usd=cap, budget_name="T")
    env = dict(fcfg=fcfg, cfg=FakeCfg(), adapters={"p": ad},
               ledger=real_ledger, events=EventLog(store, clock), clock=clock,
               request_id="r1")

    before = asyncio.all_tasks() - {asyncio.current_task()}
    with pytest.raises(BudgetTripped):
        await gather_panel(body=BODY, **env)
    after = asyncio.all_tasks() - {asyncio.current_task()}
    leftover = after - before

    assert not any(not t.done() for t in leftover), (
        "gather_panel raised BudgetTripped with a candidate task still running")
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger").fetchall()
    assert not any(r["state"] == "preflight" for r in rows), (
        f"a ledger row was left in preflight: {[dict(r) for r in rows]}")
    # "a" is the sibling genuinely in flight when the trip landed on "b": it
    # must be SETTLED with the preflight estimate (call_model's
    # CancelledError branch), never left running past this function's
    # return and never `fail`ed (fail() would post $0 for work the upstream
    # may already be billing).
    a_rows = [r for r in rows if r["model"] == "a"]
    assert len(a_rows) == 1
    assert a_rows[0]["state"] == "settled"
    assert a_rows[0]["usage_source"] == "estimated"


@pytest.mark.anyio
async def test_no_ledger_row_is_ever_left_in_preflight(tmp_path):
    # The invariant that matters most: 'preflight' is a CONSUMING_STATE cleared
    # only by _recover_orphans at startup, so a stranded row holds budget
    # forever. Exercise every path and assert none strands.
    for errors, delays in (((), {"s": 3}), (("a",), {}), (("a", "b", "s"), {})):
        ad = FakeAdapter(agree_script(), delays=delays, errors=errors)
        env, store = make_env(tmp_path / f"{errors}{delays}", ad)
        await gather_panel(body=BODY, **env)
        rows = store.conn.execute("SELECT state FROM ledger").fetchall()
        assert not any(r["state"] == "preflight" for r in rows), (errors, delays)


# -- M9 Task 3 review, finding 2: no test anywhere asserted a tool-calls-only
# candidate actually SURVIVES the panel -- only `Candidate.__bool__` stands
# between that and reinstating M8 finding 1a (the tool call made the old
# `_extract_text` return "", `collect()`'s `if text:` dropped it, a
# fully-billed panel returned 502). Mutating `__bool__` to `return
# bool(self.text)` reinstates that bug exactly and every OTHER test in the
# suite stays green.

class _ToolCallAdapter:
    """Candidate calls answer with tool-calls only (content: null); review
    calls answer honestly with VERDICT text -- isolates collect()'s
    truthiness handling of a Candidate with empty .text from everything
    else gather_panel does.

    M9 Task 5: "a" and "b" now propose DIFFERENT arguments and "s" errors
    outright. Before Task 5, all three proposed the exact same call, which
    was irrelevant here -- there was no structural comparison yet. After
    Task 5, decide_tools would read that identical call as "agree_readonly"
    (it's the default-readonly "read" tool) and correctly fast-path to a
    SINGLE winner, which is Task 5's whole point but not what THIS test —
    predating the decision tree -- is about: whether a tool-calls-only
    Candidate survives collect()'s truthiness gate at all. Keeping "a" and
    "b" genuinely disagreeing (and "s" out of the picture entirely) sends
    decide_tools straight to "disagree", plurality finds no winner once
    slow errors out and never joins, and control falls through unchanged
    into the pre-Task-5 _cross_review/is_consensus path this test actually
    exercises.
    """
    def __init__(self):
        self.calls = []

    async def chat(self, upstream_model, payload):
        self.calls.append((upstream_model, payload))
        if upstream_model == "s":
            raise ProviderError("p", "http", status=500)
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            text = "\n".join(f"VERDICT {t} correct fine" for t in targets)
            return {"choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        args = '{"path":"a.py"}' if upstream_model == "a" else '{"path":"b.py"}'
        return {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read", "arguments": args}}]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


@pytest.mark.anyio
async def test_a_tool_call_only_candidate_survives_the_panel(tmp_path):
    ad = _ToolCallAdapter()
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert set(panel.candidates) == {"a", "b"}      # not dropped, quorum reached
    assert panel.candidates["a"].text == ""
    assert panel.candidates["a"].tool_calls
    assert panel.candidates["a"].tool_calls[0]["function"]["name"] == "read"


# -- pricing a cancelled leg (2026-08-05) ----------------------------------
# The quorum short-circuit kills the slow leg for latency. It used to be
# BILLED for max_tokens regardless -- measured live, the cancelled kimi leg
# was 93% of the whole request's cost, and it scaled with the client's
# max_tokens rather than with anything the model did.

class ClockAdvancingAdapter(FakeAdapter):
    """Advances the injected clock while a delayed leg 'works'.

    FakeClock is frozen, so without this every cancelled leg has an elapsed
    time of exactly zero and the throughput estimate is untestable -- it would
    read 0 tokens no matter what the estimator did.
    """
    def __init__(self, script, clock, advance_by, **kw):
        super().__init__(script, **kw)
        self._clock, self._advance_by = clock, advance_by

    async def chat(self, upstream_model, payload):
        if upstream_model in self.delays:
            self._clock.advance(self._advance_by)
        return await super().chat(upstream_model, payload)


def _seed_reported_history(store, clock, model, out_tokens, latency_ms, n=3):
    led = Ledger(store, clock, cap_usd=None, budget_name="T")
    for _ in range(n):
        eid = led.preflight("r1", "p", model, 10, out_tokens, 1.0, 1.0)
        led.settle(eid, 10, out_tokens, "reported", latency_ms, 1.0, 1.0)


@pytest.mark.anyio
async def test_a_cancelled_leg_is_priced_from_measured_throughput(tmp_path):
    """s has emitted 100 tokens in 2s (50 tok/s) on every past call, and this
    call ran 4s before being cancelled -> ~200 tokens, NOT the 4096 cap."""
    ad = FakeAdapter(agree_script(), delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    _seed_reported_history(store, env["clock"], "s", out_tokens=100, latency_ms=2000)
    env["adapters"]["p"] = ClockAdvancingAdapter(
        agree_script(), env["clock"], advance_by=4.0, delays={"s": 3})

    await gather_panel(body=dict(BODY, max_tokens=4096), **env)

    row = store.conn.execute(
        "SELECT out_tokens, usage_source FROM ledger "
        "WHERE model='s' AND usage_source='estimated' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["out_tokens"] == 200
    assert row["out_tokens"] < 4096, "the whole point: not the max_tokens cap"


@pytest.mark.anyio
async def test_a_cancelled_leg_is_capped_at_max_tokens(tmp_path):
    """max_tokens is a ceiling the upstream could not have exceeded, however
    fast the model's history says it is."""
    ad = FakeAdapter(agree_script(), delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    _seed_reported_history(store, env["clock"], "s", out_tokens=10000, latency_ms=100)
    env["adapters"]["p"] = ClockAdvancingAdapter(
        agree_script(), env["clock"], advance_by=60.0, delays={"s": 3})

    await gather_panel(body=dict(BODY, max_tokens=64), **env)

    row = store.conn.execute(
        "SELECT out_tokens FROM ledger WHERE model='s' AND usage_source='estimated' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row["out_tokens"] == 64


@pytest.mark.anyio
async def test_a_cancelled_leg_falls_back_to_max_tokens_with_no_history(tmp_path):
    """No reported call for this model yet -> no evidence -> keep the old
    over-estimate. Over-estimating is the safe direction when nothing is
    known; under-estimating would silently under-report real spend."""
    ad = ClockAdvancingAdapter(agree_script(), FakeClock(), advance_by=4.0,
                               delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    ad._clock = env["clock"]

    await gather_panel(body=dict(BODY, max_tokens=777), **env)

    row = store.conn.execute(
        "SELECT out_tokens FROM ledger WHERE model='s' AND usage_source='estimated' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row["out_tokens"] == 777

import pytest
from gateway.fusion import (
    Candidate, PanelResult, _extract_message, best_candidate, call_model,
    decide_tools, fuser_body, gather_panel, openai_response,
)
from tests.test_fusion import FakeAdapter, make_env, BODY, FCFG  # fixtures


def _resp(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def test_extract_message_reads_plain_text():
    got = _extract_message(_resp(content="hello"))
    assert got == Candidate("hello", ())


def test_extract_message_reads_a_tool_call():
    # This is the root cause of the original CRITICAL: the old _extract_text
    # returned "" here, every candidate was dropped, and a fully-billed panel
    # handed back a 502.
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]
    got = _extract_message(_resp(content=None, tool_calls=calls))
    assert got.text == "" and len(got.tool_calls) == 1
    assert got.tool_calls[0]["function"]["name"] == "read"


def test_extract_message_reads_text_and_a_call_together():
    calls = [{"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}]
    got = _extract_message(_resp(content="let me look", tool_calls=calls))
    assert got.text == "let me look" and len(got.tool_calls) == 1


def test_extract_message_survives_hostile_shapes():
    # M9 Task 3 review, finding 3: `isinstance(got, Candidate)` alone is not
    # a bite -- ANY non-raising implementation satisfies it, including one
    # that weakens the extractor's own guard from
    # `isinstance(raw, (list, tuple))` to `raw is not None`. That weakened
    # guard would still pass this test right up until a real upstream sent
    # `tool_calls: 5` (or another non-iterable scalar), which
    # `tuple(c for c in 5 if ...)` cannot iterate -- a TypeError, i.e. a
    # gateway 500. Asserting `got.tool_calls == ()` on every shape below
    # (not just "didn't raise") is what catches that.
    for resp in ({}, {"choices": []}, {"choices": [{}]}, {"choices": "x"},
                 _resp(content=None, tool_calls="notalist"),
                 _resp(content=None, tool_calls=[None]),
                 _resp(content=None, tool_calls=5),
                 _resp(content=None, tool_calls="x"),
                 _resp(content=None, tool_calls=True)):
        got = _extract_message(resp)
        assert isinstance(got, Candidate)
        assert got.tool_calls == (), f"resp={resp!r} -> tool_calls={got.tool_calls!r}"


def test_candidate_bool_is_true_for_tool_calls_alone():
    # M9 Task 3 review, finding 2: this is the load-bearing half of the
    # CRITICAL fix (M8 final review, finding 1a). `collect()`'s `if text:`
    # guard and `best_candidate`'s `if c:` guard both rely on
    # `Candidate.__bool__` treating a tool-calls-only Candidate as truthy --
    # without it, a tool-calls-only candidate is silently dropped exactly
    # like the old bare-string `_extract_text` dropped it.
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    assert bool(Candidate("", calls)) is True


def test_candidate_bool_is_false_when_both_text_and_tool_calls_are_empty():
    assert bool(Candidate("")) is False


def test_openai_response_for_text_is_unchanged_in_shape():
    r = openai_response(Candidate("hi"), "fusion", {"path": "quorum"})
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in r["choices"][0]["message"]


def test_openai_response_for_a_tool_call():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    r = openai_response(Candidate("", calls), "fusion", {})
    msg = r["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "read"
    assert r["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_response_keeps_text_alongside_a_call():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    msg = openai_response(Candidate("looking", calls), "fusion", {})["choices"][0]["message"]
    assert msg["content"] == "looking" and len(msg["tool_calls"]) == 1


def test_openai_response_no_longer_coerces_a_bare_string():
    # Fix round 1, finding 5: proves Step 5 part 3's isinstance-coercion
    # deletion stays deleted. `test_openai_response_for_text_is_unchanged_
    # in_shape` above passes a Candidate and can't tell a re-added coercion
    # apart from its absence -- both accept a Candidate identically. Only a
    # bare string distinguishes them: it must raise now that every real
    # caller (both best_candidate's fallback and, since Task 5 step 5, the
    # fuser's own call) always produces a Candidate.
    with pytest.raises(AttributeError):
        openai_response("hi", "fusion", {"path": "quorum"})


# -- M9 Task 5: the decision tree ------------------------------------------

TOOLS_BODY = dict(BODY, tools=[{"type": "function",
                                "function": {"name": "read", "parameters": {}}}])


def tool_resp(name, args):
    return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": name, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


def test_decide_tools_treats_unparseable_calls_as_disagreement_not_prose():
    # Minor (fix round 1): decide_tools used to read "every candidate's
    # tool_calls fails to canonicalise" as `by_model`'s None-ness, which is
    # the SAME signal "no candidate proposed a call at all" uses -- an
    # all-unparseable panel read as "prose" and skipped the tool decision
    # tree entirely, contradicting both this function's own docstring and
    # tool_vote's "unparseable counts as disagreement" rule.
    bad = ({"id": "c", "type": "function",
            "function": {"name": "read", "arguments": "{not json"}},)
    candidates = {"a": Candidate("", bad), "b": Candidate("", bad)}
    verdict, winner = decide_tools(candidates, FCFG.readonly_tools)
    assert verdict == "disagree" and winner is None


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_identical_readonly_calls_emit_with_no_review_and_no_slow_leg(tmp_path):
    ad = FakeAdapter({m: (lambda p: tool_resp("read", '{"path":"a.py"}'))
                      for m in ("a", "b", "s")}, delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_fast"
    assert panel.reviews == {}                      # no review was dispatched
    # Winner determinism (fix round 1, finding 5): decide_tools picks
    # sorted(candidates)[0]. Mutating that to [-1] would silently swap in
    # "b" here and nothing about `path` alone would catch it.
    assert set(panel.candidates) == {"a"}
    # Fix round 1, finding 5: a hardcoded `degraded=True` at this
    # PanelResult site would pass every other assertion here -- this is the
    # one case that must NOT be flagged degraded (both quorum members
    # answered and agreed), mirroring the same mutation-tested assertion
    # the prose path's own quorum-consensus test carries.
    assert not panel.degraded
    # Exact row count for this branch, not a range: 2 quorum candidates +
    # the cancelled slow leg, which must settle an 'estimated' row rather
    # than vanish (the money invariant). No review, no fuser.
    rows = store.conn.execute("SELECT model, state, usage_source FROM ledger "
                              "ORDER BY id").fetchall()
    assert len(rows) == 3
    assert not any(r["state"] == "preflight" for r in rows)
    states = {(r["model"], r["state"], r["usage_source"]) for r in rows}
    assert ("a", "settled", "reported") in states
    assert ("b", "settled", "reported") in states
    assert ("s", "settled", "estimated") in states
    # fusion.tool_verdict event (fix round 1, finding 5): reports the real
    # verdict and the real winner, not a deleted event or a hardcoded None.
    verdict_events = [e for e in env["events"].trace("r1")
                      if e.kind == "fusion.tool_verdict"]
    assert len(verdict_events) == 1
    assert verdict_events[0].payload == {"verdict": "agree_readonly", "winner": "a"}


@pytest.mark.anyio
async def test_identical_write_class_calls_are_reviewed(tmp_path):
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a correct ok\nVERDICT b correct ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py","body":"x"}')
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_reviewed" and panel.reviews != {}
    assert set(panel.candidates) == {"a"}     # winner determinism
    assert not panel.degraded                 # healthy path, not a fallback
    # Exact row count: 3 candidates (a, b, s -- s cancelled once the review
    # confirms agreement) + 2 reviews (a reviews b, b reviews a). No fuser.
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 5
    assert not any(r["state"] == "preflight" for r in rows)


# -- Fix round 1, finding 1 (CRITICAL): a reviewer objection on an agreeing
# write-class call was silently discarded, not cancelled. `agree_review`
# fell into the `disagree` branch on objection, and `disagree`'s own
# `plurality` re-elected exactly the pair whose agreement was the
# precondition for reaching `agree_review` in the first place -- the
# rejected write call was served anyway, with reviews={} and no fuser, one
# wasted billed call. The fix: an objection (or a wholly-failed review
# stage) on a write-class agreement must skip `plurality` entirely and
# reach the fuser-eligible full path, carrying whatever reviews exist.
#
# My first version of this test (see task-5-report.md's "Deviation 2" for
# the history) rationalised the unsatisfiable `"s" in panel.candidates`
# assertion as a quirk of the test's own construction rather than following
# it to the actual bug: the SAME reasoning that makes "s" unable to appear
# in a `tool_plurality` result is exactly why plurality can never safely
# resolve this branch. This version asserts the real invariant instead:
# the objection survives into `panel.reviews`, and the panel reaches a
# state where app.py's `_finish_fusion` will call the fuser (>= 2
# candidates), not silently re-serve the rejected call alone.

def _objected_write_script(s_reply):
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a wrong no\nVERDICT b wrong no"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py"}')

    def s_script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            # "s" is never asked to review in round 1 (it isn't a candidate
            # yet) and isn't a configured reviewer in round 2 either
            # (FCFG.reviewers is ("a", "b")) -- this branch is defensive,
            # not expected to fire, but must not blow up parse_review if it
            # somehow does.
            return {"choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return s_reply(p)
    return {"a": script, "b": script, "s": s_script}


@pytest.mark.anyio
async def test_a_reviewer_objection_on_a_write_class_call_escalates(tmp_path):
    ad = FakeAdapter(_objected_write_script(
        lambda p: tool_resp("write", '{"path":"a.py"}')))  # s agrees too
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path not in ("tool_reviewed", "tool_plurality", "tool_fast")
    assert panel.path == "full"
    # The objection survives into the result -- not silently dropped.
    assert panel.reviews != {}
    assert any(v.verdict == "wrong" for verds in panel.reviews.values()
              for v in verds.values())
    # >= 2 candidates: app.py's _finish_fusion will call the fuser rather
    # than taking the single-candidate shortcut and re-serving the
    # rejected call alone.
    assert len(panel.candidates) >= 2
    # Exact row count: 3 candidates (a, b, s -- all genuinely run, none
    # cancelled, since this now reaches the full path) + 2 reviews in the
    # agree_review round (a reviews b, b reviews a, over {a, b} only -- s
    # hasn't joined yet) + 2 more reviews in the fall-through's own round
    # (now over {a, b, s}). Not a duplicate of the first round (fix round
    # 1, finding 3) -- s is genuinely new evidence the first round never
    # saw.
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 7
    assert not any(r["state"] == "preflight" for r in rows)


@pytest.mark.parametrize("label,s_reply", [
    ("agrees", lambda p: tool_resp("write", '{"path":"a.py"}')),
    ("differs", lambda p: tool_resp("write", '{"path":"other.py"}')),
    ("prose", lambda p: {"choices": [{"message": {"content": "I would write it"}}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
    ("unparseable", lambda p: tool_resp("write", "{not json")),
])
@pytest.mark.anyio
async def test_an_objection_survives_regardless_of_what_the_slow_leg_proposes(
        tmp_path, label, s_reply):
    # Finding 1's proof was driven across all four slow-leg outcomes; this
    # pins all four so a regression narrowly scoped to one of them (e.g.
    # only "s agrees", the shape plurality re-elects most directly) doesn't
    # slip back in unnoticed.
    ad = FakeAdapter(_objected_write_script(s_reply))
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path not in ("tool_reviewed", "tool_plurality", "tool_fast"), label
    assert panel.reviews != {}, label
    assert any(v.verdict == "wrong" for verds in panel.reviews.values()
              for v in verds.values()), label
    assert len(panel.candidates) >= 2, label


@pytest.mark.anyio
async def test_a_wholly_failed_review_stage_on_write_class_agreement_escalates(tmp_path):
    # The same hole swallowed this degradation row too: a write-class
    # agreement where EVERY review call fails outright (reviews == {}, not
    # merely an objection) must escalate exactly like an objection does --
    # `if reviews and not objected` is false either way, and both must skip
    # `plurality`.
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            raise RuntimeError("reviewer down")
        return tool_resp("write", '{"path":"a.py"}')
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path not in ("tool_reviewed", "tool_plurality", "tool_fast")
    assert len(panel.candidates) >= 2
    # 3 candidates settled + 2 failed review attempts (round 1) + 2 more
    # failed review attempts (the fall-through's own round, now also
    # covering "s") -- reviews are legitimately empty here (both attempts
    # failed), unlike the objection tests above.
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 7
    assert sorted(r["state"] for r in rows) == ["failed"] * 4 + ["settled"] * 3


# -- Fix round 2, finding 1 (Important): the escalation branch
# (`disagree_no_plurality`) called `_cross_review` unconditionally, so a
# DEAD slow leg (errors or times out, adding no candidate) still triggered a
# second, byte-identical review round over the exact {a, b} set round 1
# (inside `agree_review`) already reviewed -- Finding 3's defect class
# reappearing in the branch Fix round 1 created. This is today's PRODUCTION
# shape: kimi-k3 is 403ing right now, so every objected write-class call was
# paying for two review calls that could not tell anyone anything new.

@pytest.mark.anyio
async def test_an_objection_with_a_dead_slow_leg_reviews_only_once(tmp_path):
    review_prompts = []

    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            review_prompts.append(prompt)
            return {"choices": [{"message": {"content":
                     "VERDICT a wrong no\nVERDICT b wrong no"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py"}')

    ad = FakeAdapter({"a": script, "b": script}, errors={"s"})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path not in ("tool_reviewed", "tool_plurality", "tool_fast")
    # Exactly ONE review round: two calls (a reviews b, b reviews a), two
    # DISTINCT prompts -- not four calls with a byte-identical duplicate.
    assert len(review_prompts) == 2
    assert len(set(review_prompts)) == 2
    # Exact row count: 2 candidates settled + "s" failed (still bills a row,
    # never vanishes) + 2 reviews (ONE round). Before this fix: 7 (the same
    # 3 candidate rows + 4 review rows, 2 of them a byte-identical repeat).
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 5
    assert sorted(r["state"] for r in rows) == ["failed"] + ["settled"] * 4


# -- Fix round 2, finding 2 (Important): `_merge_reviews` let a later,
# byte-identical (or merely re-rolled) round's "correct" silently overwrite
# an earlier "wrong" for the SAME (reviewer, target) pair. Reproduced: round
# 1 objects, round 2 (triggered because the slow leg genuinely joined) comes
# back clean -> `is_consensus` reads all-correct -> `path='quorum'` ->
# the fuser is told to COPY VERBATIM an answer a reviewer just rejected.
# This does not reopen the CRITICAL (the fuser is still always reached for
# any >=2-candidate panel), but it defeats the amended spec's requirement
# that escalation carry "whatever reviews exist" to the fuser.

@pytest.mark.anyio
async def test_an_objection_survives_a_round_2_reroll_that_comes_back_clean(tmp_path):
    calls = {"a": 0, "b": 0}

    def make(name):
        def fn(p):
            prompt = p["messages"][0]["content"]
            if "VERDICT" in prompt:
                calls[name] += 1
                if calls[name] == 1:
                    # Round 1 (over {a, b} only): objects.
                    return {"choices": [{"message": {"content":
                             "VERDICT a wrong no\nVERDICT b wrong no"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                # Round 2 (over {a, b, s}, once s joins): re-rolls clean.
                return {"choices": [{"message": {"content":
                         "VERDICT a correct ok\nVERDICT b correct ok\n"
                         "VERDICT s correct ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            return tool_resp("write", '{"path":"a.py"}')
        return fn

    # "s" agrees with the objected pair -- it is a genuinely new candidate,
    # so the fall-through review round is warranted (fix round 2, finding 1
    # does not skip it here) and round 2 legitimately runs.
    ad = FakeAdapter({"a": make("a"), "b": make("b"),
                      "s": lambda p: tool_resp("write", '{"path":"a.py"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    # The objection must not be laundered into "quorum" (a silent COPY
    # VERBATIM instruction to the fuser for an answer just rejected).
    assert panel.path == "full"
    assert panel.reviews["a"]["b"].verdict == "wrong"
    assert panel.reviews["b"]["a"].verdict == "wrong"


# -- Fix round 1, finding 2 (Important): the all_readonly gate applied only
# to a quorum agreement, not to a plurality winner -- a write-class 2-of-3
# plurality was served directly, with reviews={} and no fuser, carrying no
# more of a correctness check than a write-class structural agreement does
# (the milestone's own stated reason agree_review keeps the review at all).

@pytest.mark.anyio
async def test_a_write_class_plurality_does_not_emit_directly(tmp_path):
    verdict_text = "VERDICT a wrong no\nVERDICT b wrong no\nVERDICT s wrong no"

    def make(path):
        def fn(p):
            prompt = p["messages"][0]["content"]
            if "VERDICT" in prompt:
                return {"choices": [{"message": {"content": verdict_text}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            return tool_resp("write", f'{{"path":"{path}"}}')
        return fn

    # a and s agree on a WRITE call ("x"); b differs ("y") -- a genuine
    # 2-of-3 plurality, reached directly from decide_tools's "disagree"
    # (a and b themselves disagree, so this does NOT go through
    # agree_review at all -- a distinct code path from finding 1's tests).
    ad = FakeAdapter({"a": make("x"), "b": make("y"), "s": make("x")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path != "tool_plurality"
    assert panel.reviews != {}          # reached review, not a silent direct emit
    assert len(panel.candidates) >= 2   # reaches the fuser-eligible full path
    # Exact row count: 3 candidates + 2 reviews (ONE round -- this path
    # never goes through agree_review, so there is no earlier round to
    # merge with; fix round 1, finding 3's guard is a no-op here, not a
    # source of a second round).
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 5


@pytest.mark.anyio
async def test_two_of_three_plurality_wins_without_the_fuser(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"a"}')})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_plurality"
    # The MAJORITY call won (a/s's "a", not b's dissenting "b") -- and
    # nothing else survived: no review, a single candidate (so app.py's
    # _finish_fusion takes the <2-candidate shortcut and never calls the
    # fuser), not flagged degraded.
    assert set(panel.candidates) == {"a"}
    assert panel.candidates["a"].tool_calls[0]["function"]["arguments"] == '{"path":"a"}'
    assert panel.reviews == {}
    assert not panel.degraded
    # Exact row count: 3 candidates (a, b, s -- all three needed to compute
    # plurality), no review, no fuser.
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 3


@pytest.mark.anyio
async def test_a_three_way_split_falls_through_to_the_full_path(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"c"}')})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "full" and len(panel.candidates) == 3
    # Fix round 1, finding 3: this branch collects the slow leg once (inside
    # the `disagree` block, to compute plurality) and must NOT collect it
    # again in the pre-existing full-path code below -- that would re-await
    # the (already-done) tasks, come back non-empty, and trigger a SECOND,
    # byte-identical review round over the same three candidates. Exact row
    # count: 3 candidates + 2 reviews (ONE round, reviewers a and b review
    # each other -- s is not a configured reviewer). Before the fix this
    # was 7 (3 candidates + 2 + 2 duplicate reviews); this assertion is
    # exactly what the review measured and caught.
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 5


@pytest.mark.anyio
async def test_a_text_candidate_never_matches_a_tool_call(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: {"choices": [{"message": {"content": "I will read it"}}],
                                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                      "s": lambda p: tool_resp("read", '{"path":"a"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_plurality"           # a and s agreed; b dissented


@pytest.mark.anyio
async def test_two_text_candidates_still_take_the_prose_path(tmp_path):
    # The load-bearing regression: prose must not leak into the tool path just
    # because the request carried `tools`.
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a correct ok\nVERDICT b correct ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return {"choices": [{"message": {"content": "just prose"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "quorum"
    # Byte-identical to the pre-Task-5 prose path: 2 quorum candidates + the
    # cancelled slow leg + 2 reviews. No duplicated review round (fix round
    # 1, finding 3 does not change prose-path behaviour at all).
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 5
    # Minor: fusion.tool_verdict is gated to skip "prose" -- firing it on
    # every plain request (the common case) would double the event volume
    # for a verdict nothing downstream reads.
    verdict_events = [e for e in env["events"].trace("r1")
                      if e.kind == "fusion.tool_verdict"]
    assert verdict_events == []


def test_fuser_body_forwards_tools_when_the_panel_holds_calls():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    panel = PanelResult("Q", {"a": Candidate("", calls)}, {}, "full", False)
    out = fuser_body(FCFG, panel, TOOLS_BODY)
    assert out["tools"] == TOOLS_BODY["tools"]


def test_fuser_body_still_strips_tools_on_the_prose_path():
    panel = PanelResult("Q", {"a": Candidate("prose")}, {}, "quorum", False)
    assert "tools" not in fuser_body(FCFG, panel, TOOLS_BODY)


# -- M9 Task 5 step 5: the fuser-leg landmine. Task 3's review found that
# once `tools` are forwarded and the fuser answers with content: null +
# tool_calls, call_model(kind="fuser") using _extract_text would return ""
# for that shape -- _finish_fusion's `if text:` guard would then discard the
# fully-billed fuser answer and fall back to one of the three DISAGREEING
# candidates (best_candidate, picked by panel order), throwing away the
# arbitration at exactly the moment it is needed (M8 finding 1a reproduced
# in the fuser leg). This replicates app.py's _finish_fusion logic directly
# against fusion.py's own exported primitives (call_model, fuser_body,
# best_candidate, openai_response) -- Task 5's file scope is gateway/fusion.py
# only, and _finish_fusion is a private closure inside app.py's create_app
# that cannot be reached from a bare `gather_panel` call; an HTTP-level
# TestClient test cannot reach it either, since the bypass app.py already has
# for tool-carrying requests (Task 6's job to remove) intercepts before
# gather_panel is ever called.

@pytest.mark.anyio
async def test_a_three_way_split_lets_the_fuser_answer_with_a_call(tmp_path):
    def script(model_name, own_path):
        def fn(p):
            prompt = p["messages"][0]["content"]
            if "Produce the single best final answer" in prompt:
                # The fuser's own call: arbitrates with a call NONE of the
                # three candidates proposed.
                return tool_resp("read", '{"path":"fused-choice"}')
            return tool_resp("read", f'{{"path":"{own_path}"}}')
        return fn

    ad = FakeAdapter({"a": script("a", "a"), "b": script("b", "b"),
                      "s": script("s", "c")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "full"                     # a genuine three-way split

    common = {k: v for k, v in env.items() if k != "fcfg"}
    result = await call_model(model_name=FCFG.fuser,
                              body=fuser_body(FCFG, panel, TOOLS_BODY),
                              kind="fuser", **common)
    if result:
        source = "fuser"
    else:
        fallback = best_candidate(FCFG, panel)
        result, source = (fallback[1], "candidate") if fallback else (None, "none")

    assert source == "fuser"                         # not a disagreeing candidate
    assert result.tool_calls[0]["function"]["arguments"] == '{"path":"fused-choice"}'
    meta = {"path": panel.path, "answered_by": source}
    r = openai_response(result, FCFG.model, meta)
    msg = r["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"path":"fused-choice"}'
    assert r["fusion"]["answered_by"] == "fuser"

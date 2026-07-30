import pytest
from gateway.fusion import (
    Candidate, PanelResult, _extract_message, best_candidate, call_model,
    fuser_body, gather_panel, openai_response,
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


# -- M9 Task 5: the decision tree ------------------------------------------

TOOLS_BODY = dict(BODY, tools=[{"type": "function",
                                "function": {"name": "read", "parameters": {}}}])


def tool_resp(name, args):
    return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": name, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


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
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len([r for r in rows if r["model"] in ("a", "b")]) == 2
    assert not any(r["state"] == "preflight" for r in rows)


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
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_reviewed" and panel.reviews != {}


@pytest.mark.anyio
async def test_a_reviewer_objection_on_a_write_class_call_escalates(tmp_path):
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a wrong no\nVERDICT b wrong no"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py"}')
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path != "tool_reviewed"            # did not copy on objection
    # Deviation from the brief's literal test (see task-5-report.md): the
    # brief asserted `"s" in panel.candidates` here, but that can never hold
    # for THIS script. Entering the write-class agree_review branch at all
    # already requires a and b's calls to match; once "s" joins,
    # `plurality` always finds {a, b} as its >=2 group first (regardless of
    # what "s" itself proposes) and `tool_plurality` returns only that
    # winner -- "s" can never be a key in the panel's candidates on this
    # branch. What "escalates" actually means -- "s" was genuinely awaited,
    # not skipped by an incorrectly-taken `tool_reviewed` shortcut -- is
    # what the ledger proves instead: a cancelled leg always settles with
    # usage_source "estimated" (call_model's CancelledError handler); only
    # a leg that ran to completion settles "reported".
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger WHERE model='s'").fetchall()
    assert len(rows) == 1
    assert rows[0]["state"] == "settled" and rows[0]["usage_source"] == "reported"


@pytest.mark.anyio
async def test_two_of_three_plurality_wins_without_the_fuser(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"a"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_plurality"


@pytest.mark.anyio
async def test_a_three_way_split_falls_through_to_the_full_path(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"c"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "full" and len(panel.candidates) == 3


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
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "quorum"


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

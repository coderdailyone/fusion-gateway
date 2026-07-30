import pytest
from gateway.fusion import Candidate, _extract_message, openai_response


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

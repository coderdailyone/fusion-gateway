import json
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
    for resp in ({}, {"choices": []}, {"choices": [{}]}, {"choices": "x"},
                 _resp(content=None, tool_calls="notalist"),
                 _resp(content=None, tool_calls=[None])):
        got = _extract_message(resp)
        assert isinstance(got, Candidate)


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

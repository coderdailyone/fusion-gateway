import json

from gateway.anthropic_translate import DEFAULT_MAX_TOKENS, to_anthropic_request


def test_system_messages_are_hoisted_to_the_top_level_field():
    out = to_anthropic_request({
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "system", "content": "and polite"},
            {"role": "user", "content": "hi"},
        ]}, "glm-5.2")
    assert out["system"] == "be terse\n\nand polite"
    assert [m["role"] for m in out["messages"]] == ["user"]
    assert out["model"] == "glm-5.2"


def test_max_tokens_defaults_because_anthropic_requires_it():
    out = to_anthropic_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert out["max_tokens"] == DEFAULT_MAX_TOKENS
    given = to_anthropic_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}, "m")
    assert given["max_tokens"] == 16


def test_sampling_params_and_stop_are_mapped():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2, "top_p": 0.9, "stop": ["END"],
    }, "m")
    assert out["temperature"] == 0.2 and out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"]
    assert "stop" not in out


def test_tools_are_flattened_with_input_schema():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function",
                   "function": {"name": "search", "description": "find",
                                "parameters": schema}}],
        "tool_choice": "auto",
    }, "m")
    assert out["tools"] == [{"name": "search", "description": "find",
                             "input_schema": schema}]
    assert out["tool_choice"] == {"type": "auto"}


def test_named_tool_choice_is_mapped():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "search",
                                                    "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
    }, "m")
    assert out["tool_choice"] == {"type": "tool", "name": "search"}


def test_assistant_tool_calls_become_tool_use_blocks():
    out = to_anthropic_request({"messages": [
        {"role": "user", "content": "search cats"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": '{"q": "cats"}'}}]},
    ]}, "m")
    blocks = out["messages"][1]["content"]
    assert blocks == [{"type": "tool_use", "id": "call_1", "name": "search",
                       "input": {"q": "cats"}}]


def test_tool_result_messages_become_user_tool_result_blocks():
    out = to_anthropic_request({"messages": [
        {"role": "user", "content": "search cats"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found 3"},
    ]}, "m")
    last = out["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [{"type": "tool_result", "tool_use_id": "call_1",
                                "content": "found 3"}]


def test_gateway_only_fields_are_not_forwarded():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "stream_options": {"include_usage": True},
        "model": "whatever-the-client-asked-for",
    }, "upstream-name")
    assert out["model"] == "upstream-name"
    assert "stream" not in out and "stream_options" not in out

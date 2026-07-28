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


def test_array_content_system_message_is_hoisted_as_a_string():
    # OpenAI allows system content as a list of parts. The public endpoint does
    # no pydantic validation, so this shape reaches the translator verbatim.
    out = to_anthropic_request({
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "be terse"}]},
            {"role": "system", "content": "and polite"},
            {"role": "user", "content": "hi"},
        ]}, "m")
    assert out["system"] == "be terse\n\nand polite"
    assert [m["role"] for m in out["messages"]] == ["user"]


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


def test_required_tool_choice_maps_to_anthropic_any():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "search",
                                                    "parameters": {}}}],
        "tool_choice": "required",
    }, "m")
    assert out["tool_choice"] == {"type": "any"}


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


def test_malformed_tool_call_arguments_degrade_to_empty_input():
    # A truncated/invalid arguments string must not 500 the whole request.
    out = to_anthropic_request({"messages": [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": '{"q": "cat'}}]},
    ]}, "m")
    assert out["messages"][0]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {}}]


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


from gateway.anthropic_translate import from_anthropic_response


def _resp(content, stop_reason="end_turn", usage=None):
    return {"id": "msg_1", "type": "message", "role": "assistant",
            "model": "glm-5.2", "content": content, "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 11, "output_tokens": 7}}


def test_text_blocks_concatenate_into_message_content():
    out = from_anthropic_response(
        _resp([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]), "glm-5.2")
    assert out["object"] == "chat.completion"
    assert out["model"] == "glm-5.2"
    choice = out["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"


def test_usage_is_renamed_to_openai_fields():
    out = from_anthropic_response(_resp([{"type": "text", "text": "x"}]), "m")
    assert out["usage"] == {"prompt_tokens": 11, "completion_tokens": 7,
                            "total_tokens": 18}


def test_tool_use_blocks_become_tool_calls_with_json_arguments():
    out = from_anthropic_response(_resp(
        [{"type": "tool_use", "id": "toolu_9", "name": "search",
          "input": {"q": "cats"}}], stop_reason="tool_use"), "m")
    choice = out["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "toolu_9" and call["type"] == "function"
    assert call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"q": "cats"}
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None   # tool-only reply


def test_thinking_blocks_are_dropped_not_leaked_into_content():
    out = from_anthropic_response(_resp(
        [{"type": "thinking", "thinking": "secret chain"},
         {"type": "text", "text": "answer"}]), "m")
    assert out["choices"][0]["message"]["content"] == "answer"
    assert "secret chain" not in json.dumps(out)


def test_stop_reason_mapping_covers_every_value():
    for anthropic, openai in (("end_turn", "stop"), ("max_tokens", "length"),
                              ("stop_sequence", "stop"), ("tool_use", "tool_calls")):
        out = from_anthropic_response(
            _resp([{"type": "text", "text": "x"}], stop_reason=anthropic), "m")
        assert out["choices"][0]["finish_reason"] == openai


# --- untrusted-input hardening -------------------------------------------------
# Upstream replies are as untrusted as client bodies: nothing here is validated
# against a schema before it reaches the translator, so a surprising shape has to
# degrade into a valid completion rather than raise out of the request handler.


def test_unexpected_content_shapes_degrade_instead_of_raising():
    for content in (None, "just a string", ["oops", 7, None], {"type": "text"}):
        out = from_anthropic_response(_resp(content), "m")
        assert out["choices"][0]["message"]["content"] is None
        assert "tool_calls" not in out["choices"][0]["message"]


def test_non_string_text_and_non_object_tool_input_degrade():
    out = from_anthropic_response(_resp([
        {"type": "text", "text": None},
        {"type": "tool_use", "id": "t1", "name": "search", "input": "not-an-object"},
    ], stop_reason="tool_use"), "m")
    call = out["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["arguments"] == "{}"
    assert out["choices"][0]["message"]["content"] is None


def test_missing_or_garbage_usage_degrades_to_zero_tokens():
    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    no_usage = from_anthropic_response({"content": [{"type": "text", "text": "x"}]}, "m")
    assert no_usage["usage"] == zero
    garbage = from_anthropic_response(
        _resp([{"type": "text", "text": "x"}],
              usage={"input_tokens": "?", "output_tokens": None}), "m")
    assert garbage["usage"] == zero


def test_unknown_stop_reason_falls_back_to_stop():
    for bad in (None, "overloaded", ["end_turn"]):
        out = from_anthropic_response(
            _resp([{"type": "text", "text": "x"}], stop_reason=bad), "m")
        assert out["choices"][0]["finish_reason"] == "stop"

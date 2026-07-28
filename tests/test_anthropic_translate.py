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


def test_empty_text_block_before_a_tool_use_still_yields_content_none():
    # Regression guard for the `joined or None` formula. The obvious-looking
    # `"".join(text_parts) if text_parts else None` returns "" here, because an
    # empty text block makes text_parts truthy — and "" defeats the `content is
    # None` check OpenAI clients use to detect a tool call.
    out = from_anthropic_response(_resp(
        [{"type": "text", "text": ""},
         {"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        stop_reason="tool_use"), "m")
    message = out["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["id"] == "t1"


def test_unknown_stop_reason_falls_back_to_stop():
    for bad in (None, "overloaded", ["end_turn"]):
        out = from_anthropic_response(
            _resp([{"type": "text", "text": "x"}], stop_reason=bad), "m")
        assert out["choices"][0]["finish_reason"] == "stop"


from gateway.anthropic_translate import StreamTranslator, iter_sse_data


def _drain(translator, events):
    out = []
    for e in events:
        out.extend(translator.feed(e))
    out.extend(translator.finish())
    return out


def test_iter_sse_data_parses_data_lines_and_skips_junk():
    raw = (b'event: message_start\n'
           b'data: {"type": "message_start"}\n\n'
           b': a comment line\n'
           b'data: not-json\n\n'
           b'data: {"type": "message_stop"}\n\n')
    got = list(iter_sse_data(raw))
    assert [g["type"] for g in got] == ["message_start", "message_stop"]


def test_text_stream_translates_to_openai_chunks_with_role_then_content():
    t = StreamTranslator("glm-5.2")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "he"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "llo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c["choices"])
    assert text == "hello"
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if c["choices"] and c["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]


def test_usage_is_merged_from_message_start_and_message_delta():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[-1]["usage"] == {"prompt_tokens": 12,
                                         "completion_tokens": 5,
                                         "total_tokens": 17}


def test_tool_stream_emits_name_then_reassemblable_argument_fragments():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"q":'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": ' "cats"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ])
    # the opening chunk names the tool
    starts = [c for c in chunks if c["choices"]
              and c["choices"][0]["delta"].get("tool_calls")
              and c["choices"][0]["delta"]["tool_calls"][0]["function"].get("name")]
    assert starts[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1"
    # fragments concatenate into valid JSON, exactly as an OpenAI client does
    args = "".join(
        tc["function"].get("arguments", "")
        for c in chunks if c["choices"]
        for tc in c["choices"][0]["delta"].get("tool_calls", []))
    assert json.loads(args) == {"q": "cats"}
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if c["choices"] and c["choices"][0].get("finish_reason")]
    assert finishes == ["tool_calls"]


def test_thinking_deltas_are_dropped_and_counted():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "secret"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ])
    assert "secret" not in json.dumps(chunks)
    assert t.thinking_blocks == 1


def test_delta_for_an_unknown_block_index_is_ignored_not_fatal():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_delta", "index": 7,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "message_stop"},
    ])
    # No exception -- and, more to the point, no tool_calls chunk invented for a
    # block that never opened: a fragment guessed onto slot 0 would corrupt
    # whatever real tool call lands there.
    assert not [tc for c in chunks if c["choices"]
                for tc in c["choices"][0]["delta"].get("tool_calls", [])]


def test_emitted_stream_is_parseable_by_the_existing_usage_parser():
    from gateway.providers import parse_stream_usage

    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 21}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ])
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    wire += b"data: [DONE]\n\n"
    assert parse_stream_usage(wire) == {"prompt_tokens": 21,
                                        "completion_tokens": 4,
                                        "total_tokens": 25}


def test_malformed_stream_events_degrade_instead_of_raising():
    # Upstream SSE is no more trusted than a client body, and these are the same
    # shapes the non-streaming path already guards against -- only now they
    # arrive one event at a time, where a raise aborts a stream mid-flight and
    # loses the usage chunk the ledger settles on.
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": "abc"}}},
        {"type": "message_start", "message": {"usage": 7}},
        {"type": "message_start", "message": [1]},
        {"type": "content_block_start", "index": 0, "content_block": "x"},
        {"type": "content_block_start", "index": [],
         "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "index": 0, "delta": "oops"},
        {"type": "content_block_delta", "index": [],
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": {"not": "a string"}}},
        {"type": "message_delta", "delta": "x", "usage": {"output_tokens": "x"}},
        {"type": "message_delta", "delta": {"stop_reason": ["end_turn"]}},
        {},
    ])
    assert (t.input_tokens, t.output_tokens) == (0, 0)
    assert chunks[-1]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    # whatever survives stays string-shaped, so a client can still concatenate it
    for chunk in chunks:
        for choice in chunk["choices"]:
            for key in ("content", "role"):
                assert isinstance(choice["delta"].get(key, ""), str)
            for call in choice["delta"].get("tool_calls", []):
                for key in ("name", "arguments"):
                    assert isinstance(call["function"].get(key, ""), str)


def test_infinite_token_counts_degrade_to_zero_on_both_paths():
    # json.loads accepts 1e400 (and Infinity) and hands back float("inf"), which
    # is well-formed JSON as far as iter_sse_data is concerned -- so it reaches
    # the counter, where int(inf) raises OverflowError. That is a sibling of
    # ValueError, not a subclass, so an except tuple listing only TypeError and
    # ValueError misses it. Mid-stream that aborts the response and takes the
    # usage chunk, and therefore the bill, with it.
    wire = (b'data: {"type": "message_start", "message":'
            b' {"usage": {"input_tokens": 1e400}}}\n\n')
    event, = iter_sse_data(wire)
    assert event["message"]["usage"]["input_tokens"] == float("inf")

    t = StreamTranslator("m")
    chunks = _drain(t, [event, {"type": "message_delta",
                                "usage": {"output_tokens": float("-inf")}}])
    assert (t.input_tokens, t.output_tokens) == (0, 0)
    assert chunks[-1]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0}

    # Both directions share _token_count, so the parity has to stay pinned:
    # the same garbage must cost the same nothing on the non-streaming path.
    non_streaming = from_anthropic_response(
        {"content": [{"type": "text", "text": "x"}],
         "usage": {"input_tokens": 1e400, "output_tokens": float("nan")}}, "m")
    assert non_streaming["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                                      "total_tokens": 0}


def test_feed_ignores_an_event_that_is_not_a_dict():
    # iter_sse_data only yields dicts today, so this is the outermost shape hole
    # rather than a live crash -- but feed() is the untrusted-input boundary and
    # should not depend on its caller for that.
    t = StreamTranslator("m")
    for junk in ([], "x", None, 5, ["message_start"]):
        assert t.feed(junk) == []
    assert (t.input_tokens, t.output_tokens, t.thinking_blocks) == (0, 0, 0)


def test_two_tool_blocks_map_onto_dense_openai_indices_without_cross_talk():
    # Anthropic numbers content blocks across the whole message -- a text block
    # takes 0 here, and the tool blocks land on 1 and 3 -- while OpenAI numbers
    # tool_calls densely from 0. The fragments interleave, so it is the mapping,
    # not arrival order, that keeps each argument string whole.
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "sure"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_a", "name": "search"}},
        {"type": "content_block_start", "index": 3,
         "content_block": {"type": "tool_use", "id": "toolu_b", "name": "fetch"}},
        {"type": "content_block_delta", "index": 3,
         "delta": {"type": "input_json_delta", "partial_json": '{"url":'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": "cats"'}},
        {"type": "content_block_delta", "index": 3,
         "delta": {"type": "input_json_delta", "partial_json": ' "u"}'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_stop", "index": 3},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ])
    calls = [tc for c in chunks if c["choices"]
             for tc in c["choices"][0]["delta"].get("tool_calls", [])]
    openings = [tc for tc in calls if "id" in tc]
    fragments = [tc for tc in calls if "id" not in tc]

    # Anthropic 1 -> OpenAI 0, Anthropic 3 -> OpenAI 1, in block-open order.
    assert [(tc["index"], tc["id"], tc["function"]["name"]) for tc in openings] \
        == [(0, "toolu_a", "search"), (1, "toolu_b", "fetch")]
    # A client reads name/id once; repeating them on a fragment corrupts the call.
    assert fragments and all("name" not in tc["function"] for tc in fragments)

    reassembled: dict[int, str] = {}
    for tc in calls:
        reassembled[tc["index"]] = (reassembled.get(tc["index"], "")
                                    + tc["function"].get("arguments", ""))
    assert json.loads(reassembled[0]) == {"q": "cats"}
    assert json.loads(reassembled[1]) == {"url": "u"}

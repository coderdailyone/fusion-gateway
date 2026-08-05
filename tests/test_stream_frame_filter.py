"""Pure tests for the SSE tail filter that separates the fuser's own usage
frame from the gateway's panel total."""
from gateway.app import _is_gateway_owned, _split_client_frames, _wants_usage


def test_done_is_owned_by_the_gateway():
    assert _is_gateway_owned(b"data: [DONE]")
    assert _is_gateway_owned(b"data:[DONE]  ")


def test_usage_only_frame_is_owned_by_the_gateway():
    assert _is_gateway_owned(
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}')


def test_a_content_frame_is_forwarded():
    assert not _is_gateway_owned(
        b'data: {"choices":[{"delta":{"content":"hi"}}]}')


def test_a_frame_carrying_content_AND_usage_is_forwarded():
    """Dropping it would lose tokens the client is owed. The gateway's own
    later frame still wins under parse_stream_usage's last-one-wins rule."""
    assert not _is_gateway_owned(
        b'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":1}}')


def test_non_data_lines_and_garbage_pass_through():
    assert not _is_gateway_owned(b": keepalive comment")
    assert not _is_gateway_owned(b"")
    assert not _is_gateway_owned(b"data: {not json")


def test_a_frame_split_across_two_reads_is_reassembled_not_misparsed():
    """The socket can cut a line anywhere. Classifying half a usage frame as
    content is exactly the failure this buffer prevents."""
    whole = b'data: {"choices":[],"usage":{"prompt_tokens":3}}\n\n'
    first, second = whole[:20], whole[20:]
    out1, buf1 = _split_client_frames(b"", first)
    assert out1 == b"" and buf1 == first          # nothing complete yet
    out2, buf2 = _split_client_frames(buf1, second)
    assert out2 == b"\n"                          # the frame dropped, blank line kept
    assert buf2 == b""


def test_content_survives_a_split_read():
    whole = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    out1, buf1 = _split_client_frames(b"", whole[:15])
    out2, buf2 = _split_client_frames(buf1, whole[15:])
    assert (out1 + out2).startswith(b'data: {"choices"')
    assert b'"content":"hi"' in out1 + out2


def test_wants_usage_only_when_the_client_asks():
    assert _wants_usage({"stream_options": {"include_usage": True}})
    assert not _wants_usage({"stream_options": {"include_usage": False}})
    assert not _wants_usage({"stream_options": {}})
    assert not _wants_usage({"stream_options": "nonsense"})
    assert not _wants_usage({})

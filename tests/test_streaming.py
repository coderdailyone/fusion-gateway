import json
import httpx, pytest
from tests.test_app import make_client, H
from tests.helpers import DelayedByteStream

SSE = (b'data: {"choices":[{"delta":{"content":"h"}}]}\n\n'
       b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
       b'data: [DONE]\n\n')

def stream_handler(req):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)

BODY = {"model": "deepseek-chat", "stream": True, "messages": [{"role":"user","content":"hi"}]}

def test_stream_passthrough_and_settle(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, deepseek=stream_handler)
    with c.stream("POST", "/v1/chat/completions", json=BODY, headers=H()) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_raw())
    assert b"[DONE]" in raw
    st = c.get("/admin/status", headers=H("tokB")).json()
    assert st["ledger"]["consumed_usd"] > 0

def test_prefirstbyte_failure_falls_back(tmp_path, monkeypatch):
    def boom(req): return httpx.Response(500, json={"e": 1})
    c = make_client(tmp_path, monkeypatch, deepseek=boom, glm=stream_handler)
    with c.stream("POST", "/v1/chat/completions", json=BODY, headers=H()) as r:
        raw = b"".join(r.iter_raw())
    assert b"[DONE]" in raw


class _BrokenMidStream(httpx.AsyncByteStream):
    """Yields one truncated raw chunk (no line terminator, mid-token, exactly
    the shape a real dropped TCP connection leaves behind) then blows up."""
    def __init__(self, first_chunk: bytes):
        self.first_chunk = first_chunk

    async def __aiter__(self):
        yield self.first_chunk
        raise RuntimeError("upstream connection dropped")

    async def aclose(self) -> None:
        pass


def test_mid_stream_error_envelope_does_not_glue_onto_a_truncated_chunk(tmp_path, monkeypatch):
    """Final whole-branch review, finding 2 (CRITICAL), single-model path.
    Inherited by the fusion path too, but this pins the ORIGINAL site: when
    the stream dies after the first byte, app.py used to yield the error
    envelope with no separator, concatenating it directly onto whatever raw
    (possibly mid-token) bytes were already forwarded. Fed through the real
    openai SDK's SSEDecoder that is a JSONDecodeError, not a clean recognized
    API error, because the two `data:` payloads glue into one malformed
    line. Reproduced end to end: capture the exact bytes this app produces,
    then decode them with the real SDK.
    """
    def boom(req):
        # A COMPLETE, valid data line -- the connection drops before the
        # blank-line terminator that would normally follow it, which is
        # exactly the gap the \n\n-prefix fix is meant to close. (A chunk cut
        # mid-token, as the finding's own repro shows, produces unparseable
        # JSON on its own no matter what follows it -- that is not this
        # bug, and no separator fixes it. This isolates the actual bug: the
        # concatenation, not the truncation.)
        truncated = b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"wor"},"finish_reason":null}]}'
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_BrokenMidStream(truncated))

    c = make_client(tmp_path, monkeypatch, deepseek=boom)
    with c.stream("POST", "/v1/chat/completions", json=BODY, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_raw())
    assert b"stream_failed" in raw

    from openai import OpenAI
    client = OpenAI(api_key="x", base_url="http://test",
                    http_client=httpx.Client(transport=httpx.MockTransport(
                        lambda req: httpx.Response(200, content=raw,
                            headers={"content-type": "text/event-stream"}))))
    stream = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": "hi"}], stream=True)
    try:
        list(stream)
    except Exception as e:
        assert not isinstance(e, json.JSONDecodeError), (
            f"the error envelope glued onto a truncated upstream line and "
            f"broke the real SDK's decoder: {e!r}\nraw={raw!r}")

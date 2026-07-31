"""Provider adapter for upstreams that speak the Anthropic messages protocol.

Exposes the SAME interface as gateway.providers.ProviderAdapter — chat() and
chat_stream() — so gateway/app.py is agnostic to which wire a provider uses.
All shape translation is delegated to gateway.anthropic_translate (pure).

The ProviderError timing contract is preserved: it is raised only BEFORE the
first byte reaches the caller, which is what makes the fallback chain safe.
"""
from __future__ import annotations

import json
import os
import re
from typing import AsyncIterator

import httpx

from gateway.anthropic_translate import (ANTHROPIC_VERSION, StreamTranslator,
                                         from_anthropic_response,
                                         iter_sse_data, to_anthropic_request)
from gateway.config import ProviderCfg
from gateway.providers import ProviderError


# A blank line ends an SSE event. WHATWG allows either terminator, and an
# upstream that frames with CRLF contains no b"\n\n" at all — matching only LF
# would buffer its entire body in RAM and deliver nothing until EOF, which an
# SSE client with an idle timeout sees as a hang. A newline never appears raw
# inside a `data:` payload (JSON escapes it), so this cannot split an event.
_EVENT_END = re.compile(rb"\r?\n\r?\n")


async def _iter_event_blocks(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Re-frame a byte stream into whole SSE events.

    resp.aiter_bytes() cuts at arbitrary socket boundaries, and iter_sse_data()
    does no line buffering: a `data:` line split across two reads is dropped
    silently. When the dropped line is `message_delta`, output_tokens stays 0
    and the ledger bills zero completion tokens — so the framing has to happen
    here. Events are separated by a blank line; a partial trailing event stays
    in the buffer until it completes.
    """
    buffer = b""
    async for raw in resp.aiter_bytes():
        buffer += raw
        while True:
            match = _EVENT_END.search(buffer)
            if match is None:
                break
            block, buffer = buffer[:match.start()], buffer[match.end():]
            yield block + b"\n\n"
    if buffer.strip():
        # Body ended without a terminating blank line. Emit the remainder
        # anyway: dropping it would lose a final message_delta (and its
        # output_tokens). A genuinely truncated event just fails to parse,
        # which iter_sse_data skips.
        yield buffer + b"\n\n"


class AnthropicAdapter:
    """Talks to an Anthropic-protocol upstream over httpx.

    Mirrors ProviderAdapter's contract, including no same-provider auto-retry:
    a failed call raises ProviderError and the caller's fallback chain decides.
    """

    def __init__(self, cfg: ProviderCfg, timeout_s: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)

    def _url(self) -> str:
        return f"{self.cfg.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": os.environ[self.cfg.api_key_env],
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json"}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, upstream_model: str, payload: dict) -> dict:
        body = to_anthropic_request(payload, upstream_model)
        try:
            resp = await self._client.post(self._url(), json=body,
                                           headers=self._headers())
        except httpx.TimeoutException as e:
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderError(self.cfg.name, "network") from e
        if not (200 <= resp.status_code < 300):
            raise ProviderError(self.cfg.name, "http", status=resp.status_code)
        try:
            data = resp.json()
        except ValueError as e:
            # A 200 whose body is not JSON (proxy error page, truncated read).
            raise ProviderError(self.cfg.name, "http",
                                status=resp.status_code) from e
        if not isinstance(data, dict):
            # from_anthropic_response takes a dict on purpose; a JSON array or
            # scalar here would silently become an empty completion with zero
            # usage, which the ledger would then bill as a successful call.
            raise ProviderError(self.cfg.name, "http", status=resp.status_code)
        if "error" in data or not isinstance(data.get("content"), list):
            # A 2xx is not proof of a message. configs/gateway.toml records that
            # this endpoint answers a balance failure with HTTP 200 and an error
            # object, and a messages reply always carries a `content` array.
            # Translating either shape yields a hollow success: the client gets
            # content: null and the ledger settles the call, so the fallback
            # chain never engages. `content: []` is a real, if unusual, message
            # and stays accepted -- the guard keys on shape, not emptiness.
            raise ProviderError(self.cfg.name, "http", status=resp.status_code)
        return from_anthropic_response(data, upstream_model)

    async def chat_stream(self, upstream_model: str,
                          payload: dict) -> AsyncIterator[bytes]:
        body = to_anthropic_request(payload, upstream_model)
        body["stream"] = True
        translator = StreamTranslator(upstream_model)

        def _wire(chunk: dict) -> bytes:
            return b"data: " + json.dumps(chunk).encode() + b"\n\n"

        yielded = False
        saw_event = False
        try:
            async with self._client.stream("POST", self._url(), json=body,
                                           headers=self._headers()) as resp:
                if not (200 <= resp.status_code < 300):
                    await resp.aread()
                    raise ProviderError(self.cfg.name, "http",
                                        status=resp.status_code)
                async for block in _iter_event_blocks(resp):
                    for event in iter_sse_data(block):
                        saw_event = True
                        if event.get("type") == "error":
                            if not yielded:
                                raise ProviderError(self.cfg.name, "http")
                            # Bytes are already out, so falling back is unsafe.
                            # Tell the client what happened and close the
                            # protocol properly: a bare return would leave it
                            # with no finish_reason and no [DONE], which is
                            # indistinguishable from a dropped connection.
                            # Every prior yield here is a complete `_wire()`
                            # frame (already terminated with its own \n\n), so
                            # this is not vulnerable to app.py's finding-2
                            # concatenation bug -- but a leading blank line is
                            # a no-op for a conformant parser, so there is no
                            # reason for this call site to be the one
                            # exception to "every mid-stream error envelope in
                            # this codebase is self-terminating on its own."
                            yield b'\n\ndata: {"error": {"type": "upstream_error"}}\n\n'
                            for chunk in translator.finish():
                                yield _wire(chunk)
                            yield b"data: [DONE]\n\n"
                            return
                        for chunk in translator.feed(event):
                            yielded = True
                            yield _wire(chunk)
                if not saw_event:
                    # The body ended without one parseable Anthropic event: an
                    # empty 200, or a non-SSE error page. finish() always emits
                    # at least a finish_reason chunk, so continuing here would
                    # fabricate a well-formed EMPTY answer — billed by estimate,
                    # logged as call.succeeded, with no fallback — and app.py's
                    # empty_stream guard can never fire on this wire because a
                    # zero-byte stream is impossible. Nothing has been yielded
                    # at this point (no event fed the translator), so the
                    # ProviderError-before-the-first-byte contract holds.
                    raise ProviderError(self.cfg.name, "http",
                                        status=resp.status_code)
                # finish() is NOT idempotent — exactly one call on each exit
                # path, and every chunk it returns is emitted. The usage chunk
                # it may append has an EMPTY choices list; filtering that out
                # as "no content" is what makes a stream bill zero.
                for chunk in translator.finish():
                    yielded = True
                    yield _wire(chunk)
                yield b"data: [DONE]\n\n"
        except httpx.TimeoutException as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "network") from e

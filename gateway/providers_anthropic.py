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
from typing import AsyncIterator

import httpx

from gateway.anthropic_translate import (ANTHROPIC_VERSION, StreamTranslator,
                                         from_anthropic_response,
                                         iter_sse_data, to_anthropic_request)
from gateway.config import ProviderCfg
from gateway.providers import ProviderError


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
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
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
        return from_anthropic_response(data, upstream_model)

    async def chat_stream(self, upstream_model: str,
                          payload: dict) -> AsyncIterator[bytes]:
        body = to_anthropic_request(payload, upstream_model)
        body["stream"] = True
        translator = StreamTranslator(upstream_model)

        def _wire(chunk: dict) -> bytes:
            return b"data: " + json.dumps(chunk).encode() + b"\n\n"

        yielded = False
        try:
            async with self._client.stream("POST", self._url(), json=body,
                                           headers=self._headers()) as resp:
                if not (200 <= resp.status_code < 300):
                    await resp.aread()
                    raise ProviderError(self.cfg.name, "http",
                                        status=resp.status_code)
                async for block in _iter_event_blocks(resp):
                    for event in iter_sse_data(block):
                        if event.get("type") == "error":
                            if not yielded:
                                raise ProviderError(self.cfg.name, "http")
                            return
                        for chunk in translator.feed(event):
                            yielded = True
                            yield _wire(chunk)
                # finish() is NOT idempotent — exactly one call, and every
                # chunk it returns is emitted. The last one carries usage in a
                # chunk whose choices list is empty; filtering that out as
                # "no content" is what makes a stream bill zero.
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

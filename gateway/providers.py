from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

import httpx

from gateway.config import ProviderCfg


class ProviderError(Exception):
    """Raised when an upstream provider call fails.

    kind is one of 'timeout', 'network', 'http'. status is set only for
    'http' (the upstream's non-2xx status code).
    """

    def __init__(self, provider: str, kind: str, status: Optional[int] = None):
        self.provider = provider
        self.kind = kind
        self.status = status
        detail = f" status={status}" if status is not None else ""
        super().__init__(f"provider={provider} kind={kind}{detail}")


class ProviderAdapter:
    """Talks to an OpenAI-compatible upstream API over httpx.

    A single httpx.AsyncClient is created per adapter instance. No
    same-provider auto-retry is performed here: a failed call raises
    ProviderError and the caller's fallback chain decides what to do next.
    """

    def __init__(
        self,
        cfg: ProviderCfg,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.cfg = cfg
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)

    def _url(self) -> str:
        return f"{self.cfg.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        api_key = os.environ[self.cfg.api_key_env]
        return {"Authorization": f"Bearer {api_key}"}

    async def chat(self, upstream_model: str, payload: dict) -> dict:
        body = dict(payload)
        body["model"] = upstream_model
        body["stream"] = False
        try:
            resp = await self._client.post(self._url(), json=body, headers=self._headers())
        except httpx.TimeoutException as e:
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderError(self.cfg.name, "network") from e
        if not (200 <= resp.status_code < 300):
            raise ProviderError(self.cfg.name, "http", status=resp.status_code)
        return resp.json()

    async def chat_stream(self, upstream_model: str, payload: dict) -> AsyncIterator[bytes]:
        body = dict(payload)
        body["model"] = upstream_model
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        yielded = False
        try:
            async with self._client.stream(
                "POST", self._url(), json=body, headers=self._headers()
            ) as resp:
                if not (200 <= resp.status_code < 300):
                    await resp.aread()
                    raise ProviderError(self.cfg.name, "http", status=resp.status_code)
                async for chunk in resp.aiter_bytes():
                    yielded = True
                    yield chunk
        except httpx.TimeoutException as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "network") from e


def parse_stream_usage(collected: bytes) -> dict | None:
    """Scan SSE `data:` lines and return the `usage` object from the last
    chunk that has one, or None if no chunk carried usage."""
    usage: dict | None = None
    text = collected.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]" or not data:
            continue
        try:
            obj: Any = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            usage = obj["usage"]
    return usage

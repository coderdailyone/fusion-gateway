"""Anthropic-wire provider adapter (filled in by Task 5)."""
from __future__ import annotations

import httpx

from gateway.config import ProviderCfg


class AnthropicAdapter:
    def __init__(self, cfg: ProviderCfg, timeout_s: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)

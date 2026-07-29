import asyncio
from datetime import datetime, timedelta, timezone

import httpx

class FakeClock:
    def __init__(self, start="2026-07-14T00:00:00+00:00"):
        self._t = datetime.fromisoformat(start)
    def now(self):
        return self._t
    def advance(self, seconds: float):
        self._t += timedelta(seconds=seconds)


class DelayedByteStream(httpx.AsyncByteStream):
    """An httpx streaming response body that yields `chunks` with a real
    `asyncio.sleep(delay)` between each -- for tests that need to assert
    real-time deadline behaviour (e.g. an idle-vs-wall-clock timeout) against
    a genuinely slow, real-time upstream, not a synchronously-returned blob.
    """
    def __init__(self, chunks, delay: float):
        self.chunks = chunks
        self.delay = delay

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        pass

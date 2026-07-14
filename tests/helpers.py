from datetime import datetime, timedelta, timezone

class FakeClock:
    def __init__(self, start="2026-07-14T00:00:00+00:00"):
        self._t = datetime.fromisoformat(start)
    def now(self):
        return self._t
    def advance(self, seconds: float):
        self._t += timedelta(seconds=seconds)

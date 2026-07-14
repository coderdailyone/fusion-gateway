from __future__ import annotations
import json
from typing import NamedTuple

from gateway.clock import Clock
from gateway.db import Store


class Event(NamedTuple):
    seq: int
    request_id: str
    parent_seq: int | None
    created_at: str
    kind: str
    payload: dict


class EventLog:
    def __init__(self, store: Store, clock: Clock):
        self.store = store
        self.clock = clock

    def append(
        self,
        request_id: str,
        kind: str,
        payload: dict,
        parent_seq: int | None = None,
    ) -> int:
        with self.store.lock:
            cursor = self.store.conn.execute(
                "INSERT INTO events (request_id, parent_seq, created_at, kind, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    parent_seq,
                    self.clock.now().isoformat(),
                    kind,
                    json.dumps(payload),
                ),
            )
            self.store.conn.commit()
            return cursor.lastrowid

    def trace(self, request_id: str) -> list[Event]:
        rows = self.store.conn.execute(
            "SELECT seq, request_id, parent_seq, created_at, kind, payload "
            "FROM events WHERE request_id = ? ORDER BY seq",
            (request_id,),
        ).fetchall()
        return [
            Event(
                seq=row["seq"],
                request_id=row["request_id"],
                parent_seq=row["parent_seq"],
                created_at=row["created_at"],
                kind=row["kind"],
                payload=json.loads(row["payload"]),
            )
            for row in rows
        ]


def replay_actions(events: list[Event]) -> list[str]:
    return [f"{e.kind}:{e.payload.get('model', '-')}" for e in events]

from __future__ import annotations

from gateway.clock import Clock
from gateway.db import Store

CONSUMING_STATES = ("preflight", "settled", "orphaned")
ALERT_THRESHOLD = 0.8


class BudgetTripped(Exception):
    pass


def estimate_tokens(messages: list[dict], max_tokens: int | None) -> tuple[int, int]:
    total_chars = sum(len(m.get("content") or "") for m in messages)
    in_tokens = total_chars // 4
    out_tokens = max_tokens if max_tokens is not None else 1024
    return in_tokens, out_tokens


class Ledger:
    def __init__(
        self,
        store: Store,
        clock: Clock,
        cap_usd: float,
        budget_name: str,
        alert_cb=None,
    ):
        self.store = store
        self.clock = clock
        self.budget_name = budget_name
        self.alert_cb = alert_cb

        with self.store.lock:
            row = self.store.conn.execute(
                "SELECT name FROM budgets WHERE name = ?", (budget_name,)
            ).fetchone()
            if row is None:
                self.store.conn.execute(
                    "INSERT INTO budgets (name, cap_usd, state, updated_at) "
                    "VALUES (?, ?, 'active', ?)",
                    (budget_name, cap_usd, self.clock.now().isoformat()),
                )
                self.store.conn.commit()

        # cap_usd/state always sourced from the budgets row (single source of
        # truth); this mirrors what was upserted (or already present) above.
        self._alerted = self.consumed() >= ALERT_THRESHOLD * self.cap_usd

    # -- internal helpers (assume caller already holds store.lock) ----------

    def _consumed_locked(self) -> float:
        row = self.store.conn.execute(
            "SELECT SUM(COALESCE(actual_cost_usd, est_cost_usd)) AS c FROM ledger "
            "WHERE state IN (?, ?, ?)",
            CONSUMING_STATES,
        ).fetchone()
        return row["c"] or 0.0

    def _budget_row_locked(self):
        return self.store.conn.execute(
            "SELECT cap_usd, state FROM budgets WHERE name = ?", (self.budget_name,)
        ).fetchone()

    # -- public API -----------------------------------------------------

    @property
    def cap_usd(self) -> float:
        with self.store.lock:
            return self._budget_row_locked()["cap_usd"]

    def consumed(self) -> float:
        with self.store.lock:
            return self._consumed_locked()

    def status(self) -> dict:
        with self.store.lock:
            row = self._budget_row_locked()
            consumed = self._consumed_locked()
        return {
            "budget": self.budget_name,
            "cap_usd": row["cap_usd"],
            "consumed_usd": consumed,
            "state": row["state"],
        }

    def preflight(
        self,
        request_id: str,
        provider: str,
        model: str,
        est_in: int,
        est_out: int,
        in_price: float,
        out_price: float,
    ) -> int:
        cost = est_in * in_price / 1e6 + est_out * out_price / 1e6

        with self.store.lock:
            budget = self._budget_row_locked()
            consumed = self._consumed_locked()
            if budget["state"] == "tripped" or consumed + cost > budget["cap_usd"]:
                self.store.conn.execute(
                    "UPDATE budgets SET state='tripped', updated_at=? WHERE name=?",
                    (self.clock.now().isoformat(), self.budget_name),
                )
                self.store.conn.commit()
                raise BudgetTripped(
                    f"budget '{self.budget_name}' tripped "
                    f"(consumed={consumed} + est={cost} vs cap={budget['cap_usd']})"
                )

            cursor = self.store.conn.execute(
                "INSERT INTO ledger "
                "(request_id, provider, model, state, est_cost_usd, created_at) "
                "VALUES (?, ?, ?, 'preflight', ?, ?)",
                (request_id, provider, model, cost, self.clock.now().isoformat()),
            )
            self.store.conn.commit()
            entry_id = cursor.lastrowid
            consumed_now = self._consumed_locked()
            cap = budget["cap_usd"]

        if not self._alerted and consumed_now >= ALERT_THRESHOLD * cap:
            self._alerted = True
            if self.alert_cb is not None:
                self.alert_cb(consumed_now, cap)

        return entry_id

    def settle(
        self,
        entry_id: int,
        in_tokens: int,
        out_tokens: int,
        usage_source: str,
        latency_ms: int,
        in_price: float,
        out_price: float,
    ) -> float:
        actual_cost = in_tokens * in_price / 1e6 + out_tokens * out_price / 1e6

        with self.store.lock:
            row = self.store.conn.execute(
                "SELECT est_cost_usd FROM ledger WHERE id = ?", (entry_id,)
            ).fetchone()
            est_cost = row["est_cost_usd"]
            self.store.conn.execute(
                "UPDATE ledger SET state='settled', actual_cost_usd=?, in_tokens=?, "
                "out_tokens=?, usage_source=?, latency_ms=?, settled_at=? WHERE id=?",
                (
                    actual_cost,
                    in_tokens,
                    out_tokens,
                    usage_source,
                    latency_ms,
                    self.clock.now().isoformat(),
                    entry_id,
                ),
            )
            self.store.conn.commit()

        return abs(actual_cost - est_cost) / max(est_cost, 1e-9)

    def fail(self, entry_id: int) -> None:
        with self.store.lock:
            self.store.conn.execute(
                "UPDATE ledger SET state='failed' WHERE id=?", (entry_id,)
            )
            self.store.conn.commit()

    def release(self) -> None:
        with self.store.lock:
            self.store.conn.execute(
                "UPDATE budgets SET state='active', updated_at=? WHERE name=?",
                (self.clock.now().isoformat(), self.budget_name),
            )
            self.store.conn.commit()

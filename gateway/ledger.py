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
        cap_usd: float | None,
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
            else:
                # The config is authoritative for the CAP; the row stays the
                # runtime source of truth. Without this reconcile the row was
                # written once and never revisited, so editing cap_usd in
                # gateway.toml silently did nothing on any host that had
                # already run -- an operator raising the cap would go on
                # tripping at the old one. `state` is deliberately NOT synced:
                # only an explicit release may clear a trip, never a restart.
                self.store.conn.execute(
                    "UPDATE budgets SET cap_usd=?, updated_at=? WHERE name=?",
                    (cap_usd, self.clock.now().isoformat(), budget_name),
                )
            self.store.conn.commit()

        # cap_usd/state always sourced from the budgets row (single source of
        # truth); this mirrors what was upserted (or already present) above.
        cap = self.cap_usd
        self._alerted = cap is not None and self.consumed() >= ALERT_THRESHOLD * cap

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

    def usage_for_request(self, request_id: str) -> tuple[int, int]:
        """(in_tokens, out_tokens) billed to one request, over ALL its calls.

        A fusion request fans out to 3-8 upstream calls -- candidates, reviews,
        the fuser, and any fallback. None of them individually is "the" usage,
        and reporting the final leg alone would understate what the request
        cost by several times. Summing the ledger is the only figure that
        cannot drift from what was actually spent, because the ledger is the
        thing that spends it.

        Counts the same states the budget counts (CONSUMING_STATES). That
        deliberately includes a call cancelled mid-flight, which settles with
        estimated tokens rather than failing: the upstream did the work and
        may bill for it, so hiding it from the client would misreport the
        cost downward. Rows in 'failed' contribute nothing -- that state is
        for a call that never reached an upstream at all.

        COALESCE, not a bare SUM: a 'preflight' row has NULL token columns
        until it settles, and one NULL would otherwise poison the whole sum.
        """
        placeholders = ", ".join("?" * len(CONSUMING_STATES))
        with self.store.lock:
            row = self.store.conn.execute(
                "SELECT COALESCE(SUM(in_tokens), 0) AS in_tok, "
                "       COALESCE(SUM(out_tokens), 0) AS out_tok "
                f"FROM ledger WHERE request_id = ? AND state IN ({placeholders})",
                (request_id, *CONSUMING_STATES),
            ).fetchone()
        return int(row["in_tok"]), int(row["out_tok"])

    def observed_out_rate(self, model: str, limit: int = 50) -> float | None:
        """Output tokens per second this model has actually produced, or None.

        Used to price a call that was cancelled in flight. The alternative --
        charging the preflight estimate, which is `max_tokens` verbatim -- is
        not a neutral guess but a near-certain over-estimate: it assumes a leg
        we killed for being slow nonetheless ran all the way to the cap. On a
        live quorum request that made the cancelled kimi leg 93% of the whole
        request's cost, and it scaled with the CLIENT's max_tokens rather than
        with anything the model did.

        Only `reported` rows count. An estimated row is this function's own
        output, and feeding it back in would let the estimator drift away from
        the measurements it is supposed to be grounded in -- each cancellation
        teaching the next one a number no upstream ever confirmed.

        Median, not mean: one pathological call (a reasoning model that
        thought for 40s and emitted 12 tokens) should not move the estimate
        for every cancellation after it.
        """
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT out_tokens, latency_ms FROM ledger "
                "WHERE model = ? AND usage_source = 'reported' "
                "  AND latency_ms > 0 AND out_tokens IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (model, limit),
            ).fetchall()
        rates = sorted(r["out_tokens"] / (r["latency_ms"] / 1000.0) for r in rows)
        if not rates:
            return None
        mid = len(rates) // 2
        return rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2

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
            # cap_usd NULL means "no ceiling": the automatic trip is off, but a
            # budget tripped by hand still blocks -- an unbounded budget must
            # not be an unstoppable one.
            cap = budget["cap_usd"]
            over_cap = cap is not None and consumed + cost > cap
            if budget["state"] == "tripped" or over_cap:
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

        if cap is not None and not self._alerted and consumed_now >= ALERT_THRESHOLD * cap:
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

    def trip(self) -> None:
        """Stop spending now. With no cap configured this is the only brake."""
        with self.store.lock:
            self.store.conn.execute(
                "UPDATE budgets SET state='tripped', updated_at=? WHERE name=?",
                (self.clock.now().isoformat(), self.budget_name),
            )
            self.store.conn.commit()

    def release(self) -> None:
        with self.store.lock:
            self.store.conn.execute(
                "UPDATE budgets SET state='active', updated_at=? WHERE name=?",
                (self.clock.now().isoformat(), self.budget_name),
            )
            self.store.conn.commit()

# M1 Acceptance — Minimal Production Gateway

M1 is accepted when the gateway has carried **real traffic for one week with no
gateway-caused incident**, and the ledger reconciles within tolerance.

## What "an incident" means (falsifiable)

An incident is any of:

1. **Gateway 5xx not caused by upstream** — a 5xx the gateway itself produced
   (an unhandled exception, a crash), as opposed to a clean `502 upstream_exhausted`
   or `503 budget_exhausted`, which are *expected* responses, not incidents.
2. **Ledger drift over tolerance** — any settled ledger row whose
   `|actual − estimate| / estimate` exceeds **20%** and was not explained
   (this is the single reconciliation tolerance used across the system).
3. **Unexpected budget trip** — the kill switch tripping when spend did not
   actually reach the cap (a false positive), or failing to trip when it did.
4. **systemd restart loop** — the service restarting repeatedly
   (`systemctl status fusion-gateway` shows more than a couple of restarts/day).

Expected, non-incident events: upstream provider outages surfaced as
`502 upstream_exhausted`, deliberate `503 budget_exhausted`, single restarts.

## Daily check (operator)

Run the rollup and eyeball it:

```bash
ssh vps '/opt/fusion-gateway/.venv/bin/python /opt/fusion-gateway/scripts/rollup.py /opt/fusion-gateway/data/gateway.sqlite'
```

Record a one-line log entry per day below. If any incident class above fires,
note it, its cause, and the fix.

| Day | Requests | Cost (USD) | P95 latency | Fallbacks | Incidents |
|----:|---------:|-----------:|------------:|----------:|-----------|
| 1   |          |            |             |           |           |
| 2   |          |            |             |           |           |
| 3   |          |            |             |           |           |
| 4   |          |            |             |           |           |
| 5   |          |            |             |           |           |
| 6   |          |            |             |           |           |
| 7   |          |            |             |           |           |

## Exit review

After 7 clean days: M1 is done. If an incident fired, fix it, reset the clock,
and restart the week. Budget consumed over the window must stay under the M1 cap
and reconcile against the provider's own billing.

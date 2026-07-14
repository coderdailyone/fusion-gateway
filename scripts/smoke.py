#!/usr/bin/env python3
"""Smoke test against a *running* gateway: one minimal real request per
configured model.

Usage:
    GATEWAY_URL=http://127.0.0.1:8000 GATEWAY_TOKEN=<bearer-token> \\
        python scripts/smoke.py

For each model returned by GET {GATEWAY_URL}/v1/models, sends exactly one
POST /v1/chat/completions with a minimal, cheap body
(max_tokens=16, "Reply with exactly: ok"). Prints, per model: returned
model, latency, HTTP status. Reads GET /admin/status before and after the
run to print the ledger consumed_usd delta (requires GATEWAY_TOKEN to map
to the 'admin' principal; if it doesn't, the delta is skipped with a
warning rather than failing the run).

Exits non-zero if any model's request fails.

This script performs real, billed API calls when pointed at a live
gateway backed by real provider keys -- it is not exercised by the test
suite.
"""
from __future__ import annotations

import os
import sys
import time

import httpx

PROMPT = "Reply with exactly: ok"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def fetch_models(client: httpx.Client, base_url: str, token: str) -> list[str]:
    resp = client.get(f"{base_url}/v1/models", headers=_headers(token))
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def fetch_admin_status(client: httpx.Client, base_url: str, token: str) -> dict | None:
    """Returns the /admin/status payload, or None if unavailable (e.g. the
    token isn't admin-scoped)."""
    resp = client.get(f"{base_url}/admin/status", headers=_headers(token))
    if resp.status_code != 200:
        return None
    return resp.json()


def send_one(client: httpx.Client, base_url: str, token: str, model: str) -> tuple[bool, dict]:
    body = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    start = time.monotonic()
    try:
        resp = client.post(
            f"{base_url}/v1/chat/completions", json=body, headers=_headers(token)
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return False, {
            "model": model,
            "latency_ms": latency_ms,
            "status": None,
            "returned_model": None,
            "error": str(exc),
        }

    latency_ms = (time.monotonic() - start) * 1000
    ok = resp.status_code == 200
    returned_model = None
    error = None
    if ok:
        try:
            returned_model = resp.json().get("model")
        except ValueError:
            ok = False
            error = "response body was not valid JSON"
    else:
        error = resp.text[:200]

    return ok, {
        "model": model,
        "latency_ms": latency_ms,
        "status": resp.status_code,
        "returned_model": returned_model,
        "error": error,
    }


def run(base_url: str, token: str) -> int:
    failures = 0
    with httpx.Client(timeout=60.0) as client:
        try:
            models = fetch_models(client, base_url, token)
        except httpx.HTTPError as exc:
            print(f"failed to fetch /v1/models: {exc}", file=sys.stderr)
            return 1

        if not models:
            print("no models returned by /v1/models", file=sys.stderr)
            return 1

        before = fetch_admin_status(client, base_url, token)

        print(f"{'model':<20} {'returned':<20} {'status':<8} {'latency_ms':>10}")
        for model in models:
            ok, info = send_one(client, base_url, token, model)
            returned = info["returned_model"] or "-"
            status_str = "ERR" if info["status"] is None else str(info["status"])
            print(f"{model:<20} {returned:<20} {status_str:<8} {info['latency_ms']:>10.1f}")
            if not ok:
                failures += 1
                print(f"  FAILED: {info['error']}", file=sys.stderr)

        after = fetch_admin_status(client, base_url, token)

    if before is not None and after is not None:
        delta = after["ledger"]["consumed_usd"] - before["ledger"]["consumed_usd"]
        print(f"ledger consumed_usd delta: ${delta:.6f}")
    else:
        print(
            "admin/status unavailable (GATEWAY_TOKEN must map to the 'admin' "
            "principal) - skipping ledger delta",
            file=sys.stderr,
        )

    if failures:
        print(f"{failures}/{len(models)} model(s) failed", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    base_url = os.environ.get("GATEWAY_URL")
    token = os.environ.get("GATEWAY_TOKEN")
    if not base_url or not token:
        print("GATEWAY_URL and GATEWAY_TOKEN must both be set", file=sys.stderr)
        return 1
    return run(base_url.rstrip("/"), token)


if __name__ == "__main__":
    sys.exit(main())

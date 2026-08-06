#!/usr/bin/env python3
"""Is the fusion panel actually fusing? (stdlib only)

A degraded panel does not fail. It answers 200, with a plausible answer, from
whichever members are left -- and bills you for a fusion. On 2026-08-05 two of
three members rejected every call for hours; two benchmark runs completed, one
costing $11, before anyone noticed that what had been measured was a single
model wearing the panel's name.

Three checks, cheapest first:

  1. /healthz     -- WHICH code and config is answering. A gateway started
                     before your deploy looks identical to one started after
                     it, and that is how the runs above happened.
  2. /admin/panel -- per-member success rate over recent traffic, from the
                     event log. $0, no upstream calls, and it names the member
                     that is failing AND the upstream's own reason.
  3. one request  -- optional (--live, costs money): sends a multi-turn TOOL
                     conversation, the shape that broke it, and reports which
                     members appear in `fusion.panel` on the response.

Exit code is 1 if any member is unhealthy, so it can gate a benchmark run.

Usage:
    panel_health.py <base_url> <admin_token> [client_token] [--live]

    panel_health.py http://127.0.0.1:8800 $ADMIN_TOK
    panel_health.py http://127.0.0.1:8800 $ADMIN_TOK $CLIENT_TOK --live
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]

# Deliberately multi-turn and tool-carrying with a temperature: the exact shape
# that a single-turn smoke test passes and a real agent client does not.
LIVE_BODY = {
    "model": "fusion", "max_tokens": 2048, "temperature": 0.0, "tools": TOOLS,
    "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "List the files."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py  b.py"},
        {"role": "user", "content": "Reply with the single word DONE."},
    ],
}


def get(url: str, token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:160].decode('utf8', 'replace')}"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base = sys.argv[1].rstrip("/")
    admin = sys.argv[2]
    client = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else None
    live = "--live" in sys.argv

    print("=" * 68)
    print("1. WHICH gateway is answering")
    print("=" * 68)
    h, err = get(f"{base}/healthz")
    if err:
        print(f"  unreachable: {err}")
        return 1
    print(f"  config_sha  {h.get('config_sha')}   ({h.get('config_path')})")
    print(f"  started_at  {h.get('started_at')}")
    print(f"  panel       {h.get('fusion_panel')}")
    print("  ^ compare config_sha against `sha256sum <config> | cut -c1-12` on")
    print("    the host. A mismatch means the running process predates your edit.")

    print()
    print("=" * 68)
    print("2. PANEL HEALTH from the event log ($0)")
    print("=" * 68)
    d, err = get(f"{base}/admin/panel", admin)
    if err:
        print(f"  {err}")
        return 1
    if d.get("fusion") is None and "members" not in d:
        print("  no fusion configured")
        return 0
    print(f"  {'member':<16}{'ok':>6}{'failed':>8}{'ok_rate':>9}   last_error")
    for m in d["panel"]:
        v = d["members"][m]
        e = v["last_error"]
        note = "" if e is None else f"{e['status']} {e['body'][:70]}"
        rate = "—" if v["ok_rate"] is None else f"{v['ok_rate']:.3f}"
        print(f"  {m:<16}{v['ok']:>6}{v['failed']:>8}{rate:>9}   {note}")
    print(f"  paths: {d['paths']}")

    unhealthy = d["unhealthy"]
    if unhealthy:
        print(f"\n  UNHEALTHY: {unhealthy}")
        print("  These members are answering less than half the time. The panel")
        print("  is running on whatever is left, and every response still calls")
        print("  itself a fusion. Read last_error above -- it is the upstream's")
        print("  own words, and it is usually a rejected parameter.")

    if live and client:
        print()
        print("=" * 68)
        print("3. ONE LIVE REQUEST (multi-turn + tools + temperature) -- COSTS MONEY")
        print("=" * 68)
        r, err = get(f"{base}/v1/chat/completions", client, LIVE_BODY)
        if err:
            print(f"  {err}")
            return 1
        f = r.get("fusion") or {}
        print(f"  panel answered : {f.get('panel')}")
        print(f"  path           : {f.get('path')}   degraded={f.get('degraded')}")
        print(f"  usage          : {r.get('usage')}")
        if len(f.get("panel") or []) < 2:
            print("  ^ fewer than two members answered: this is NOT a fusion.")
            return 1
    elif live:
        print("\n  --live needs a client token as the third argument; skipped.")

    return 1 if unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())

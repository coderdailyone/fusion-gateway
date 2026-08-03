#!/usr/bin/env python3
"""Where is the latency? Run this ON the gateway host. (stdlib only)

A slow gateway and a slow path *to* the gateway look identical from a client.
This separates them by timing the same work at three distances:

    1. loopback     http://127.0.0.1:8800   — no TLS, no proxy, no network
    2. via nginx    https://<domain>        — adds TLS + reverse proxy
    3. upstreams    each provider directly  — how slow are the models from here

and at three depths, so gateway compute is separated from upstream wait:

    /healthz        returns {"ok": true}. No upstream, no DB, no compute.
    /v1/models      an in-memory dict. No upstream.
    one model       exactly one upstream call.
    fusion          the panel: 3+ upstream calls.

Read the result like this:

    loopback /healthz slow      -> the gateway process or the host is the problem
    loopback fast, domain slow  -> TLS / reverse proxy / the network path
    both fast, one model slow   -> the upstream provider is slow FROM THIS HOST
    single fast, fusion slow    -> expected; the panel waits on its slowest member

Usage:
    python3 scripts/diagnose_latency.py <token> [domain]

    token   a GATEWAY_TOKENS client token
    domain  public hostname, e.g. fusion.xinshu.ai (optional; skips section 2)
"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

LOCAL = "http://127.0.0.1:8800"


def timed(url: str, token: str, body: dict | None = None, timeout: float = 300):
    """Return (status, seconds, note). Never raises."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
        dt = time.monotonic() - t0
        note = ""
        try:
            d = json.loads(payload)
            if "choices" in d:
                note = f"model={d.get('model')}"
                if isinstance(d.get("fusion"), dict):
                    note += f" path={d['fusion'].get('path')}"
        except Exception:
            pass
        return 200, dt, note
    except urllib.error.HTTPError as e:
        return e.code, time.monotonic() - t0, e.read()[:80].decode("utf8", "replace")
    except Exception as e:
        return None, time.monotonic() - t0, f"{type(e).__name__}: {e}"[:80]


def row(label: str, status, dt: float, note: str = "") -> None:
    s = "ERR" if status is None else str(status)
    print(f"  {label:<34} {s:>4}  {dt:7.2f}s  {note}")


def tls_handshake(host: str, port: int = 443) -> None:
    """Time DNS, TCP and the TLS handshake separately.

    A slow handshake with a fast TCP connect means the far end or the path is
    slow, not the application — the application has not been reached yet.
    """
    try:
        t0 = time.monotonic()
        addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
        t_dns = time.monotonic() - t0
        t0 = time.monotonic()
        sock = socket.create_connection(addr, timeout=30)
        t_tcp = time.monotonic() - t0
        t0 = time.monotonic()
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(sock, server_hostname=host) as ss:
            t_tls = time.monotonic() - t0
            ver = ss.version()
        print(f"  resolved {host} -> {addr[0]}")
        print(f"  DNS {t_dns:.3f}s   TCP {t_tcp:.3f}s   TLS handshake {t_tls:.3f}s   ({ver})")
        if t_tls > 1.0:
            print("  ^^ a handshake over 1s means the PATH is slow; the gateway")
            print("     has not even been reached at this point.")
    except Exception as e:
        print(f"  handshake probe failed: {type(e).__name__}: {e}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    token = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else None

    chat = {"max_tokens": 16, "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}

    print("=" * 72)
    print("1. LOOPBACK — no TLS, no proxy, no network. This is the gateway itself.")
    print("=" * 72)
    row("GET /healthz", *timed(f"{LOCAL}/healthz", token))
    row("GET /v1/models", *timed(f"{LOCAL}/v1/models", token))
    for m in ("deepseek-chat", "glm-5.2", "kimi-k3"):
        row(f"POST one model: {m}", *timed(f"{LOCAL}/v1/chat/completions", token,
                                           dict(chat, model=m)))
    row("POST fusion (panel)", *timed(f"{LOCAL}/v1/chat/completions", token,
                                      dict(chat, model="fusion")))

    if domain:
        print()
        print("=" * 72)
        print(f"2. VIA {domain} — adds TLS + reverse proxy + the network path.")
        print("=" * 72)
        tls_handshake(domain)
        base = f"https://{domain}"
        row("GET /healthz", *timed(f"{base}/healthz", token))
        row("GET /v1/models", *timed(f"{base}/v1/models", token))
        row("POST one model: deepseek-chat",
            *timed(f"{base}/v1/chat/completions", token, dict(chat, model="deepseek-chat")))

    print()
    print("=" * 72)
    print("3. UPSTREAMS DIRECT — how slow are the providers FROM THIS HOST.")
    print("=" * 72)
    import os
    for label, url, key_env, payload in (
        ("deepseek", "https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY",
         {"model": "deepseek-v4-flash", "max_tokens": 16,
          "messages": [{"role": "user", "content": "hi"}]}),
        ("kimi", "https://api.kimi.com/coding/v1/chat/completions", "MOONSHOT_API_KEY",
         {"model": "k3", "max_tokens": 16,
          "messages": [{"role": "user", "content": "hi"}]}),
    ):
        k = os.environ.get(key_env)
        if not k:
            print(f"  {label:<34}  skipped ({key_env} not in env — "
                  f"run with `set -a; . /opt/fusion-gateway/.env; set +a`)")
            continue
        req = urllib.request.Request(url, data=json.dumps(payload).encode())
        req.add_header("Authorization", f"Bearer {k}")
        req.add_header("Content-Type", "application/json")
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                r.read()
            row(f"{label} direct", 200, time.monotonic() - t0)
        except urllib.error.HTTPError as e:
            row(f"{label} direct", e.code, time.monotonic() - t0,
                e.read()[:60].decode("utf8", "replace"))
        except Exception as e:
            row(f"{label} direct", None, time.monotonic() - t0, type(e).__name__)

    print()
    print("=" * 72)
    print("HOW TO READ IT")
    print("=" * 72)
    print("  loopback /healthz > 0.1s      the gateway process or the host is sick")
    print("  loopback fast, domain slow    TLS / reverse proxy / network path — not the gateway")
    print("  both fast, one model slow     that provider is slow from this host")
    print("  one model fast, fusion slow   expected: the panel waits on its slowest member")
    print("  loopback vs domain difference is the cost of everything in front of the gateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

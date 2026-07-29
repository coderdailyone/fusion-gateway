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

## Smoke executed 2026-07-26 (the human-gated step, finally run)

Ran `scripts/smoke.py` against a locally-started gateway with real provider keys.

| model | returned | status | latency |
|---|---|---|---|
| deepseek-chat | deepseek-chat | 200 | 812 ms |
| glm-4.5-flash | glm-4.5-flash | 200 | 1019 ms |

Ledger `consumed_usd` delta: **$0.000006** — cost metering, the preflight-settle
ledger, and `/admin/status` verified end to end.

**The smoke caught a config defect that would have broken the deploy:** `glm-4.6`
on `open.bigmodel.cn/api/paas/v4` returns error `1113` (no balance / no resource
pack), surfacing as `502 upstream_exhausted`. This reproduces the M2c finding.
`configs/gateway.toml` now serves **glm-4.5-flash**, which works on the same
endpoint and is priced at 0 pending a paid plan.

## Smoke executed 2026-07-28 — glm-5.2 over the Anthropic wire (M7)

The M7 adapter's paid live gate. Gateway started locally with real provider
keys; `configs/gateway.toml` now carries a **second** GLM provider,
`glm_anthropic` (`wire = "anthropic"`), rather than a `wire` flip on the
existing `glm` block — flipping that one would have rerouted `glm-4.5-flash`,
which is `deepseek-chat`'s fallback, onto a wire its endpoint does not speak.

### Non-streaming (`scripts/smoke.py`, every configured model)

| model | returned | status | latency |
|---|---|---|---|
| deepseek-chat | deepseek-chat | 200 | 1172 ms |
| glm-4.5-flash | glm-4.5-flash | 200 | 9328 ms |
| **glm-5.2** | **glm-5.2** | **200** | **849 ms** |

Ledger `consumed_usd` delta: **$0.000017**.

### The billing claim, verified

Every glm-5.2 row settled with `usage_source='reported'` — real translated
counts, not estimates:

| # | call | in | out | actual_usd |
|---|---|---:|---:|---:|
| 3 | non-streaming | 11 | 2 | $0.000011 |
| 4 | streaming | 9 | 9 | $0.0000252 |
| 5 | tool call, non-streaming | 157 | 12 | $0.0001206 |
| 6 | tool call, streaming | 157 | 12 | $0.0001206 |

All 6 ledger rows reached `settled`; none stranded in a consuming state. Total
outlay for the whole gate: **$0.000283**.

### Streaming

`stream: true` to glm-5.2 returned 11 chunks, every one `"chat.completion.chunk"`,
reassembling to `1, 2, 3.`, `finish_reason: "stop"`, then a usage-bearing chunk
and `data: [DONE]`. The existing OpenAI-wire `parse_stream_usage()` parses that
chunk unchanged — `{'prompt_tokens': 9, 'completion_tokens': 9, 'total_tokens': 18}` —
which is what lets `settle()` bill a translated stream with no ledger change.

### Tool calls, both directions, live

Non-streaming returned `finish_reason: "tool_calls"` with
`get_weather({"city": "Beijing"})`. Streaming delivered the same call as
`input_json_delta` fragments that reassembled into valid JSON
(`{"city":"Beijing"}`) — the fragment-reassembly path had until now only been
exercised against recorded fixtures.

### Raw upstream usage

```json
{"input_tokens": 11, "output_tokens": 2, "cache_read_input_tokens": 0,
 "server_tool_use": {"web_search_requests": 0}, "service_tier": "standard"}
```

`cache_read_input_tokens` is **0**, so M7's decision to read only
`input_tokens`/`output_tokens` costs nothing today. If it ever goes non-zero we
would *under*-bill (Anthropic excludes cache reads from `input_tokens`), so this
field is worth re-checking before glm-5.2 carries prompt-cached traffic.

**Prices are unverified.** `in 0.60 / out 2.20 USD-per-Mtok` mirror
`configs/pricing.toml`, still flagged `VERIFY` against Zhipu's published table.
They drive the ledger and therefore the budget killswitch — confirm them before
real traffic.

## Deployed to production 2026-07-28

`HOST=vps bash scripts/deploy.sh` — the M1 deploy, finally executed. Service is
`active`, `enabled`, **0 restarts**, listening on `127.0.0.1:8800`.

### Footprint (measured, not estimated)

| | size |
|---|---:|
| venv (fastapi/uvicorn/httpx/tomli) | 36 MB |
| source | ~1 MB |
| SQLite ledger | 40 KB at 7 requests; ~1–2 KB/request marginal |
| **total on host** | **36 MB** |

Host has 41 GB free. `python3-venv` was already present.

**The deploy script was shipping 1.07 GB.** `.gitignore` does not bind rsync:
`runs/` (972 MB of frozen evaluation samples) and `evaluator/runs/` (45 MB) were
git-ignored but not rsync-excluded — a 1000× payload of artifacts the production
gateway has no use for. Fixed by adding them to the exclude list; payload is now
983 KB. Verified after deploy: `/opt/fusion-gateway/runs` does not exist.

`runs/secrets/.env` was never at risk — `--exclude '.env'` carries no slash, so
rsync matches it at any depth (confirmed by dry run: 0 `.env` files transferred).
The only `.env` on the host is the one deliberately placed there, mode 600.

### Production smoke

| model | returned | status | latency |
|---|---|---|---|
| deepseek-chat | deepseek-chat | 200 | 759 ms |
| glm-4.5-flash | glm-4.5-flash | 200 | 666 ms |
| glm-5.2 | glm-5.2 | 200 | 1707 ms |

Ledger delta **$0.000017**. Streaming to glm-5.2 through production terminated
with `data: [DONE]` and settled `usage_source='reported'` (9 in / 9 out). All 4
ledger rows `settled`; none stranded.

Auth verified: no token → 401, unknown token → 401, non-admin principal on
`/admin/status` → 403.

`systemctl restart` came back clean — `active`, healthy, ledger still 4/4
settled, no orphan rows.

### Day-1 log

| Day | Requests | Cost (USD) | P95 latency | Fallbacks | Incidents |
|----:|---------:|-----------:|------------:|----------:|-----------|
| 1 | 4 (smoke) | 0.000017 | 1707 ms | 0 | none |

**Still open before real traffic:** glm-5.2's prices (0.60/2.20) are unverified
and drive the budget killswitch; the M1 cap is $5 and trips hard (503 on every
request until `/admin/killswitch/release`); the bind is loopback-only, so
external clients would need a reverse proxy and TLS that do not exist yet.

## Public endpoint live 2026-07-29 — https://gateway.cutecookie.xyz

nginx reverse proxy + Let's Encrypt (certbot `--nginx`, auto-renew scheduled).
HTTP 301s to HTTPS. The other 15 vhosts on the host were unaffected
(`nginx -t` before every reload; prism/ember/status still answering).

Verified from outside the host, over the public internet:

| check | result |
|---|---|
| TLS verify | 0 (valid) |
| `http://` → `https://` | 301 |
| no token / bad token on `/v1/models` | 401 / 401 |
| valid token on `/v1/models` | 200, all four models |
| non-streaming completion (glm-5.2) | 200, `"ok"`, usage reported |
| **streaming** (glm-5.2) | 19 SSE events, first byte 7.09 s, last 7.43 s — **0.33 s spread, so genuinely incremental**, not one buffered burst |
| `/admin/*` with a valid admin token | **403 — blocked at nginx** |
| `/admin/status` from the box | 200 |

**No SSO in front.** API clients authenticate with `Authorization: Bearer`;
a browser-redirect SSO layer would bounce every one of them to a login page.
The gateway's own token map is the auth boundary, and it holds: two 401s and a
403 above.

**`/admin/` is denied at nginx.** Budget status and the killswitch have no
reason to be internet-facing when the operator reaches them over SSH. Remove
the `location /admin/ { deny all; }` block to open it.

Streaming needed `proxy_buffering off` and a 900 s read timeout — the nginx
default of 60 s would cut off a reasoning model mid-generation.

### Client usage

```
base_url: https://gateway.cutecookie.xyz/v1
api_key:  <the prism token from /opt/fusion-gateway/.env>
models:   deepseek-chat | glm-4.5-flash | glm-5.2 | kimi-k3
```

**The budget is now uncapped** (`cap_usd` omitted from `[budgets.M1]`), so a
public endpoint holding real provider credit has no automatic ceiling. nginx
rate-limits to 10 r/s with burst 20 per IP; the only hard brake is
`POST /admin/killswitch/trip`, reachable on the box.

## M8 fusion live smoke 2026-07-29 — fusion is now the default path

First real fused requests. `policy.default_model = "fusion"`, so `model: "auto"`
is answered by the panel.

| | wall clock | result |
|---|---:|---|
| non-streaming | **5.86 s** | `path: quorum`, answer correct (17×23 = 391) |
| streaming | **6.44 s** | 12 valid `chat.completion.chunk`s, text `1\n2\n3\n4\n5`, `[DONE]` |

Streaming emitted **2 SSE comment keepalives** (`: fusion panel`, `: fusion
fusing`) during stages 1–2, then the fuser's stream — comments a conformant SDK
skips, so no idle timeout fires during the silent panel phase.

### Ledger — 6 rows per request under one `request_id`, none stranded

```
deepseek-chat  settled  reported   $0.00003122   <- candidate
glm-5.2        settled  reported   $0.00025560   <- candidate
kimi-k3        failed   -          -             <- 403, uncharged
deepseek-chat  settled  reported   $0.00005180   <- review
glm-5.2        settled  reported   $0.00014140   <- review
glm-5.2        settled  reported   $0.00030720   <- fuser
```

**Zero rows in `preflight`.** Total for both requests: **$0.001175**. Events
recorded: `fusion.started`, `fusion.candidate` ×3, `fusion.review` ×2,
`fusion.consensus`, `fusion.fused`, plus the usual `call.*` — full trace per
request.

### What this smoke does NOT establish

**kimi-k3's quota is still exhausted** (HTTP 403 `access_terminated_error`,
re-probed today after a reported top-up; the error says quota refreshes next
cycle and that continuing now needs "extra usage" specifically). So the panel
ran as **two models**, and kimi's leg failed instantly instead of taking ~34 s.

That means the measured 5.9 s / 6.4 s **does not validate the latency model**
(~15 s quorum path, ~44 s full path) — the slow leg that model exists to avoid
was never slow. It does validate that the degradation ladder works under a real
upstream failure: kimi 403s, its row is `failed` and uncharged, and the request
succeeds on the remaining two with `degraded: false` (correct — kimi is not a
quorum member and is cancelled on that path anyway).

Re-run this smoke once kimi-k3 has quota to measure the real latency split.

### Also unverified

Whether the fusion quality gain transfers from benchmarks to chat. There is no
grader on production traffic, so this cannot be measured online. M5's +1.1 pt
was on benchmark tasks at p = 0.176 — not significant. See the spec's "What we
know, and what we do not".

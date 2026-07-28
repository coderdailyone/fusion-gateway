# M7 — Anthropic Wire-Protocol Adapter

**Status:** design approved 2026-07-26
**Milestone:** M7 (gateway feature; unblocks serving glm-5.2 and Claude)

## Why

The gateway must serve **glm-5.2**, and the user's paid GLM balance is reachable
**only** through the Anthropic-compatible endpoint. Measured 2026-07-26:

| endpoint | model | result |
|---|---|---|
| `open.bigmodel.cn/api/anthropic/v1/messages` | glm-5.2 | **200 OK** |
| `open.bigmodel.cn/api/paas/v4/chat/completions` | glm-5.2 | `1113 余额不足` |
| `open.bigmodel.cn/api/paas/v4/chat/completions` | glm-4.5-flash | 200 (free tier) |

`gateway/providers.py::ProviderAdapter` hardcodes the OpenAI wire protocol —
`f"{base_url}/chat/completions"` with `Authorization: Bearer`. Serving an
Anthropic upstream therefore needs a second wire protocol, translated in both
directions, because **the gateway's public API stays OpenAI-compatible** (clients
only speak `/v1/chat/completions`).

The same protocol is what the Claude relay (`api.aicodemirror.com/api/claudecode`)
speaks, so this adapter unlocks a Claude leg too — which is what a later
verify-cascade milestone needs for its "escalate to a strong model" step.

## Positioning (locked with the user)

- **Real streaming, translated event by event.** Not buffer-then-convert.
- **A separate `AnthropicAdapter` class + a factory.** `ProviderAdapter` is not
  touched — the OpenAI path and its 251 passing tests carry zero regression risk.
- **Tool calls (`tool_use`) are in scope**, both directions, including streaming.
- Outward API stays OpenAI-compatible; no client change.

## Reference implementation studied

`Wei-Shaw/sub2api` (★34.8k), `backend/internal/service/gateway_forward_as_chat_completions.go`
(513 lines) does the same Anthropic→OpenAI direction. Four things it taught us,
all folded into this design:

1. **Usage arrives across two events, not one.** `message_start` carries
   `input_tokens` (plus cache counters); `message_delta` carries the final
   `output_tokens`. They must be merged — this feeds our `settle()`, so getting
   it wrong mis-bills every request.
2. **`content_block_delta` has three delta types** — `text_delta`,
   `thinking_delta`, `input_json_delta`. Not just text.
3. **Tool arguments stream as fragments** (`input_json_delta.partial_json`) that
   must be concatenated into one JSON string. This is the easiest part of
   tool streaming to get wrong.
4. **It buffers the whole stream and rebuilds a response** before converting.
   We deliberately do *not* copy that: it sacrifices real streaming, which the
   user chose to keep. We translate incrementally instead.

## Architecture

```
gateway/providers.py            ProviderAdapter        (OpenAI wire — UNCHANGED)
gateway/providers_anthropic.py  AnthropicAdapter       (Anthropic wire — new)
gateway/anthropic_translate.py  pure translation fns   (no IO — new)
gateway/providers.py            make_adapter(cfg)      (factory — small addition)
```

`AnthropicAdapter` exposes the **same interface** as `ProviderAdapter`:
`chat(upstream_model, payload) -> dict` and
`chat_stream(upstream_model, payload) -> AsyncIterator[bytes]`. `gateway/app.py`
therefore needs no change beyond obtaining adapters from the factory.

Translation lives in a **pure module** (`anthropic_translate.py`) with no IO, so
every mapping rule is unit-testable without a network or a fake server.

### Config

`ProviderCfg` gains one field: **`wire: str = "openai"`** (`"openai"` |
`"anthropic"`). Chosen explicitly in `configs/gateway.toml` rather than guessed
from the URL — guessing would silently mis-route a provider whose URL changes.

```toml
[providers.glm_anthropic]
base_url = "https://open.bigmodel.cn/api/anthropic"
api_key_env = "GLM_API_KEY"
wire = "anthropic"
```

`make_adapter(cfg)` returns `AnthropicAdapter` when `cfg.wire == "anthropic"`,
else `ProviderAdapter`. An unknown `wire` value is a `ConfigError` at load time,
not a runtime surprise.

## Request translation (OpenAI → Anthropic)

| OpenAI | Anthropic | rule |
|---|---|---|
| `messages[]` with `role: "system"` | top-level `system` string | Anthropic has no system *message*; concatenate all system messages |
| `messages[]` user/assistant | `messages[]` | passed through; content coerced to Anthropic block form |
| `max_tokens` | `max_tokens` | **required** by Anthropic — default to 4096 when the client omits it |
| `temperature`, `top_p`, `stop` → `stop_sequences` | same | direct |
| `tools[].function` | `tools[]` with `input_schema` | OpenAI nests under `function`; Anthropic is flat with `input_schema` |
| `tool_choice` | `tool_choice` | `"auto"`/`"none"`/named-function mapped |
| assistant `tool_calls` | `content` blocks of `type: "tool_use"` | for multi-turn tool conversations |
| `role: "tool"` result message | user message with `type: "tool_result"` block | Anthropic returns results as user content |

Headers: `x-api-key: <key>` + `anthropic-version: 2023-06-01` (not `Bearer`).
URL: `f"{base_url}/v1/messages"`.

## Response translation (Anthropic → OpenAI), non-streaming

`content` blocks → one OpenAI message:
- `text` blocks concatenate into `message.content`;
- `tool_use` blocks become `message.tool_calls[]` with
  `function.name` / `function.arguments` (JSON-encoded `input`);
- `thinking` blocks are **dropped from `content`** (OpenAI has no field for
  them) and their presence is recorded in the trace event, not silently lost.

`stop_reason` maps `end_turn|max_tokens|stop_sequence|tool_use` →
`stop|length|stop|tool_calls`. `usage.input_tokens`/`output_tokens` →
`usage.prompt_tokens`/`completion_tokens` (+ `total_tokens`).

## Streaming translation (the crux)

Anthropic SSE events → OpenAI `chat.completion.chunk` SSE, emitted incrementally:

| Anthropic event | emitted OpenAI chunk |
|---|---|
| `message_start` | first chunk with `delta: {"role": "assistant"}`; **record `input_tokens`** |
| `content_block_start` (`text`) | nothing (wait for deltas) |
| `content_block_start` (`tool_use`) | chunk with `tool_calls[i]` carrying `id` + `function.name`, empty arguments |
| `content_block_delta` / `text_delta` | chunk with `delta.content` |
| `content_block_delta` / `input_json_delta` | chunk with `tool_calls[i].function.arguments` = the fragment (**concatenated by the client, as OpenAI does**) |
| `content_block_delta` / `thinking_delta` | dropped (counted) |
| `content_block_stop` | nothing |
| `message_delta` | **record `output_tokens`**; carry `stop_reason` → `finish_reason` |
| `message_stop` | final chunk with `finish_reason`, then a chunk carrying `usage`, then `data: [DONE]` |
| `error` | if before first byte → `ProviderError`; after → terminate the stream |

**The first-byte contract is preserved:** `AnthropicAdapter.chat_stream` raises
`ProviderError` only before it has yielded its first byte — exactly like the
existing adapter — so the fallback chain stays safe to use. Non-2xx status and
transport errors are detected before any translation begins.

**Usage for billing:** the adapter emits a final chunk containing an OpenAI-shaped
`usage` object built from the merged `message_start` + `message_delta` counters,
so the existing `parse_stream_usage()` in `gateway/providers.py` finds it
unchanged and `settle()` bills real tokens. This is what makes billing correct
without touching the ledger.

## Error handling

- Non-2xx from upstream → `ProviderError(provider, "http", status)` before any
  bytes are yielded; the fallback chain proceeds.
- Timeout / transport error before first byte → `ProviderError("timeout"/"network")`;
  after first byte → re-raised (same rule as the OpenAI adapter).
- Anthropic `error` SSE event before first byte → `ProviderError("http")`;
  after first byte → stop translating and end the stream cleanly.
- Malformed SSE line or unparseable JSON → skipped, counted, never fatal.
- `content_block_delta` for an index we never saw start → ignored.
- A client request with no `max_tokens` gets **4096**, since Anthropic rejects
  the request without it.

## Testing

- Unit (pure, no network): request translation — system-message hoisting,
  `max_tokens` default, tools/`input_schema` shape, `tool_choice`, assistant
  `tool_calls` → `tool_use`, tool-result → user `tool_result`.
- Unit: non-streaming response translation — text, `tool_use` → `tool_calls`,
  `stop_reason` mapping, usage mapping, thinking dropped.
- Unit: streaming translation over **recorded Anthropic SSE fixtures** — a text
  stream, a tool-call stream whose `input_json_delta` fragments reassemble into
  valid JSON, and a stream whose `usage` spans `message_start` + `message_delta`.
- Contract: `ProviderError` before first byte; not raised after first byte.
- Contract: the emitted stream ends with a `usage`-bearing chunk that the
  existing `parse_stream_usage()` parses.
- Factory: `wire="anthropic"` yields `AnthropicAdapter`, default yields
  `ProviderAdapter`, unknown value raises `ConfigError`.
- Regression: the existing 251 tests must still pass untouched.
- Live smoke (gated, real key): one non-streaming and one streaming call to
  glm-5.2 through the gateway, checking a 200, sane content, and a non-zero
  ledger delta.

## Acceptance criteria

1. `configs/gateway.toml` serves **glm-5.2** through the Anthropic wire, and
   `scripts/smoke.py` passes for both glm-5.2 and deepseek-chat.
2. A streaming request to glm-5.2 through the gateway returns valid OpenAI
   `chat.completion.chunk` SSE that an OpenAI SDK consumes without modification.
3. Tool calls work in both directions, non-streaming and streaming, with
   fragment reassembly producing valid JSON arguments.
4. The ledger records **real** token counts for Anthropic-wire calls (usage
   merged from `message_start` + `message_delta`), verified in the smoke.
5. `ProviderAdapter` and the OpenAI path are unmodified; all 251 pre-existing
   tests still pass.
6. `ProviderError` timing contract holds for the new adapter (before first byte
   only), so fallback remains safe.

## Non-goals

- No vision/image content blocks.
- No `thinking` exposure to clients (dropped and counted; a later milestone can
  surface it if a client needs it).
- No prompt-caching control surface (cache counters are read for billing only).
- No change to the gateway's public API, auth, ledger, or event schema.
- Not wiring the learned router or verify-cascade — a separate milestone.

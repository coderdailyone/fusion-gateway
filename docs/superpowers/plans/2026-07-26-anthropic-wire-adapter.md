# M7 — Anthropic Wire-Protocol Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the gateway serve `glm-5.2` (and any Claude relay) by talking the Anthropic wire protocol upstream, while its public API stays OpenAI-compatible.

**Architecture:** A pure translation module (`gateway/anthropic_translate.py`, zero IO) converts requests, non-streaming responses, and SSE events between the two protocols. A new `AnthropicAdapter` (`gateway/providers_anthropic.py`) exposes the *same* interface as the existing `ProviderAdapter` and does only HTTP. A one-line factory picks between them on a new `ProviderCfg.wire` field. The existing OpenAI adapter is not modified.

**Tech Stack:** Python 3.10, httpx (async, `MockTransport` for tests), pytest, FastAPI (unchanged), tomllib.

## Global Constraints

- **`gateway/providers.py::ProviderAdapter` must not be modified.** Only *additions* to that file (the factory) are allowed. The OpenAI path carries zero regression risk.
- **All 251 pre-existing tests must still pass**, untouched.
- **`ProviderError` timing contract:** the new adapter raises `ProviderError` **only before it has yielded its first byte** to the caller. After the first byte, errors propagate as ordinary exceptions. `gateway/app.py` relies on this to make fallback safe.
- **Billing contract:** the translated stream must end with an OpenAI-shaped chunk carrying `usage`, so the existing `gateway/providers.py::parse_stream_usage()` finds it **unchanged** and `settle()` bills real tokens.
- **Usage must be merged across two events:** `message_start` carries `input_tokens`, `message_delta` carries the final `output_tokens`. Neither alone is sufficient.
- **Public API unchanged:** clients keep using `POST /v1/chat/completions` with `Authorization: Bearer <gateway token>`. No client-visible change.
- Anthropic upstream call: `POST {base_url}/v1/messages`, headers `x-api-key: <key>` and `anthropic-version: 2023-06-01` (**not** `Authorization: Bearer`).
- `max_tokens` is **required** by Anthropic; default to **4096** when the client omits it.
- Out of scope: vision/image blocks, exposing `thinking` to clients, prompt-caching controls.
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Branch** `feat/m0-m1-gateway`; **venv** `.venv` (Python 3.10); gateway tests live in `tests/` (evaluator tests in `tests/eval/`); run the whole suite with `.venv/bin/pytest -q`.

---

## File Structure

- `gateway/config.py` — **modify**: `ProviderCfg` gains `wire: str = "openai"`; `load_config` reads and validates it.
- `gateway/providers.py` — **additive only**: `make_adapter(cfg, transport=None)` factory. `ProviderAdapter`, `ProviderError`, `parse_stream_usage` untouched.
- `gateway/anthropic_translate.py` — **new**: pure translation. Request, non-streaming response, and a stateful `StreamTranslator` for SSE. No IO, no httpx.
- `gateway/providers_anthropic.py` — **new**: `AnthropicAdapter`, HTTP only, delegating all shape work to the translation module.
- `gateway/app.py` — **modify**: one line, `ProviderAdapter(...)` → `make_adapter(...)`.
- `configs/gateway.toml` — **modify**: add the Anthropic-wire GLM provider and the `glm-5.2` model.
- Tests: `tests/test_anthropic_translate.py`, `tests/test_providers_anthropic.py`, plus additions to `tests/test_config.py`.

---

### Task 1: `wire` field on the config + adapter factory

**Files:**
- Modify: `gateway/config.py` (the `ProviderCfg` dataclass and the `providers = {...}` comprehension in `load_config`)
- Modify: `gateway/providers.py` (append `make_adapter`; change nothing else)
- Modify: `gateway/app.py` (line 135: construct via the factory)
- Test: `tests/test_config.py` (append), `tests/test_providers_anthropic.py` (new)

**Interfaces:**
- Produces: `ProviderCfg(name, base_url, api_key_env, wire="openai")`; `make_adapter(cfg: ProviderCfg, transport=None) -> ProviderAdapter | AnthropicAdapter`.
- Consumes: `ConfigError` from `gateway/config.py`; `AnthropicAdapter` from Task 5 — **it does not exist yet**, so `make_adapter` imports it lazily *inside* the function body. That keeps this task independently testable and avoids a circular import (`providers_anthropic` imports `ProviderError` from `providers`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py  (append to the existing file)
import pytest
from pathlib import Path
from gateway.config import load_config, ConfigError

_BASE = """
[budget]
active = "T"
[budgets.T]
cap_usd = 1.0
[providers.p_openai]
base_url = "https://example.test/v1"
api_key_env = "X_KEY"
[providers.p_anthropic]
base_url = "https://example.test/anthropic"
api_key_env = "X_KEY"
wire = "anthropic"
[models."m1"]
provider = "p_openai"
upstream_model = "u1"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 2.0
[policy]
version = "test-v0"
default_model = "m1"
"""


def test_wire_defaults_to_openai_and_is_read_when_given(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(_BASE)
    cfg = load_config(p)
    assert cfg.providers["p_openai"].wire == "openai"      # default
    assert cfg.providers["p_anthropic"].wire == "anthropic"  # explicit


def test_unknown_wire_is_a_config_error(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(_BASE.replace('wire = "anthropic"', 'wire = "carrier-pigeon"'))
    with pytest.raises(ConfigError):
        load_config(p)
```

```python
# tests/test_providers_anthropic.py  (new file)
from gateway.config import ProviderCfg
from gateway.providers import ProviderAdapter, make_adapter
from gateway.providers_anthropic import AnthropicAdapter


def test_factory_picks_the_adapter_from_the_wire_field():
    openai_cfg = ProviderCfg("p", "https://example.test/v1", "X_KEY")
    anthropic_cfg = ProviderCfg("q", "https://example.test/anthropic", "X_KEY",
                                wire="anthropic")
    assert isinstance(make_adapter(openai_cfg), ProviderAdapter)
    assert isinstance(make_adapter(anthropic_cfg), AnthropicAdapter)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_providers_anthropic.py -v`
Expected: config tests FAIL (`ProviderCfg` takes 3 args / no `wire`); the factory test FAILs with `ModuleNotFoundError: gateway.providers_anthropic`.

- [ ] **Step 3: Implement**

In `gateway/config.py`, give the dataclass a defaulted field:

```python
@dataclass(frozen=True)
class ProviderCfg:
    name: str; base_url: str; api_key_env: str; wire: str = "openai"
```

and validate it in `load_config`, replacing the existing `providers = {...}` comprehension:

```python
    providers = {}
    for n, p in data["providers"].items():
        wire = p.get("wire", "openai")
        if wire not in ("openai", "anthropic"):
            raise ConfigError(f"provider {n}: unknown wire {wire!r}")
        providers[n] = ProviderCfg(n, p["base_url"], p["api_key_env"], wire)
```

Append the factory to `gateway/providers.py` (do not touch anything above it):

```python
def make_adapter(cfg: ProviderCfg, transport: httpx.AsyncBaseTransport | None = None):
    """Return the adapter that speaks this provider's wire protocol.

    Imported lazily: gateway.providers_anthropic imports ProviderError from
    this module, so a top-level import here would be circular.
    """
    if cfg.wire == "anthropic":
        from gateway.providers_anthropic import AnthropicAdapter

        return AnthropicAdapter(cfg, transport=transport)
    return ProviderAdapter(cfg, transport=transport)
```

In `gateway/app.py`, change the single construction site (line 135) from
`adapters[name] = ProviderAdapter(provider_cfg, transport=transport)` to
`adapters[name] = make_adapter(provider_cfg, transport=transport)`, and update
that module's import to bring in `make_adapter` alongside what it already imports.

> `tests/test_providers_anthropic.py` cannot pass until Task 5 creates
> `AnthropicAdapter`. Create a **minimal placeholder now** so this task is
> self-contained — Task 5 replaces its body:
> ```python
> # gateway/providers_anthropic.py
> """Anthropic-wire provider adapter (filled in by Task 5)."""
> from __future__ import annotations
>
> import httpx
>
> from gateway.config import ProviderCfg
>
>
> class AnthropicAdapter:
>     def __init__(self, cfg: ProviderCfg, timeout_s: float = 120.0,
>                  transport: httpx.AsyncBaseTransport | None = None):
>         self.cfg = cfg
>         self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
> ```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py tests/test_providers_anthropic.py -v`
Expected: PASS (3 new tests).

- [ ] **Step 5: Run the whole suite — nothing may regress**

Run: `.venv/bin/pytest -q`
Expected: `254 passed` (251 + 3).

- [ ] **Step 6: Commit**

```bash
git add gateway/config.py gateway/providers.py gateway/providers_anthropic.py gateway/app.py tests/test_config.py tests/test_providers_anthropic.py
git commit -m "feat(gateway): provider wire field + adapter factory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Request translation (OpenAI → Anthropic)

**Files:**
- Create: `gateway/anthropic_translate.py`
- Test: `tests/test_anthropic_translate.py`

**Interfaces:**
- Produces: `to_anthropic_request(payload: dict, upstream_model: str) -> dict`; module constant `DEFAULT_MAX_TOKENS = 4096`; `ANTHROPIC_VERSION = "2023-06-01"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anthropic_translate.py  (new file)
import json

from gateway.anthropic_translate import DEFAULT_MAX_TOKENS, to_anthropic_request


def test_system_messages_are_hoisted_to_the_top_level_field():
    out = to_anthropic_request({
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "system", "content": "and polite"},
            {"role": "user", "content": "hi"},
        ]}, "glm-5.2")
    assert out["system"] == "be terse\n\nand polite"
    assert [m["role"] for m in out["messages"]] == ["user"]
    assert out["model"] == "glm-5.2"


def test_max_tokens_defaults_because_anthropic_requires_it():
    out = to_anthropic_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert out["max_tokens"] == DEFAULT_MAX_TOKENS
    given = to_anthropic_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}, "m")
    assert given["max_tokens"] == 16


def test_sampling_params_and_stop_are_mapped():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2, "top_p": 0.9, "stop": ["END"],
    }, "m")
    assert out["temperature"] == 0.2 and out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"]
    assert "stop" not in out


def test_tools_are_flattened_with_input_schema():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function",
                   "function": {"name": "search", "description": "find",
                                "parameters": schema}}],
        "tool_choice": "auto",
    }, "m")
    assert out["tools"] == [{"name": "search", "description": "find",
                             "input_schema": schema}]
    assert out["tool_choice"] == {"type": "auto"}


def test_named_tool_choice_is_mapped():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "search",
                                                    "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
    }, "m")
    assert out["tool_choice"] == {"type": "tool", "name": "search"}


def test_assistant_tool_calls_become_tool_use_blocks():
    out = to_anthropic_request({"messages": [
        {"role": "user", "content": "search cats"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": '{"q": "cats"}'}}]},
    ]}, "m")
    blocks = out["messages"][1]["content"]
    assert blocks == [{"type": "tool_use", "id": "call_1", "name": "search",
                       "input": {"q": "cats"}}]


def test_tool_result_messages_become_user_tool_result_blocks():
    out = to_anthropic_request({"messages": [
        {"role": "user", "content": "search cats"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found 3"},
    ]}, "m")
    last = out["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [{"type": "tool_result", "tool_use_id": "call_1",
                                "content": "found 3"}]


def test_gateway_only_fields_are_not_forwarded():
    out = to_anthropic_request({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "stream_options": {"include_usage": True},
        "model": "whatever-the-client-asked-for",
    }, "upstream-name")
    assert out["model"] == "upstream-name"
    assert "stream" not in out and "stream_options" not in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gateway.anthropic_translate'`.

- [ ] **Step 3: Implement**

```python
# gateway/anthropic_translate.py
"""Pure translation between the OpenAI and Anthropic wire protocols.

No IO lives here — every function is a data transform, so each mapping rule is
unit-testable without a network or a fake server. The gateway's public API stays
OpenAI-shaped; these functions exist so an Anthropic upstream can serve it.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096  # Anthropic rejects a request without max_tokens

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _content_to_anthropic(content: Any) -> Any:
    """OpenAI content is a string or a list of parts; Anthropic takes either a
    string or a list of blocks. Strings pass through unchanged."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
    return blocks or ""


def to_anthropic_request(payload: dict, upstream_model: str) -> dict:
    """Translate an OpenAI chat-completions body into an Anthropic messages body."""
    system_parts: list[str] = []
    messages: list[dict] = []

    for msg in payload.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_parts.append(msg.get("content") or "")
            continue
        if role == "tool":
            # Anthropic returns tool results as USER content blocks.
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content") or "",
            }]})
            continue
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            text = msg.get("content")
            if text:
                blocks.append({"type": "text", "text": text})
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                raw = fn.get("arguments") or "{}"
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {}
                blocks.append({"type": "tool_use", "id": call.get("id", ""),
                               "name": fn.get("name", ""), "input": parsed})
            messages.append({"role": "assistant", "content": blocks})
            continue
        messages.append({"role": role, "content": _content_to_anthropic(msg.get("content"))})

    out: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "max_tokens": int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS),
    }
    if system_parts:
        out["system"] = "\n\n".join(s for s in system_parts if s)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if payload.get(src) is not None:
            out[dst] = payload[src]
    stop = payload.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    if payload.get("tools"):
        tools = []
        for t in payload["tools"]:
            fn = t.get("function", t)
            tool = {"name": fn.get("name", ""),
                    "input_schema": fn.get("parameters") or {"type": "object"}}
            if fn.get("description"):
                tool["description"] = fn["description"]
            tools.append(tool)
        out["tools"] = tools

    choice = payload.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(choice, dict):
        name = choice.get("function", {}).get("name") or choice.get("name")
        if name:
            out["tool_choice"] = {"type": "tool", "name": name}
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: PASS (8 tests). Note the `description` key: the flattened-tools test expects it present, and it is only emitted when the source has one.

- [ ] **Step 5: Whole suite + commit**

Run: `.venv/bin/pytest -q` → expected `262 passed`.

```bash
git add gateway/anthropic_translate.py tests/test_anthropic_translate.py
git commit -m "feat(gateway): OpenAI to Anthropic request translation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Non-streaming response translation (Anthropic → OpenAI)

**Files:**
- Modify: `gateway/anthropic_translate.py` (append)
- Test: `tests/test_anthropic_translate.py` (append)

**Interfaces:**
- Produces: `from_anthropic_response(resp: dict, model: str) -> dict` returning an OpenAI `chat.completion` object.
- Consumes: `_STOP_REASON_MAP` from Task 2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anthropic_translate.py  (append)
from gateway.anthropic_translate import from_anthropic_response


def _resp(content, stop_reason="end_turn", usage=None):
    return {"id": "msg_1", "type": "message", "role": "assistant",
            "model": "glm-5.2", "content": content, "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 11, "output_tokens": 7}}


def test_text_blocks_concatenate_into_message_content():
    out = from_anthropic_response(
        _resp([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]), "glm-5.2")
    assert out["object"] == "chat.completion"
    assert out["model"] == "glm-5.2"
    choice = out["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"


def test_usage_is_renamed_to_openai_fields():
    out = from_anthropic_response(_resp([{"type": "text", "text": "x"}]), "m")
    assert out["usage"] == {"prompt_tokens": 11, "completion_tokens": 7,
                            "total_tokens": 18}


def test_tool_use_blocks_become_tool_calls_with_json_arguments():
    out = from_anthropic_response(_resp(
        [{"type": "tool_use", "id": "toolu_9", "name": "search",
          "input": {"q": "cats"}}], stop_reason="tool_use"), "m")
    choice = out["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "toolu_9" and call["type"] == "function"
    assert call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"q": "cats"}
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None   # tool-only reply


def test_thinking_blocks_are_dropped_not_leaked_into_content():
    out = from_anthropic_response(_resp(
        [{"type": "thinking", "thinking": "secret chain"},
         {"type": "text", "text": "answer"}]), "m")
    assert out["choices"][0]["message"]["content"] == "answer"
    assert "secret chain" not in json.dumps(out)


def test_stop_reason_mapping_covers_every_value():
    for anthropic, openai in (("end_turn", "stop"), ("max_tokens", "length"),
                              ("stop_sequence", "stop"), ("tool_use", "tool_calls")):
        out = from_anthropic_response(
            _resp([{"type": "text", "text": "x"}], stop_reason=anthropic), "m")
        assert out["choices"][0]["finish_reason"] == openai
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: the 5 new tests FAIL with `ImportError: cannot import name 'from_anthropic_response'`.

- [ ] **Step 3: Implement (append to `gateway/anthropic_translate.py`)**

```python
def from_anthropic_response(resp: dict, model: str) -> dict:
    """Translate an Anthropic messages response into an OpenAI chat.completion."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    thinking_blocks = 0

    for block in resp.get("content", []) or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {"name": block.get("name", ""),
                             "arguments": json.dumps(block.get("input", {}))},
            })
        elif btype == "thinking":
            thinking_blocks += 1  # counted, never surfaced (OpenAI has no field)

    message: dict[str, Any] = {"role": "assistant",
                               "content": "".join(text_parts) if text_parts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_in = resp.get("usage", {}) or {}
    prompt = int(usage_in.get("input_tokens", 0) or 0)
    completion = int(usage_in.get("output_tokens", 0) or 0)

    return {
        "id": resp.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _STOP_REASON_MAP.get(resp.get("stop_reason"), "stop"),
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: PASS (13 tests in the file).

- [ ] **Step 5: Whole suite + commit**

Run: `.venv/bin/pytest -q` → expected `267 passed`.

```bash
git add gateway/anthropic_translate.py tests/test_anthropic_translate.py
git commit -m "feat(gateway): Anthropic to OpenAI response translation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Streaming translation (the crux)

**Files:**
- Modify: `gateway/anthropic_translate.py` (append)
- Test: `tests/test_anthropic_translate.py` (append)

**Interfaces:**
- Produces: `iter_sse_data(raw: bytes) -> Iterator[dict]` (parses `data:` lines, skipping malformed ones); `StreamTranslator` with `__init__(self, model: str)`, `feed(self, event: dict) -> list[dict]`, `finish(self) -> list[dict]`, and attributes `input_tokens`, `output_tokens`, `thinking_blocks`.
- Consumes: `_STOP_REASON_MAP`.

> **Why a class:** streaming translation is stateful — usage accumulates across
> `message_start` and `message_delta`, and tool-call indices must map from
> Anthropic's content-block indices to OpenAI's `tool_calls[].index`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anthropic_translate.py  (append)
from gateway.anthropic_translate import StreamTranslator, iter_sse_data


def _drain(translator, events):
    out = []
    for e in events:
        out.extend(translator.feed(e))
    out.extend(translator.finish())
    return out


def test_iter_sse_data_parses_data_lines_and_skips_junk():
    raw = (b'event: message_start\n'
           b'data: {"type": "message_start"}\n\n'
           b': a comment line\n'
           b'data: not-json\n\n'
           b'data: {"type": "message_stop"}\n\n')
    got = list(iter_sse_data(raw))
    assert [g["type"] for g in got] == ["message_start", "message_stop"]


def test_text_stream_translates_to_openai_chunks_with_role_then_content():
    t = StreamTranslator("glm-5.2")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "he"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "llo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c["choices"])
    assert text == "hello"
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if c["choices"] and c["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]


def test_usage_is_merged_from_message_start_and_message_delta():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[-1]["usage"] == {"prompt_tokens": 12,
                                         "completion_tokens": 5,
                                         "total_tokens": 17}


def test_tool_stream_emits_name_then_reassemblable_argument_fragments():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"q":'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": ' "cats"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ])
    # the opening chunk names the tool
    starts = [c for c in chunks if c["choices"]
              and c["choices"][0]["delta"].get("tool_calls")
              and c["choices"][0]["delta"]["tool_calls"][0]["function"].get("name")]
    assert starts[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1"
    # fragments concatenate into valid JSON, exactly as an OpenAI client does
    args = "".join(
        tc["function"].get("arguments", "")
        for c in chunks if c["choices"]
        for tc in c["choices"][0]["delta"].get("tool_calls", []))
    assert json.loads(args) == {"q": "cats"}
    finishes = [c["choices"][0]["finish_reason"] for c in chunks
                if c["choices"] and c["choices"][0].get("finish_reason")]
    assert finishes == ["tool_calls"]


def test_thinking_deltas_are_dropped_and_counted():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "secret"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ])
    assert "secret" not in json.dumps(chunks)
    assert t.thinking_blocks == 1


def test_delta_for_an_unknown_block_index_is_ignored_not_fatal():
    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_delta", "index": 7,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "message_stop"},
    ])
    assert isinstance(chunks, list)   # no exception raised
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: the 6 new tests FAIL with `ImportError: cannot import name 'StreamTranslator'`.

- [ ] **Step 3: Implement (append to `gateway/anthropic_translate.py`)**

```python
def iter_sse_data(raw: bytes) -> Iterator[dict]:
    """Yield the JSON object of every well-formed SSE `data:` line.

    Anthropic also sends `event:` lines, but each data payload carries its own
    "type", so the event lines are redundant. Malformed lines are skipped —
    never fatal.
    """
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


class StreamTranslator:
    """Anthropic SSE events -> OpenAI chat.completion.chunk objects.

    Stateful on purpose: usage arrives split across `message_start`
    (input_tokens) and `message_delta` (output_tokens), and Anthropic's
    content-block indices must be mapped onto OpenAI's tool_calls[].index.
    """

    def __init__(self, model: str):
        self.model = model
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_blocks = 0
        self._finish_reason: str | None = None
        self._tool_index: dict[int, int] = {}   # anthropic block idx -> tool_calls idx
        self._next_tool_index = 0
        self._role_sent = False

    def _chunk(self, delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": self.id, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish_reason}],
        }

    def feed(self, event: dict) -> list[dict]:
        etype = event.get("type")
        out: list[dict] = []

        if etype == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            self.input_tokens = int(usage.get("input_tokens", 0) or 0)
            if not self._role_sent:
                self._role_sent = True
                out.append(self._chunk({"role": "assistant"}))

        elif etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = self._next_tool_index
                self._next_tool_index += 1
                self._tool_index[event.get("index", -1)] = idx
                out.append(self._chunk({"tool_calls": [{
                    "index": idx, "id": block.get("id", ""), "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }]}))
            elif block.get("type") == "thinking":
                self.thinking_blocks += 1

        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                out.append(self._chunk({"content": delta.get("text", "")}))
            elif dtype == "input_json_delta":
                idx = self._tool_index.get(event.get("index", -1))
                if idx is not None:      # unknown block index -> ignore, not fatal
                    out.append(self._chunk({"tool_calls": [{
                        "index": idx, "type": "function",
                        "function": {"arguments": delta.get("partial_json", "")},
                    }]}))
            elif dtype == "thinking_delta":
                pass                      # dropped: OpenAI has no field for it

        elif etype == "message_delta":
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self.output_tokens = int(usage["output_tokens"] or 0)
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                self._finish_reason = _STOP_REASON_MAP.get(reason, "stop")

        return out

    def finish(self) -> list[dict]:
        """Terminal chunks: finish_reason, then a usage-bearing chunk.

        The usage chunk is what gateway.providers.parse_stream_usage() finds,
        which is how settle() bills real tokens for an Anthropic upstream.
        """
        out = [self._chunk({}, finish_reason=self._finish_reason or "stop")]
        usage_chunk = {
            "id": self.id, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model, "choices": [],
            "usage": {"prompt_tokens": self.input_tokens,
                      "completion_tokens": self.output_tokens,
                      "total_tokens": self.input_tokens + self.output_tokens},
        }
        out.append(usage_chunk)
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_anthropic_translate.py -v`
Expected: PASS (19 tests in the file).

- [ ] **Step 5: Verify the billing contract explicitly**

Add this test to `tests/test_anthropic_translate.py`, then run the file again:

```python
def test_emitted_stream_is_parseable_by_the_existing_usage_parser():
    from gateway.providers import parse_stream_usage

    t = StreamTranslator("m")
    chunks = _drain(t, [
        {"type": "message_start", "message": {"usage": {"input_tokens": 21}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ])
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    wire += b"data: [DONE]\n\n"
    assert parse_stream_usage(wire) == {"prompt_tokens": 21,
                                        "completion_tokens": 4,
                                        "total_tokens": 25}
```

Expected: PASS. This is the contract that keeps the ledger honest — if it fails, the adapter would bill zeros.

- [ ] **Step 6: Whole suite + commit**

Run: `.venv/bin/pytest -q` → expected `274 passed`.

```bash
git add gateway/anthropic_translate.py tests/test_anthropic_translate.py
git commit -m "feat(gateway): Anthropic SSE to OpenAI chunk stream translation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `AnthropicAdapter` (HTTP + the error-timing contract)

**Files:**
- Modify: `gateway/providers_anthropic.py` (replace the Task-1 placeholder body)
- Test: `tests/test_providers_anthropic.py` (append)

**Interfaces:**
- Produces: `AnthropicAdapter(cfg, timeout_s=120.0, transport=None)` with `async chat(upstream_model, payload) -> dict` and `async chat_stream(upstream_model, payload) -> AsyncIterator[bytes]`.
- Consumes: `to_anthropic_request`, `from_anthropic_response`, `iter_sse_data`, `StreamTranslator`, `ANTHROPIC_VERSION` (Tasks 2–4); `ProviderError` from `gateway/providers.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers_anthropic.py  (append)
import json

import httpx
import pytest

from gateway.config import ProviderCfg
from gateway.providers import ProviderError, parse_stream_usage
from gateway.providers_anthropic import AnthropicAdapter

CFG = ProviderCfg("glm_anthropic", "https://example.test/anthropic", "X_KEY",
                  wire="anthropic")


def _adapter(handler, monkeypatch):
    monkeypatch.setenv("X_KEY", "secret-key")
    return AnthropicAdapter(CFG, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_chat_posts_to_v1_messages_with_anthropic_headers(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}})

    out = await _adapter(handler, monkeypatch).chat(
        "glm-5.2", {"messages": [{"role": "user", "content": "hi"}]})

    assert seen["url"] == "https://example.test/anthropic/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert "authorization" not in {k.lower() for k in seen["headers"]}
    assert seen["body"]["model"] == "glm-5.2"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["usage"]["prompt_tokens"] == 5


@pytest.mark.asyncio
async def test_chat_raises_provider_error_on_non_2xx(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ProviderError) as exc:
        await _adapter(handler, monkeypatch).chat(
            "m", {"messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.kind == "http" and exc.value.status == 429


@pytest.mark.asyncio
async def test_stream_translates_and_ends_with_parseable_usage(monkeypatch):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 8}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    body = b"".join(b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
                    for e in events)

    def handler(request):
        assert json.loads(request.content).get("stream") is True
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    collected = bytearray()
    async for chunk in _adapter(handler, monkeypatch).chat_stream(
            "m", {"messages": [{"role": "user", "content": "hi"}], "stream": True}):
        collected.extend(chunk)

    text = collected.decode()
    assert text.rstrip().endswith("data: [DONE]")
    assert '"chat.completion.chunk"' in text
    assert parse_stream_usage(bytes(collected)) == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}


@pytest.mark.asyncio
async def test_stream_raises_provider_error_before_the_first_byte(monkeypatch):
    def handler(request):
        return httpx.Response(503, json={"error": {"message": "down"}})

    agen = _adapter(handler, monkeypatch).chat_stream(
        "m", {"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ProviderError) as exc:
        async for _ in agen:
            pass
    assert exc.value.kind == "http" and exc.value.status == 503
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_providers_anthropic.py -v`
Expected: the 4 new tests FAIL (the placeholder has no `chat`/`chat_stream`).
If `pytest.mark.asyncio` errors as unknown, check how `tests/test_providers.py` marks its async tests and follow that project convention.

- [ ] **Step 3: Implement (replace the placeholder body)**

```python
# gateway/providers_anthropic.py
"""Provider adapter for upstreams that speak the Anthropic messages protocol.

Exposes the SAME interface as gateway.providers.ProviderAdapter — chat() and
chat_stream() — so gateway/app.py is agnostic to which wire a provider uses.
All shape translation is delegated to gateway.anthropic_translate (pure).

The ProviderError timing contract is preserved: it is raised only BEFORE the
first byte reaches the caller, which is what makes the fallback chain safe.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

from gateway.anthropic_translate import (ANTHROPIC_VERSION, StreamTranslator,
                                         from_anthropic_response,
                                         iter_sse_data, to_anthropic_request)
from gateway.config import ProviderCfg
from gateway.providers import ProviderError


class AnthropicAdapter:
    def __init__(self, cfg: ProviderCfg, timeout_s: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)

    def _url(self) -> str:
        return f"{self.cfg.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": os.environ[self.cfg.api_key_env],
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json"}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, upstream_model: str, payload: dict) -> dict:
        body = to_anthropic_request(payload, upstream_model)
        try:
            resp = await self._client.post(self._url(), json=body,
                                           headers=self._headers())
        except httpx.TimeoutException as e:
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderError(self.cfg.name, "network") from e
        if not (200 <= resp.status_code < 300):
            raise ProviderError(self.cfg.name, "http", status=resp.status_code)
        return from_anthropic_response(resp.json(), upstream_model)

    async def chat_stream(self, upstream_model: str,
                          payload: dict) -> AsyncIterator[bytes]:
        body = to_anthropic_request(payload, upstream_model)
        body["stream"] = True
        translator = StreamTranslator(upstream_model)

        def _wire(chunk: dict) -> bytes:
            return b"data: " + json.dumps(chunk).encode() + b"\n\n"

        yielded = False
        try:
            async with self._client.stream("POST", self._url(), json=body,
                                           headers=self._headers()) as resp:
                if not (200 <= resp.status_code < 300):
                    await resp.aread()
                    raise ProviderError(self.cfg.name, "http",
                                        status=resp.status_code)
                buffer = b""
                async for raw in resp.aiter_bytes():
                    buffer += raw
                    # SSE events are separated by a blank line; keep any partial
                    # trailing event in the buffer until it completes.
                    while b"\n\n" in buffer:
                        block, buffer = buffer.split(b"\n\n", 1)
                        for event in iter_sse_data(block + b"\n\n"):
                            if event.get("type") == "error":
                                if not yielded:
                                    raise ProviderError(self.cfg.name, "http")
                                return
                            for chunk in translator.feed(event):
                                yielded = True
                                yield _wire(chunk)
                for chunk in translator.finish():
                    yielded = True
                    yield _wire(chunk)
                yield b"data: [DONE]\n\n"
        except httpx.TimeoutException as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "timeout") from e
        except httpx.TransportError as e:
            if yielded:
                raise
            raise ProviderError(self.cfg.name, "network") from e
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_providers_anthropic.py -v`
Expected: PASS (5 tests in the file, including Task 1's factory test).

- [ ] **Step 5: Whole suite — the OpenAI path must be untouched**

Run: `.venv/bin/pytest -q`
Expected: `278 passed`. Also confirm `git diff` shows **no change** to `ProviderAdapter` in `gateway/providers.py` (only the appended factory from Task 1).

- [ ] **Step 6: Commit**

```bash
git add gateway/providers_anthropic.py tests/test_providers_anthropic.py
git commit -m "feat(gateway): AnthropicAdapter over the pure translation layer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Serve glm-5.2 — config wiring + live smoke (PAID, human-gated)

**Files:**
- Modify: `configs/gateway.toml`
- Modify: `docs/M1_ACCEPTANCE.md` (append the smoke result)

- [ ] **Step 1: Add the Anthropic-wire GLM provider and model**

Append to `configs/gateway.toml` (keep the existing `deepseek` and `glm` providers and their models as they are):

```toml
[providers.glm_anthropic]
base_url = "https://open.bigmodel.cn/api/anthropic"
api_key_env = "GLM_API_KEY"
wire = "anthropic"

[models."glm-5.2"]
provider = "glm_anthropic"
upstream_model = "glm-5.2"
# GLM's paid balance is reachable only on the Anthropic-compatible endpoint;
# the same key returns 1113 "余额不足" on /paas/v4 (measured 2026-07-26).
in_usd_per_mtok = 0.60
out_usd_per_mtok = 2.20
fallback = ["deepseek-chat"]
```

Then verify the config loads:
Run: `.venv/bin/python -c "from pathlib import Path; from gateway.config import load_config; c=load_config(Path('configs/gateway.toml')); print(sorted(c.models), {n:p.wire for n,p in c.providers.items()})"`
Expected: `glm-5.2` present and `glm_anthropic` mapped to `wire='anthropic'`.

- [ ] **Step 2: Non-streaming live smoke (real key, ~$0.001)**

Start the gateway locally with real keys, then run `scripts/smoke.py` against it:

```bash
set -a; source runs/secrets/.env; set +a
export GATEWAY_TOKENS="smoke:smoketok,admin:admintok" \
       GATEWAY_CONFIG=configs/gateway.toml GATEWAY_DB=/tmp/gw_m7.sqlite
rm -f /tmp/gw_m7.sqlite
.venv/bin/uvicorn --factory gateway.app:create_app_from_env \
  --host 127.0.0.1 --port 8911 &
sleep 10
GATEWAY_URL=http://127.0.0.1:8911 GATEWAY_TOKEN=admintok .venv/bin/python scripts/smoke.py
```

Expected: **every** configured model returns 200 — including `glm-5.2` — and the printed `ledger consumed_usd delta` is **greater than zero** (proving usage translation feeds `settle()`).

- [ ] **Step 3: Streaming live smoke (real key)**

```bash
curl -sN http://127.0.0.1:8911/v1/chat/completions \
  -H "Authorization: Bearer smoketok" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","stream":true,"max_tokens":32,
       "messages":[{"role":"user","content":"count to three"}]}' | head -20
```

Expected: a stream of `data: {...}` lines whose objects are
`"chat.completion.chunk"`, ending with a usage-bearing chunk and `data: [DONE]`.
Confirm the text content reassembles into a sensible reply.

- [ ] **Step 4: Record the result and commit**

Append the smoke outcome (per-model status, latency, ledger delta, and a note that streaming was verified) to `docs/M1_ACCEPTANCE.md`, then:

```bash
git add configs/gateway.toml docs/M1_ACCEPTANCE.md
git commit -m "feat(gateway): serve glm-5.2 over the Anthropic wire

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** `wire` field + factory + `ConfigError` → Task 1 ✓ · request translation incl. system hoisting, `max_tokens` default 4096, tools/`input_schema`, `tool_choice`, `tool_calls`→`tool_use`, tool-result→user block → Task 2 ✓ · non-streaming response incl. `tool_calls`, `stop_reason` map, usage rename, thinking dropped → Task 3 ✓ · streaming event-by-event map incl. `input_json_delta` reassembly, thinking dropped, usage merged across `message_start`+`message_delta` → Task 4 ✓ · `ProviderError`-before-first-byte contract → Task 5 (test) ✓ · `parse_stream_usage` compatibility → Task 4 Step 5 + Task 5 test ✓ · `ProviderAdapter` untouched / 251 still pass → Global Constraints + Task 5 Step 5 ✓ · glm-5.2 served + live smoke incl. streaming and non-zero ledger → Task 6 ✓ · headers/URL → Task 5 test ✓.

**Placeholder scan:** none. Every code step carries complete code; Task 6's steps carry exact commands and expected outputs. The Task-1 placeholder class is explicitly temporary and replaced in Task 5.

**Type consistency:** `ProviderCfg(name, base_url, api_key_env, wire="openai")` is used identically in Tasks 1, 5, 6. `to_anthropic_request(payload, upstream_model)`, `from_anthropic_response(resp, model)`, `iter_sse_data(raw)`, `StreamTranslator(model)` with `.feed()`/`.finish()`/`.thinking_blocks` match between Tasks 2–4 and their consumer in Task 5. `_STOP_REASON_MAP` is defined in Task 2 and reused in Tasks 3–4. `make_adapter(cfg, transport=None)` matches its `app.py` call site.

**Test-count arithmetic:** 251 → 254 (T1: 2 config + 1 factory) → 262 (T2: 8) → 267 (T3: 5) → 274 (T4: 6 + 1 contract) → 278 (T5: 4).

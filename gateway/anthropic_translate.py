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
    string or a list of blocks. Strings pass through unchanged.

    Only `type: "text"` parts survive: vision is a stated non-goal for this
    gateway, so image/audio parts are dropped rather than translated."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
    return blocks or ""


def _system_text(content: Any) -> str:
    """Flatten one system message's content to a string.

    Anthropic's top-level `system` is a plain string, but OpenAI allows the
    array-of-parts shape here too — and `/v1/chat/completions` reads the body
    with no schema validation, so any client-controlled JSON reaches us. Parts
    that are not text contribute nothing instead of raising."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str) and part["text"]
        )
    return ""


def to_anthropic_request(payload: dict, upstream_model: str) -> dict:
    """Translate an OpenAI chat-completions body into an Anthropic messages body."""
    system_parts: list[str] = []
    messages: list[dict] = []

    for msg in payload.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_parts.append(_system_text(msg.get("content")))
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
    elif choice == "required":
        # OpenAI's "must call some tool" is Anthropic's {"type": "any"}.
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(choice, dict):
        name = choice.get("function", {}).get("name") or choice.get("name")
        if name:
            out["tool_choice"] = {"type": "tool", "name": name}
    return out


def _token_count(value: Any) -> int:
    """Coerce one upstream usage number to a non-negative int.

    The result feeds the ledger's settle(), so a garbage or absent count must
    become 0 rather than raise — a mis-typed field should not cost a request.

    OverflowError is in the tuple because `json.loads` accepts `1e400` and
    `Infinity` and returns `float("inf")`, which `int()` refuses to convert; it
    is a sibling of ValueError, not a subclass, so it needs naming."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def from_anthropic_response(resp: dict, model: str) -> dict:
    """Translate an Anthropic messages response into an OpenAI chat.completion.

    Upstream JSON is no more trusted than a client body: every block is shape-
    checked, and anything unrecognised is skipped instead of raising out of the
    request handler."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    thinking_blocks = 0

    content_blocks = resp.get("content")
    if not isinstance(content_blocks, list):
        content_blocks = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif btype == "tool_use":
            args = block.get("input")
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {"name": block.get("name", ""),
                             "arguments": json.dumps(args) if isinstance(args, dict) else "{}"},
            })
        elif btype == "thinking":
            thinking_blocks += 1  # counted, never surfaced (OpenAI has no field)

    # `content` is None rather than "" when there is no text: OpenAI clients
    # branch on `content is None` to recognise a tool-only reply.
    joined = "".join(text_parts)
    message: dict[str, Any] = {"role": "assistant", "content": joined or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = resp.get("stop_reason")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop") \
        if isinstance(stop_reason, str) else "stop"

    out: dict[str, Any] = {
        "id": resp.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
    }

    # Same rule as StreamTranslator.finish(): the `usage` key exists only when
    # the upstream actually STATED both counts. app.py branches on key presence
    # and settles a present usage dict as usage_source="reported", so a
    # manufactured {0, 0} would record a real, billed call as $0.00
    # authoritative -- the budget guard would see no consumption and the
    # killswitch would fail open. With no key, app.py degrades to an estimate,
    # exactly like the OpenAI path. An explicit 0 still counts as stated, and a
    # mis-typed count was still counted: _token_count coerces it to 0.
    usage_in = resp.get("usage")
    if not isinstance(usage_in, dict):
        usage_in = {}
    raw_in = usage_in.get("input_tokens")
    raw_out = usage_in.get("output_tokens")
    if raw_in is not None and raw_out is not None:
        prompt = _token_count(raw_in)
        completion = _token_count(raw_out)
        out["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                        "total_tokens": prompt + completion}
    return out


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


def _as_text(value: Any) -> str:
    """An upstream string field, or "" when it is absent or not a string.

    Every delta value the client concatenates has to be a string: a dict where
    a fragment was expected would break the client's own reassembly, not ours."""
    return value if isinstance(value, str) else ""


def _block_key(index: Any) -> Any:
    """Anthropic's content-block index, made safe to use as a dict key.

    A missing index collapses to -1 so a stream that omits it still pairs a
    block's start with its deltas; an unhashable one yields None, which is
    never registered and never matches — an ignored fragment, not a TypeError."""
    if index is None:
        return -1
    try:
        hash(index)
    except TypeError:
        return None
    return index


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
        # Whether the stream ever *stated* each count. "Never counted" and
        # "counted zero" are different answers, and finish() must not turn the
        # first into the second -- see the note there.
        self._input_reported = False
        self._output_reported = False
        self._finish_reason: str | None = None
        self._tool_index: dict[Any, int] = {}   # anthropic block idx -> tool_calls idx
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
        if not isinstance(event, dict):
            return []       # untrusted-input boundary: don't rely on the caller
        etype = event.get("type")
        out: list[dict] = []

        if etype == "message_start":
            message = event.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                # Same rule as message_delta below: only a STATED count writes.
                # An unconditional write lets a usage-less (or duplicate, or
                # out-of-order) message_start zero a count already banked, and
                # the stream still reports it as authoritative.
                if usage.get("input_tokens") is not None:
                    self._input_reported = True
                    self.input_tokens = _token_count(usage.get("input_tokens"))
            if not self._role_sent:
                self._role_sent = True
                out.append(self._chunk({"role": "assistant"}))

        elif etype == "content_block_start":
            block = event.get("content_block")
            if not isinstance(block, dict):
                block = {}
            if block.get("type") == "tool_use":
                idx = self._next_tool_index
                self._next_tool_index += 1
                key = _block_key(event.get("index"))
                if key is not None:
                    self._tool_index[key] = idx
                out.append(self._chunk({"tool_calls": [{
                    "index": idx, "id": _as_text(block.get("id")), "type": "function",
                    "function": {"name": _as_text(block.get("name")), "arguments": ""},
                }]}))
            elif block.get("type") == "thinking":
                self.thinking_blocks += 1

        elif etype == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                out.append(self._chunk({"content": _as_text(delta.get("text"))}))
            elif dtype == "input_json_delta":
                key = _block_key(event.get("index"))
                idx = self._tool_index.get(key) if key is not None else None
                if idx is not None:      # unknown block index -> ignore, not fatal
                    out.append(self._chunk({"tool_calls": [{
                        "index": idx, "type": "function",
                        "function": {"arguments": _as_text(delta.get("partial_json"))},
                    }]}))
            elif dtype == "thinking_delta":
                pass                      # dropped: OpenAI has no field for it

        elif etype == "message_delta":
            usage = event.get("usage")
            if isinstance(usage, dict):
                # Anthropic proper puts input_tokens on message_start only, but
                # a clone may restate BOTH counts here. Reading output alone
                # would leave such a stream looking "never counted" on the input
                # side, so finish() withholds the usage chunk and a real call is
                # billed by estimate with no error anywhere. Only a STATED count
                # overwrites: a later message_delta that omits input_tokens must
                # not erase the number banked at message_start.
                if usage.get("input_tokens") is not None:
                    self._input_reported = True
                    self.input_tokens = _token_count(usage.get("input_tokens"))
                if usage.get("output_tokens") is not None:
                    self._output_reported = True
                    self.output_tokens = _token_count(usage.get("output_tokens"))
            delta = event.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if isinstance(reason, str) and reason:
                self._finish_reason = _STOP_REASON_MAP.get(reason, "stop")

        return out

    def finish(self) -> list[dict]:
        """Terminal chunks: finish_reason, then a usage-bearing chunk.

        The usage chunk is what gateway.providers.parse_stream_usage() finds,
        which is how settle() bills real tokens for an Anthropic upstream.

        It is emitted only when the stream actually stated BOTH counts. A
        stream that ends before message_delta never stated an output count, and
        app.py consumes the usage dict wholesale as usage_source="reported" --
        so a half-report would bill a confident zero for one direction. With no
        usage chunk, parse_stream_usage returns None and app.py falls back to an
        estimate, which is how the OpenAI path already behaves when usage is
        missing. An explicit zero still counts as stated, and is still reported.
        """
        out = [self._chunk({}, finish_reason=self._finish_reason or "stop")]
        if not (self._input_reported and self._output_reported):
            return out
        usage_chunk = {
            "id": self.id, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model, "choices": [],
            "usage": {"prompt_tokens": self.input_tokens,
                      "completion_tokens": self.output_tokens,
                      "total_tokens": self.input_tokens + self.output_tokens},
        }
        out.append(usage_chunk)
        return out

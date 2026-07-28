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

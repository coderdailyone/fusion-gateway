"""Structural comparison of OpenAI tool calls.

This module is why tool calls can be fused at all. M5 and M6 both measured a
+0.7pt ceiling over the best pool member, and the M6 report named the cause: on
prose there is no answer extractor, so agreement can only be *judged* by an
LLM, and judgement is noisy. A tool call carries its own extractor -- `name`
plus JSON `arguments` -- so deciding whether the models agree costs nothing and
involves no model.

Pure: no IO, no network, no gateway imports.

THE RULE THAT HOLDS THE REST UP: `canonical_calls` returns None for an empty or
unusable list, and None never matches anything -- not even another None. A
text-only candidate has `tool_calls == []`; if that canonicalised to `()`, two
prose candidates would compare equal and prose would be routed through the tool
path. Two models failing to produce parseable arguments is likewise not
agreement.
"""
from __future__ import annotations

import json

CanonCall = tuple[str, str]      # (tool name, canonical arguments JSON)


def _canonical_one(call) -> CanonCall | None:
    if not isinstance(call, dict):
        return None
    # M9 final review, finding 4 (Minor): only `function.name` was read
    # below, so a call shaped like OpenAI's `type: "custom"` --
    # {"type": "custom", "custom": {"name": "bash", ...}, "function":
    # {"name": "read", ...}} -- classified (and forwarded) as "read" purely
    # from the `function` block, even though a client dispatching on `type`
    # would execute the `custom` block's "bash" instead. No current upstream
    # emits this shape, but nothing enforced `type` before trusting
    # `function.name`. `type` omitted entirely is still accepted, matching
    # OpenAI's own wire format (an absent `type` defaults to "function")
    # rather than tightening this into a stricter gate than the spec's own
    # one-line fix calls for.
    if call.get("type") not in (None, "function"):
        return None
    fn = call.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "arguments" in fn:
        raw = fn.get("arguments")
    else:
        raw = "{}"
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return (name, canon)


def canonical_calls(tool_calls) -> tuple[CanonCall, ...] | None:
    """Canonical, order-insensitive form of a candidate's calls.

    Returns None when there is nothing comparable: no calls at all, or any
    single call unusable. Sorted, so ordering differences between models do not
    read as disagreement -- but it is a multiset, not a set, so a duplicated
    call still differs from a single one.
    """
    if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
        return None
    out: list[CanonCall] = []
    for call in tool_calls:
        one = _canonical_one(call)
        if one is None:
            return None
        out.append(one)
    return tuple(sorted(out))


def plurality(by_model: dict[str, tuple[CanonCall, ...] | None]) -> str | None:
    """A model whose canonical calls at least one other model also produced.

    Returns None when no two agree. Deterministic: models are considered in
    sorted name order, so a tie does not depend on dict iteration order.
    """
    usable = {m: c for m, c in sorted(by_model.items()) if c is not None}
    counts: dict[tuple[CanonCall, ...], list[str]] = {}
    for model, canon in usable.items():
        counts.setdefault(canon, []).append(model)
    for canon, models in counts.items():
        if len(models) >= 2:
            return models[0]
    return None


def all_readonly(canon: tuple[CanonCall, ...] | None,
                 readonly: frozenset[str]) -> bool:
    """True only when every call names a tool on the read-only list.

    Default-deny: an unlisted tool is write-class, so a new tool is reviewed
    rather than waved through. Exact, case-sensitive name matching -- a prefix
    rule like `read*` would silently admit a future `readwrite`.
    """
    if not canon:
        return False
    return all(name in readonly for name, _ in canon)

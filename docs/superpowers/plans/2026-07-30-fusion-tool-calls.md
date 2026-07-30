# Tool Calls Through the Fusion Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a request carrying `tools` reach the fusion panel instead of bypassing it, deciding agreement structurally so the common agent step costs two upstream calls and ~2 s.

**Architecture:** A new pure module compares tool calls by canonical `(name, arguments)` — no LLM needed to decide agreement, which is what made this tractable. `PanelResult.candidates` grows from `dict[str, str]` to `dict[str, Candidate]` so a candidate can carry tool calls. The orchestrator gains a decision tree: identical read-only calls emit immediately; write-class calls keep the cross-review; disagreement is arbitrated by the third member's vote before the LLM fuser is consulted.

**Tech Stack:** Python 3.10, FastAPI, httpx (`MockTransport` in tests), asyncio, pytest, SQLite.

## Global Constraints

- **The prose fusion path and the single-model path must behave identically.** A `Candidate` with empty `tool_calls` must behave exactly as the bare string did, and prompts generated from text-only candidates must be **byte-identical** to today's. The existing 422 tests are the guard.
- A ledger row must **never** be left in `preflight` — a consuming state cleared only at startup.
- A cancelled call is `settle`d with `usage_source="estimated"`, never `fail`ed.
- Fusion must **never** return a 5xx the gateway itself produced.
- `gateway/` must not import `evaluator/` or `router/`.
- `gateway/providers.py`, `gateway/providers_anthropic.py`, `gateway/anthropic_translate.py`, `gateway/ledger.py`, `gateway/db.py`, `gateway/events.py` are **unchanged**. glm-5.2's Anthropic-wire tool translation already works and is reused as-is.
- Comparison is **exact after canonicalisation**, never semantic. Tool-name matching is exact and case-sensitive — no prefix or regex rules.
- Classification is **default-deny**: a tool absent from `readonly_tools` is write-class.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Run `.venv/bin/python -m pytest tests/ -q` — the **whole** suite — after every task. `tests/test_config.py`, `tests/test_policy.py`, `tests/test_app.py` and `tests/test_streaming.py` all read the real `configs/gateway.toml`; editing it has broken distant tests twice in this project.
- **Assert ledger row counts per branch, never as a global range.** The M8 spec claimed "5–7" and the final review proved it wrong.
- **Every new test needs red→green mutation evidence.** This milestone has already shipped two decorative tests that passed against deliberately broken code. Apply the break, watch the test fail, revert, watch it pass, and put both outputs in the task report.

## File Structure

| file | responsibility |
|---|---|
| `gateway/tool_vote.py` (new) | pure: canonicalise calls, compare, plurality, read-only classification |
| `gateway/fusion.py` (modify) | `Candidate` type, `_extract_message`, the tool decision tree |
| `gateway/fusion_prompts.py` (modify) | render a tool-call candidate; the fuser's action rule |
| `gateway/config.py` (modify) | `readonly_tools` + validation |
| `gateway/app.py` (modify) | remove the bypass; tool-call response and synthesised stream |
| `configs/gateway.toml` (modify) | `readonly_tools` |
| `tests/test_tool_vote.py` (new) | Task 1 |
| `tests/test_fusion_tools.py` (new) | Tasks 3, 5 |
| `tests/test_app_fusion.py` (modify) | Task 6 — five bypass tests change meaning |

---

### Task 1: The pure comparison core

**Files:**
- Create: `gateway/tool_vote.py`
- Test: `tests/test_tool_vote.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CanonCall = tuple[str, str]`; `canonical_calls(tool_calls) -> tuple[CanonCall, ...] | None`; `plurality(by_model: dict[str, tuple[CanonCall, ...] | None]) -> str | None`; `all_readonly(canon: tuple[CanonCall, ...], readonly: frozenset[str]) -> bool`.

**Context.** This module knows nothing about the gateway — it takes raw OpenAI `tool_calls` lists. The single most important design point: **`canonical_calls` returns `None` for an empty or unusable list.** `None` never matches anything, including another `None`. Without that, two text-only candidates (both with `tool_calls == []`) would canonicalise to the same empty tuple and be declared in agreement, silently routing prose through the tool path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_vote.py
import pytest
from gateway.tool_vote import canonical_calls, plurality, all_readonly

READONLY = frozenset({"read", "ls", "grep", "find"})


def call(name, args):
    return {"id": "x", "type": "function",
            "function": {"name": name, "arguments": args}}


def test_key_order_and_whitespace_do_not_matter():
    a = canonical_calls([call("read", '{"path":"a.py","limit":10}')])
    b = canonical_calls([call("read", '{ "limit": 10 , "path" : "a.py" }')])
    assert a == b is not None


def test_comparison_is_exact_not_semantic():
    # "a.py" and "./a.py" name the same file to a human. Treating them as
    # equal would need an LLM, which is the cost this module exists to avoid.
    a = canonical_calls([call("read", '{"path":"a.py"}')])
    b = canonical_calls([call("read", '{"path":"./a.py"}')])
    assert a != b


def test_a_different_tool_name_never_matches():
    assert canonical_calls([call("read", "{}")]) != canonical_calls([call("write", "{}")])


def test_unparseable_arguments_are_unusable():
    # Unusable must be None, not some sentinel that could match another
    # unusable call -- two models failing differently is not agreement.
    assert canonical_calls([call("read", "{not json")]) is None
    assert canonical_calls([call("read", None)]) is None
    assert plurality({"a": None, "b": None, "c": None}) is None


def test_an_empty_call_list_is_unusable():
    # THE load-bearing case: a text-only candidate has tool_calls == [].
    # If that canonicalised to (), two prose candidates would "agree" and
    # prose would be routed through the tool path.
    assert canonical_calls([]) is None
    assert canonical_calls(None) is None
    assert plurality({"a": canonical_calls([]), "b": canonical_calls([])}) is None


def test_malformed_shapes_never_raise():
    for bad in ("notalist", [None], [{}], [{"function": None}],
                [{"function": {"name": 5, "arguments": "{}"}}],
                [{"function": {"arguments": "{}"}}]):
        assert canonical_calls(bad) is None


def test_parallel_calls_ignore_order_but_not_duplication():
    one = canonical_calls([call("read", '{"path":"a"}'), call("read", '{"path":"b"}')])
    two = canonical_calls([call("read", '{"path":"b"}'), call("read", '{"path":"a"}')])
    assert one == two is not None
    dup = canonical_calls([call("read", '{"path":"a"}'), call("read", '{"path":"a"}')])
    single = canonical_calls([call("read", '{"path":"a"}')])
    assert dup != single


def test_plurality_returns_a_two_of_three_winner():
    same = canonical_calls([call("read", '{"path":"a"}')])
    other = canonical_calls([call("write", '{"path":"a"}')])
    winner = plurality({"m1": same, "m2": other, "m3": same})
    assert winner in ("m1", "m3")


def test_plurality_is_none_on_a_three_way_split():
    got = plurality({"m1": canonical_calls([call("read", '{"path":"a"}')]),
                     "m2": canonical_calls([call("read", '{"path":"b"}')]),
                     "m3": canonical_calls([call("read", '{"path":"c"}')])})
    assert got is None


def test_plurality_is_deterministic():
    # Two models tie; the winner must not depend on dict iteration order.
    same = canonical_calls([call("read", '{"path":"a"}')])
    first = plurality({"b": same, "a": same})
    second = plurality({"a": same, "b": same})
    assert first == second


def test_all_readonly_is_exact_and_default_deny():
    assert all_readonly(canonical_calls([call("read", "{}")]), READONLY)
    assert not all_readonly(canonical_calls([call("write", "{}")]), READONLY)
    # An unlisted tool -- a new Pi tool, or another client's -- is write-class.
    assert not all_readonly(canonical_calls([call("brand_new_tool", "{}")]), READONLY)
    # No prefix matching: "read" being listed must not admit "readwrite".
    assert not all_readonly(canonical_calls([call("readwrite", "{}")]), READONLY)
    # Case-sensitive.
    assert not all_readonly(canonical_calls([call("Read", "{}")]), READONLY)
    # A mixed batch is write-class: one unsafe call taints the whole step.
    mixed = canonical_calls([call("read", "{}"), call("write", "{}")])
    assert not all_readonly(mixed, READONLY)


def test_all_readonly_rejects_an_unusable_batch():
    assert not all_readonly(None, READONLY)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tool_vote.py -q`
Expected: collection error — `No module named 'gateway.tool_vote'`.

- [ ] **Step 3: Write the implementation**

```python
# gateway/tool_vote.py
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
    fn = call.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    raw = fn.get("arguments")
    if raw is None:
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tool_vote.py -q`
Expected: 12 passed.

- [ ] **Step 5: Prove two tests bite**

Mutate `canonical_calls` to return `()` instead of `None` for an empty list; confirm `test_an_empty_call_list_is_unusable` FAILS. Revert. Mutate `all_readonly` to `any(...)` instead of `all(...)`; confirm the mixed-batch assertion FAILS. Revert. Record both outputs in the report.

- [ ] **Step 6: Run the whole suite and commit**

Run: `.venv/bin/python -m pytest tests/ -q` → 434 passed (422 + 12).

```bash
git add gateway/tool_vote.py tests/test_tool_vote.py
git commit -m "feat(gateway): structural comparison of tool calls

A tool call carries its own extractor, so deciding whether models agree
costs nothing and involves no LLM -- which is what makes fusing tool
calls tractable at all. Empty or unusable call lists canonicalise to
None and never match, so prose cannot leak into the tool path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `readonly_tools` config

**Files:**
- Modify: `gateway/config.py`
- Modify: `configs/gateway.toml`
- Test: `tests/test_fusion_config.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `FusionCfg.readonly_tools: frozenset[str]`.

**Context.** Read `gateway/config.py`'s existing `[fusion]` block first — it already validates panel/quorum/reviewers/fuser membership, subset relations, non-emptiness, list types and duplicates, and raises `ConfigError` naming the field. Match that style exactly. `FusionCfg` is a frozen dataclass; add the field with a default so existing construction sites keep working.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_fusion_config.py

def test_readonly_tools_defaults_to_pis_read_only_set(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    assert cfg.fusion.readonly_tools == frozenset({"read", "ls", "grep", "find"})


def test_readonly_tools_can_be_overridden(tmp_path):
    text = BASE.replace("[fusion]", '[fusion]\nreadonly_tools = ["cat", "stat"]')
    assert load_config(write(tmp_path, text)).fusion.readonly_tools == frozenset({"cat", "stat"})


def test_readonly_tools_may_be_empty_meaning_review_everything(tmp_path):
    # An explicit empty list is a legitimate, maximally-cautious choice:
    # every tool call gets the cross-review.
    text = BASE.replace("[fusion]", "[fusion]\nreadonly_tools = []")
    assert load_config(write(tmp_path, text)).fusion.readonly_tools == frozenset()


@pytest.mark.parametrize("value", [
    '"read"',                    # a bare string would iterate as characters
    '["read", "read"]',          # duplicates
    '["read", ""]',              # empty name
    '["read", 5]',               # non-string entry
])
def test_bad_readonly_tools_is_rejected(tmp_path, value):
    text = BASE.replace("[fusion]", f"[fusion]\nreadonly_tools = {value}")
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_the_real_config_declares_readonly_tools(tmp_path):
    cfg = load_config(REAL)
    assert "read" in cfg.fusion.readonly_tools
    assert "write" not in cfg.fusion.readonly_tools   # write-class by design
    assert "bash" not in cfg.fusion.readonly_tools    # can rm -rf; must be reviewed
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_config.py -q`
Expected: FAIL — `FusionCfg` has no attribute `readonly_tools`.

- [ ] **Step 3: Add the field**

In `gateway/config.py`, add to `FusionCfg` (last, with a default):

```python
    readonly_tools: frozenset[str] = frozenset({"read", "ls", "grep", "find"})
```

- [ ] **Step 4: Parse and validate it**

Inside the `if "fusion" in data:` block, after the existing checks and before `FusionCfg(...)` is constructed:

```python
        if "readonly_tools" in f:
            raw_ro = f["readonly_tools"]
            if not isinstance(raw_ro, list):
                raise ConfigError(
                    f"fusion.readonly_tools must be a list, got "
                    f"{type(raw_ro).__name__}"
                )
            for entry in raw_ro:
                if not isinstance(entry, str) or not entry:
                    raise ConfigError(
                        f"fusion.readonly_tools entries must be non-empty "
                        f"strings, got {entry!r}"
                    )
            if len(set(raw_ro)) != len(raw_ro):
                raise ConfigError("fusion.readonly_tools has duplicate entries")
            readonly_tools = frozenset(raw_ro)
        else:
            readonly_tools = frozenset({"read", "ls", "grep", "find"})
```

and pass `readonly_tools=readonly_tools` to the `FusionCfg(...)` call.

- [ ] **Step 5: Declare it in the real config**

In `configs/gateway.toml`, inside `[fusion]`, after `stage_timeout_s`:

```toml
# Tools whose calls may be emitted on structural agreement alone, with no
# cross-review. Everything NOT listed is write-class and keeps the review even
# when the models agree, so a new or unknown tool is reviewed rather than waved
# through. These four are Pi's read-only tools; `bash` is deliberately absent
# because it can `ls -la` or `rm -rf` and the name cannot tell you which.
readonly_tools = ["read", "ls", "grep", "find"]
```

- [ ] **Step 6: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_fusion_config.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → zero failures. The real config changed, so distant modules that read it must be checked, not assumed.

- [ ] **Step 7: Commit**

```bash
git add gateway/config.py configs/gateway.toml tests/test_fusion_config.py
git commit -m "feat(gateway): readonly_tools, default-deny

A positive list of known-safe tool names. Anything absent is write-class
and keeps the cross-review, so a new tool is reviewed rather than waved
through. Exact case-sensitive matching only -- a prefix rule like read*
would silently admit a future readwrite.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `Candidate` — candidates that can carry tool calls

**Files:**
- Modify: `gateway/fusion.py`
- Test: `tests/test_fusion_tools.py` (new)

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `Candidate(text: str, tool_calls: tuple[dict, ...] = ())` (frozen dataclass); `_extract_message(resp) -> Candidate`; `PanelResult.candidates: dict[str, Candidate]`; `best_candidate(fcfg, panel) -> tuple[str, Candidate] | None`; `openai_response(candidate: Candidate, model: str, meta: dict) -> dict`.

**Context — this is the risky task.** `PanelResult.candidates` is `dict[str, str]` today and five things read it: `is_consensus` (keys only — unaffected), `build_review_prompt`, `build_fusion_prompt`, `best_candidate`, and `openai_response`. `gateway/app.py` also unpacks `best_candidate(...)` and passes its second element to `_as_chunks`.

**The bar: the prose path must not change at all.** A `Candidate("hi")` must render, fall back and serialise exactly as `"hi"` did. The 422 existing tests are the guard — if any of them needs editing to accommodate this task, that is a signal the refactor broke something, not that the test was wrong. Read `gateway/fusion.py` in full before editing.

Do **not** change orchestration logic in this task — only the type and the code that reads it. The decision tree is Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fusion_tools.py
import json
import pytest
from gateway.fusion import Candidate, _extract_message, openai_response


def _resp(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def test_extract_message_reads_plain_text():
    got = _extract_message(_resp(content="hello"))
    assert got == Candidate("hello", ())


def test_extract_message_reads_a_tool_call():
    # This is the root cause of the original CRITICAL: the old _extract_text
    # returned "" here, every candidate was dropped, and a fully-billed panel
    # handed back a 502.
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]
    got = _extract_message(_resp(content=None, tool_calls=calls))
    assert got.text == "" and len(got.tool_calls) == 1
    assert got.tool_calls[0]["function"]["name"] == "read"


def test_extract_message_reads_text_and_a_call_together():
    calls = [{"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}]
    got = _extract_message(_resp(content="let me look", tool_calls=calls))
    assert got.text == "let me look" and len(got.tool_calls) == 1


def test_extract_message_survives_hostile_shapes():
    for resp in ({}, {"choices": []}, {"choices": [{}]}, {"choices": "x"},
                 _resp(content=None, tool_calls="notalist"),
                 _resp(content=None, tool_calls=[None])):
        got = _extract_message(resp)
        assert isinstance(got, Candidate)


def test_openai_response_for_text_is_unchanged_in_shape():
    r = openai_response(Candidate("hi"), "fusion", {"path": "quorum"})
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in r["choices"][0]["message"]


def test_openai_response_for_a_tool_call():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    r = openai_response(Candidate("", calls), "fusion", {})
    msg = r["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "read"
    assert r["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_response_keeps_text_alongside_a_call():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    msg = openai_response(Candidate("looking", calls), "fusion", {})["choices"][0]["message"]
    assert msg["content"] == "looking" and len(msg["tool_calls"]) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_tools.py -q`
Expected: ImportError — `Candidate` / `_extract_message` do not exist.

- [ ] **Step 3: Add the type and the extractor**

In `gateway/fusion.py`, beside `PanelResult`:

```python
@dataclass(frozen=True)
class Candidate:
    """One panel member's answer: prose, tool calls, or both.

    Candidates used to be bare strings, which is why a tool call made
    `_extract_text` return "" and got the candidate dropped -- a fully-billed
    panel then returned 502 (M8 final review, finding 1a). A Candidate with
    empty `tool_calls` must behave exactly as the string did.
    """
    text: str
    tool_calls: tuple[dict, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text or self.tool_calls)
```

and, next to `_extract_text` (which stays — it is still the right tool for the review and fuser calls, which return prose):

```python
def _extract_message(resp) -> Candidate:
    """Pull text AND tool calls out of an upstream response, defensively.

    The response is upstream-controlled, so every access is guarded: a raw
    index would turn a malformed 200 into a gateway 500.
    """
    text = _extract_text(resp)
    calls: tuple[dict, ...] = ()
    try:
        raw = resp["choices"][0]["message"].get("tool_calls")
    except (TypeError, KeyError, IndexError, AttributeError):
        raw = None
    if isinstance(raw, (list, tuple)):
        calls = tuple(c for c in raw if isinstance(c, dict))
    return Candidate(text, calls)
```

- [ ] **Step 4: Make the readers Candidate-aware**

`openai_response` — emit either shape:

```python
def openai_response(candidate: Candidate, model: str, meta: dict) -> dict:
    message: dict = {"role": "assistant"}
    if candidate.tool_calls:
        # OpenAI sends content: null alongside tool_calls unless the model also
        # produced prose; finish_reason must be "tool_calls" or clients will not
        # execute the call.
        message["content"] = candidate.text or None
        message["tool_calls"] = list(candidate.tool_calls)
        finish = "tool_calls"
    else:
        message["content"] = candidate.text
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "fusion": meta,
    }
```

`_candidate_block` in `gateway/fusion_prompts.py` and `build_*_prompt` are Task 4 — for now, make them accept a `Candidate` and render `c.text`, so text-only output is byte-identical.

`best_candidate` — return the `Candidate`, and keep the truthiness guard both loops need:

```python
def best_candidate(fcfg, panel: PanelResult):
    """The answer to fall back on when the fuser fails: the first surviving
    member in configured panel order. Returns (model, Candidate) or None."""
    for m in fcfg.panel:
        c = panel.candidates.get(m)
        if c:
            return m, c
    for m in sorted(panel.candidates):
        c = panel.candidates[m]
        if c:
            return m, c
    return None
```

In `call_model`, return `_extract_message(resp)` instead of `_extract_text(resp)` for `kind == "candidate"`; leave the review and fuser paths returning text. In `collect()`, the `if text:` guard becomes a truthiness check on the `Candidate` (which `__bool__` above makes correct).

In `gateway/app.py`, the two `best_candidate(...)` call sites unpack a `Candidate`: pass `fallback[1]` to `openai_response`, and `fallback[1].text` to `_as_chunks` for now (the tool-call stream is Task 6).

- [ ] **Step 5: Run the new tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_fusion_tools.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → **zero failures, and no existing test edited.** If an existing test fails, the refactor changed prose behaviour — fix the code, not the test.

- [ ] **Step 6: Commit**

```bash
git add gateway/fusion.py gateway/fusion_prompts.py gateway/app.py tests/test_fusion_tools.py
git commit -m "refactor(gateway): candidates carry tool calls, not just text

A candidate is now a Candidate(text, tool_calls) rather than a bare
string. This is the root fix for M8 final-review finding 1a: a tool call
made the old extractor return \"\", every candidate was dropped, and a
fully-billed panel returned 502. The prose path is untouched -- a
Candidate with empty tool_calls renders and serialises exactly as the
string did.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Prompts that can show a tool call

**Files:**
- Modify: `gateway/fusion_prompts.py`
- Test: `tests/test_fusion_prompts.py` (append)

**Interfaces:**
- Consumes: `Candidate` from Task 3.
- Produces: `render_candidate(c: Candidate) -> str`; `build_review_prompt`/`build_fusion_prompt` unchanged in signature, now accepting `dict[str, Candidate]`.

**Context.** A reviewer must be able to judge an action, not just prose. The `VERDICT <target> <correct|wrong|unsure> <reason>` format needs no change — "is this the right tool with the right arguments" fits it directly. The fusion prompt gains one rule for the three-way-split case.

**Byte-identical requirement:** for a text-only `Candidate`, both prompts must be exactly what they are today. There is already a guard test asserting no benchmark scaffolding leaked in; add one asserting prose-prompt stability.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_fusion_prompts.py
from gateway.fusion import Candidate

TOOL = ({"id": "c1", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"a.py"}'}},)


def test_a_tool_call_candidate_renders_its_name_and_arguments():
    p = build_review_prompt("Q", {"m1": Candidate("", TOOL),
                                  "m2": Candidate("prose")}, reviewer="m2")
    assert "read" in p and '"path"' in p and "a.py" in p


def test_a_reviewer_still_never_sees_its_own_tool_call():
    p = build_review_prompt("Q", {"m1": Candidate("", TOOL),
                                  "m2": Candidate("", TOOL)}, reviewer="m1")
    assert "--- Candidate m1 ---" not in p


def test_the_fusion_prompt_tells_the_fuser_to_act_not_narrate():
    p = build_fusion_prompt("Q", {"m1": Candidate("", TOOL)}, {})
    low = p.lower()
    assert "tool" in low and ("call" in low or "action" in low)


def test_prose_prompts_are_byte_identical_to_the_string_era():
    # The prose path must not shift by a single character. These are the exact
    # strings the pre-Candidate implementation produced.
    cands = {"a": Candidate("first"), "b": Candidate("second")}
    review = build_review_prompt("CONV", cands, reviewer="b")
    assert "--- Candidate a ---\nfirst" in review
    assert "--- Candidate b ---" not in review
    fusion = build_fusion_prompt("CONV", cands, {})
    assert "--- Candidate a ---\nfirst\n\n--- Candidate b ---\nsecond" in fusion
    assert "(no reviews available)" in fusion
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_prompts.py -q`
Expected: FAIL — the tool call does not render; the action rule is absent.

- [ ] **Step 3: Render candidates**

In `gateway/fusion_prompts.py`:

```python
def render_candidate(c) -> str:
    """Render a candidate for a reviewer or the fuser.

    Text-only candidates render as their bare text, byte-identical to when
    candidates were strings -- the prose path must not shift.
    """
    text = getattr(c, "text", c) or ""
    calls = getattr(c, "tool_calls", ()) or ()
    if not calls:
        return text
    lines = []
    if text:
        lines.append(text)
    for call in calls:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name", "?")
        args = fn.get("arguments", "")
        lines.append(f"TOOL_CALL {name} {args}")
    return "\n".join(lines)
```

and use it in `_candidate_block`:

```python
        parts.append(f"--- Candidate {model} ---\n{render_candidate(text)}")
```

- [ ] **Step 4: Add the fuser's action rule**

Append to the `rules` list in `build_fusion_prompt`, before the final
"Reply with the answer itself" rule:

```python
        "- If the candidates proposed tool calls (shown as TOOL_CALL lines), "
        "the conversation calls for an action, not an explanation. Choose the "
        "correct call, or state the corrected one in the same TOOL_CALL form. "
        "Do not answer in prose instead of acting.",
```

- [ ] **Step 5: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_fusion_prompts.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → zero failures.

- [ ] **Step 6: Prove the byte-identical test bites**

Change `render_candidate`'s text-only branch to `return text.strip() + " "`; confirm `test_prose_prompts_are_byte_identical_to_the_string_era` FAILS. Revert. Record both outputs.

- [ ] **Step 7: Commit**

```bash
git add gateway/fusion_prompts.py tests/test_fusion_prompts.py
git commit -m "feat(gateway): render tool calls in review and fusion prompts

A reviewer judges an action the same way it judges prose -- the VERDICT
format needed no change. The fuser gains one rule: when the candidates
proposed calls, act rather than narrate. Text-only candidates render
byte-identically to the string era, pinned by a test.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: The decision tree

**Files:**
- Modify: `gateway/fusion.py`
- Test: `tests/test_fusion_tools.py` (append)

**Interfaces:**
- Consumes: Task 1's `canonical_calls`/`plurality`/`all_readonly`; Task 2's `FusionCfg.readonly_tools`; Task 3's `Candidate`.
- Produces: `PanelResult.path` gains the values `"tool_fast"`, `"tool_reviewed"`, `"tool_plurality"`; `fuser_body(fcfg, panel, body)` forwards `tools`/`tool_choice` when the panel holds tool calls.

**Context.** Read `gather_panel` in full first. Today it collects the quorum, cross-reviews, checks `is_consensus`, and either short-circuits or waits out the slow leg. The tool branches slot in **after the quorum is collected and before the review is dispatched**, because the whole point is that a read-only agreement needs no review.

Rules carried over verbatim from the prose path, not reinvented: a reviewer objection cancels a copy, and a **missing** review is not agreement.

- [ ] **Step 1: Write the failing tests**

Reuse `tests/test_fusion.py`'s existing fixtures by importing them — `FCFG`
(line 22), `FakeCfg`, `FakeAdapter`, `make_env` (line 57) and `BODY` (line 70).
**`make_env(tmp_path, adapter, fcfg=FCFG)` already takes an `fcfg` override**, so
a test needing a different `readonly_tools` builds one with
`dataclasses.replace(FCFG, readonly_tools=frozenset({"read"}))` and passes it in
rather than rebuilding the environment. Do not duplicate the fixtures.

```python
# append to tests/test_fusion_tools.py
import asyncio
from gateway.fusion import gather_panel, fuser_body, PanelResult
from tests.test_fusion import FakeAdapter, FakeCfg, make_env, BODY, FCFG  # fixtures

TOOLS_BODY = dict(BODY, tools=[{"type": "function",
                                "function": {"name": "read", "parameters": {}}}])


def tool_resp(name, args):
    return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": name, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_identical_readonly_calls_emit_with_no_review_and_no_slow_leg(tmp_path):
    ad = FakeAdapter({m: (lambda p: tool_resp("read", '{"path":"a.py"}'))
                      for m in ("a", "b", "s")}, delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_fast"
    assert panel.reviews == {}                      # no review was dispatched
    rows = store.conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len([r for r in rows if r["model"] in ("a", "b")]) == 2
    assert not any(r["state"] == "preflight" for r in rows)


@pytest.mark.anyio
async def test_identical_write_class_calls_are_reviewed(tmp_path):
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a correct ok\nVERDICT b correct ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py","body":"x"}')
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_reviewed" and panel.reviews != {}


@pytest.mark.anyio
async def test_a_reviewer_objection_on_a_write_class_call_escalates(tmp_path):
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a wrong no\nVERDICT b wrong no"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return tool_resp("write", '{"path":"a.py"}')
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path != "tool_reviewed"            # did not copy on objection
    assert "s" in panel.candidates                  # escalated to the slow leg


@pytest.mark.anyio
async def test_two_of_three_plurality_wins_without_the_fuser(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"a"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_plurality"


@pytest.mark.anyio
async def test_a_three_way_split_falls_through_to_the_full_path(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: tool_resp("read", '{"path":"b"}'),
                      "s": lambda p: tool_resp("read", '{"path":"c"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "full" and len(panel.candidates) == 3


@pytest.mark.anyio
async def test_a_text_candidate_never_matches_a_tool_call(tmp_path):
    ad = FakeAdapter({"a": lambda p: tool_resp("read", '{"path":"a"}'),
                      "b": lambda p: {"choices": [{"message": {"content": "I will read it"}}],
                                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                      "s": lambda p: tool_resp("read", '{"path":"a"}')})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "tool_plurality"           # a and s agreed; b dissented


@pytest.mark.anyio
async def test_two_text_candidates_still_take_the_prose_path(tmp_path):
    # The load-bearing regression: prose must not leak into the tool path just
    # because the request carried `tools`.
    def script(p):
        prompt = p["messages"][0]["content"]
        if "VERDICT" in prompt:
            return {"choices": [{"message": {"content":
                     "VERDICT a correct ok\nVERDICT b correct ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        return {"choices": [{"message": {"content": "just prose"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    ad = FakeAdapter({m: script for m in ("a", "b", "s")})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=TOOLS_BODY, **env)
    assert panel.path == "quorum"


def test_fuser_body_forwards_tools_when_the_panel_holds_calls():
    calls = ({"id": "c", "type": "function",
              "function": {"name": "read", "arguments": "{}"}},)
    panel = PanelResult("Q", {"a": Candidate("", calls)}, {}, "full", False)
    out = fuser_body(FCFG, panel, TOOLS_BODY)
    assert out["tools"] == TOOLS_BODY["tools"]


def test_fuser_body_still_strips_tools_on_the_prose_path():
    panel = PanelResult("Q", {"a": Candidate("prose")}, {}, "quorum", False)
    assert "tools" not in fuser_body(FCFG, panel, TOOLS_BODY)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_tools.py -q`
Expected: FAIL — `path` is never `"tool_fast"`; `fuser_body` never forwards `tools`.

- [ ] **Step 3: Add the decision helper**

In `gateway/fusion.py`, above `gather_panel`:

```python
def decide_tools(candidates: dict[str, Candidate], readonly: frozenset[str]):
    """Classify a set of candidates by their tool calls.

    Returns one of:
      ("agree_readonly", winner)  every candidate proposed the same calls, all
                                  read-only -- emit with no review
      ("agree_review", winner)    same calls, but at least one is write-class --
                                  the review still runs
      ("disagree", None)          the calls differ, or one is prose and one a
                                  call, or none are comparable
      ("prose", None)             no candidate proposed a call at all
    """
    by_model = {m: canonical_calls(c.tool_calls) for m, c in candidates.items()}
    if all(v is None for v in by_model.values()):
        return "prose", None
    if len(candidates) < 2:
        return "disagree", None
    values = list(by_model.values())
    if any(v is None for v in values) or len(set(values)) != 1:
        return "disagree", None
    winner = sorted(candidates)[0]
    kind = "agree_readonly" if all_readonly(values[0], readonly) else "agree_review"
    return kind, winner
```

- [ ] **Step 4: Wire it into `gather_panel`**

After the quorum is collected and **before** `_cross_review` is called:

```python
        verdict, winner = decide_tools(candidates, fcfg.readonly_tools)
        events.append(request_id, "fusion.tool_verdict",
                      {"verdict": verdict, "winner": winner})

        if verdict == "agree_readonly":
            # Structural agreement is an exact agreement signal, unlike mutual
            # "correct" verdicts -- so no review is needed. It carries no
            # independent correctness check, which is why write-class calls
            # below keep the review even when the models agree.
            await cancel(slow)
            return PanelResult(conversation, {winner: candidates[winner]}, {},
                               "tool_fast",
                               degraded=len(candidates) < len(fcfg.quorum))

        if verdict == "agree_review":
            reviews = await _cross_review(
                candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
                ledger=ledger, events=events, clock=clock,
                request_id=request_id, conversation=conversation)
            objected = any(v.verdict == "wrong"
                           for verds in reviews.values() for v in verds.values())
            # A MISSING review is not agreement either -- the same conservative
            # rule the prose path's is_consensus uses.
            if reviews and not objected:
                await cancel(slow)
                return PanelResult(conversation, {winner: candidates[winner]},
                                   reviews, "tool_reviewed",
                                   degraded=len(candidates) < len(fcfg.quorum))
            verdict = "disagree"

        if verdict == "disagree":
            candidates.update(await collect(slow))
            plur = plurality({m: canonical_calls(c.tool_calls)
                              for m, c in candidates.items()})
            if plur is not None:
                return PanelResult(conversation, {plur: candidates[plur]}, {},
                                   "tool_plurality",
                                   degraded=len(candidates) < len(fcfg.panel))
            # Three-way split: fall through to the existing full path, which
            # cross-reviews everything and lets the fuser decide.
```

Then let control continue into the existing prose flow (`_cross_review` +
`is_consensus` + the `full` return) for the `"prose"` and three-way-split cases.
Import `canonical_calls`, `plurality`, `all_readonly` from `gateway.tool_vote`.

- [ ] **Step 5: Forward `tools` to the fuser — and read its answer back**

**Task 3's review found a landmine here; this step closes it.** Once `tools` are
forwarded and Task 4's action rule tells the fuser to act, the fuser will answer
`content: null` + `tool_calls`. But `call_model(kind="fuser")` returns
`_extract_text(resp)`, which is `""` for that shape — so `_finish_fusion`'s
`if text:` is false, the **fully-billed fuser answer is discarded**, and the
gateway falls back to `best_candidate`: one of the three *disagreeing* calls,
picked by panel order. The arbitration the fuser exists to perform would be
thrown away at exactly the moment it is needed. This is M8 finding 1a reproduced
in the fuser leg.

So this step has three parts, and the first two are not optional:

1. `call_model` must return `_extract_message(resp)` for `kind == "fuser"` as
   well as `kind == "candidate"` — only the *review* calls stay text-only.
2. `_finish_fusion` must test the returned `Candidate`'s truthiness rather than a
   string's, and pass the `Candidate` through unchanged.
3. Once (1) and (2) hold, **delete the `isinstance` coercion at the top of
   `openai_response`** and restore the brief's original `candidate: Candidate`
   signature. That coercion existed only because the fuser leg still returned a
   bare string; keeping it after this step would turn a loud `AttributeError`
   into a silent wrong answer.

Add a test: a three-way split where the fuser answers with a tool call, asserting
the response carries **the fuser's** call and not any candidate's, and that the
`fusion` metadata records the fuser as the source. Prove it bites by reverting
(1) — the test must fail with a candidate's call in place of the fuser's.

Then the `tools` forwarding itself:

```python
def fuser_body(fcfg, panel: PanelResult, body: dict) -> dict:
    """The OpenAI body for the fuser call: one user message.

    `tools`/`tool_choice` are forwarded only when the panel actually holds tool
    calls -- the fuser has to be able to answer with a call, and on the prose
    path forwarding them would invite a call nobody asked for.
    """
    out = {k: body[k] for k in _FUSER_PASSTHROUGH if k in body}
    if any(c.tool_calls for c in panel.candidates.values()):
        for k in ("tools", "tool_choice"):
            if k in body:
                out[k] = body[k]
    out["messages"] = [{"role": "user", "content": build_fusion_prompt(
        panel.conversation, panel.candidates, panel.reviews)}]
    return out
```

- [ ] **Step 6: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_fusion_tools.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → zero failures.

- [ ] **Step 7: Prove three tests bite**

Mutate `decide_tools` to return `"agree_readonly"` where it returns
`"agree_review"`; confirm `test_identical_write_class_calls_are_reviewed` FAILS.
Revert. Mutate `canonical_calls([])` to return `()`; confirm
`test_two_text_candidates_still_take_the_prose_path` FAILS. Revert. Delete the
`await cancel(slow)` in the `agree_readonly` branch; confirm the fast-path test's
ledger assertion FAILS. Revert. Record all three.

- [ ] **Step 8: Commit**

```bash
git add gateway/fusion.py tests/test_fusion_tools.py
git commit -m "feat(gateway): decide tool calls structurally, escalate on disagreement

Identical read-only calls emit after two upstream calls with no review
and no fuser -- structural agreement is exact, unlike a judged verdict.
Write-class calls keep the review because agreement carries no
correctness check. Disagreement is arbitrated by the third member's
vote (M6's plurality, applicable at last because the ballots compare)
and only a three-way split reaches the fuser.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Remove the bypass and serve tool calls

**Files:**
- Modify: `gateway/app.py`
- Modify: `tests/test_app_fusion.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `_as_tool_chunks(candidate, model) -> list[bytes]`.

**Context — read this before editing tests.** Five tests in `tests/test_app_fusion.py` currently pin the bypass and **will change meaning**:

- `test_a_tool_calling_request_bypasses_fusion_and_keeps_the_tool_call`
- `test_tool_choice_alone_also_bypasses_fusion`
- `test_legacy_functions_request_bypasses_fusion`
- `test_legacy_function_call_alone_also_bypasses_fusion`
- `test_streaming_tool_calling_request_also_bypasses_fusion`

They were written to protect one invariant: **a tool request must never be billed for a panel and then handed a 502.** That invariant still holds and must still be asserted — by the new mechanism (the panel now handles tool calls) rather than by the bypass. Rewrite each to assert the new behaviour: the request reaches fusion, the response carries the tool call, and the ledger row count matches the branch taken. Keep a test that a tool request never returns 5xx even when the panel is in bad shape.

The `legacy functions/function_call` pair matter for a different reason now: those requests must reach fusion too, and the response must still carry a call.

- [ ] **Step 1: Rewrite the five tests**

```python
def test_a_tool_calling_request_now_reaches_the_fusion_panel(tmp_path, monkeypatch):
    # Was: asserted the request BYPASSED fusion (M8 finding 1a's mitigation).
    # Now the panel handles tool calls, so it must fuse -- while keeping the
    # invariant the old test protected: never billed-then-502.
    def h(req):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "read a.py"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fusion"                       # fused, not bypassed
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert body["fusion"]["path"] == "tool_fast"
    # Exact row count for THIS branch. Note it is 3, not 2, on a 3-member
    # panel: the two quorum candidates plus the cancelled slow leg, which must
    # settle an 'estimated' row rather than vanish -- the money invariant says a
    # cancelled call is settled, never failed, because the upstream did work.
    # Task 5's review measured this; an earlier draft of this plan said 2.
    rid = r.headers["x-fusion-trace-id"]
    import sqlite3
    rows = sqlite3.connect(tmp_path / "g.sqlite").execute(
        "SELECT model, state, usage_source FROM ledger WHERE request_id=?",
        (rid,)).fetchall()
    assert len(rows) == 3
    assert sorted(st for _, st, _ in rows) == ["settled", "settled", "settled"]
    # ...and no review or fuser call happened: no model appears twice.
    assert len({m for m, _, _ in rows}) == 3
```

Apply the same rewrite to `tool_choice`-only, `functions`-only and
`function_call`-only variants (each must fuse and return the call), and to the
streaming one — asserting a synthesised tool-call chunk stream. Add:

```python
def test_a_tool_request_never_5xxs_when_the_whole_panel_is_down(tmp_path, monkeypatch):
    # The invariant the deleted bypass tests existed to protect.
    c = make_client(tmp_path, monkeypatch, h=lambda req: httpx.Response(500, json={}))
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_exhausted"


def test_streaming_tool_call_is_a_valid_openai_chunk_stream(tmp_path, monkeypatch):
    def h(req):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "fusion", "stream": True,
        "messages": [{"role": "user", "content": "read a.py"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }, headers=H()) as r:
        raw = b"".join(r.iter_bytes()).decode()
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert any(c_["choices"][0]["delta"].get("tool_calls") for c_ in objs
               if c_.get("choices"))
    assert any(c_["choices"][0].get("finish_reason") == "tool_calls" for c_ in objs
               if c_.get("choices"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app_fusion.py -q`
Expected: FAIL — the bypass is still in place, so `model` is `"a"` not `"fusion"`.

- [ ] **Step 3: Remove the bypass**

In `gateway/app.py`, delete the `wants_tools` computation and the bypass block
(the `events.append(..., "fusion.bypassed", ...)` and
`requested_model = fcfg.panel[0]` lines), so the branch becomes:

```python
        fcfg = cfg.fusion
        resolved = (cfg.default_model
                    if requested_model in ("", "auto") else requested_model)
        if fcfg is not None and resolved == fcfg.model:
            # Tool calls used to be routed away from the panel because
            # `_extract_text` dropped them and a fully-billed panel then
            # returned 502 (M8 final review, finding 1a). `Candidate` +
            # `gateway/tool_vote.py` fixed the root cause, so tool calls now
            # go through the panel like anything else.
            return await _fusion_request(
                request_id=request_id, body=body, streaming=streaming,
                fcfg=fcfg,
            )
```

- [ ] **Step 4: Add the tool-call chunk synthesiser**

Beside `_as_chunks` in `gateway/app.py`:

```python
def _as_tool_chunks(candidate, model: str) -> list[bytes]:
    """Render a candidate's tool calls as a minimal OpenAI chunk stream.

    Candidates are non-streaming calls, so the complete call is already in hand
    and `arguments` needs no fragmenting -- one chunk carries the whole array.
    `finish_reason: "tool_calls"` is what makes a client execute it.
    """
    base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}
    calls = [{"index": i, "id": c.get("id") or f"call_{i}", "type": "function",
              "function": {"name": c.get("function", {}).get("name", ""),
                           "arguments": c.get("function", {}).get("arguments", "")}}
             for i, c in enumerate(candidate.tool_calls)]
    out = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                "finish_reason": None}])]
    if candidate.text:
        out.append(dict(base, choices=[{"index": 0,
                                        "delta": {"content": candidate.text},
                                        "finish_reason": None}]))
    out.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": calls},
                                    "finish_reason": None}]))
    out.append(dict(base, choices=[{"index": 0, "delta": {},
                                    "finish_reason": "tool_calls"}]))
    return [f"data: {json.dumps(o)}\n\n".encode() for o in out] + [b"data: [DONE]\n\n"]
```

- [ ] **Step 5: Use it on the streaming fusion path**

In `_fusion_request`'s streaming generator, when the panel resolved to a tool
call the fuser must not be called at all — the answer is already decided. Before
the fuser block, add:

```python
            decided = panel.path in ("tool_fast", "tool_reviewed", "tool_plurality")
            if decided:
                model_name, cand = next(iter(panel.candidates.items()))
                events.append(request_id, "fusion.fused",
                              {"path": panel.path, "source": "candidate",
                               "model": model_name})
                _finish_request(store, request_id, "succeeded", clock)
                for piece in (_as_tool_chunks(cand, fcfg.model) if cand.tool_calls
                              else _as_chunks(cand.text, fcfg.model)):
                    yield piece
                return
```

**Task 5's review flagged one more gap here:** the streaming generator has no
`len(panel.candidates) < 2` fuser skip of its own, so without the `decided`
short-circuit above a streaming `tool_fast` request would still pay for a fuser
call — now with `tools` forwarded to it. The `decided` block is therefore not an
optimisation, it is what keeps the streaming path's cost equal to the
non-streaming path's.

and mirror it in the non-streaming path: when `panel.path` is one of the three
tool verdicts, return `openai_response(cand, fcfg.model, meta)` directly without
calling the fuser. Also switch the fuser-failure fallback from
`_as_chunks(fallback[1].text, ...)` to the tool-aware pair, so a fallback that is
a tool call is delivered as one.

- [ ] **Step 6: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_app_fusion.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → zero failures.

- [ ] **Step 7: Prove the invariant test bites**

Mutate the `<2 candidates` rung to raise instead of returning 502; confirm
`test_a_tool_request_never_5xxs_when_the_whole_panel_is_down` FAILS. Revert.
Mutate `_as_tool_chunks` to emit `finish_reason: "stop"`; confirm the streaming
test FAILS. Revert. Record both.

- [ ] **Step 8: Commit**

```bash
git add gateway/app.py tests/test_app_fusion.py
git commit -m "feat(gateway): serve tool calls from the fusion panel

Removes the M8 bypass now that the root cause is fixed: a tool request
reaches the panel, and the decided call is returned directly without a
fuser round. Streaming synthesises a tool-call chunk stream from the
non-streaming candidate. The five bypass tests are rewritten to the new
behaviour, keeping the invariant they protected -- a tool request is
never billed for a panel and then handed a 5xx.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Live smoke — PAID

> **This makes real, billed calls against a deployed gateway with no budget cap.**

**Files:**
- Modify: `docs/M1_ACCEPTANCE.md`

**Context.** `kimi-k3`'s quota is exhausted (HTTP 403 `access_terminated_error`, re-probed 2026-07-30 after a reported top-up). It is the third panel member, so **the plurality branch cannot be exercised live** — record that as a gap, do not claim coverage for it. The read-only fast path and the write-class review path both run on the two quorum members and are fully testable.

- [ ] **Step 1: Start a local gateway with real keys**

```bash
cd ~/git_projects/fusion-gateway
set -a; source runs/secrets/.env; set +a
export GATEWAY_TOKENS="smoke:smoketok,admin:admintok" \
       GATEWAY_CONFIG=configs/gateway.toml GATEWAY_DB=/tmp/gw_m9.sqlite
rm -f /tmp/gw_m9.sqlite*
.venv/bin/uvicorn --factory gateway.app:create_app_from_env \
  --host 127.0.0.1 --port 8916 &
until curl -s -o /dev/null http://127.0.0.1:8916/healthz; do sleep 1; done
```

- [ ] **Step 2: Read-only tool call, timed**

```bash
time curl -s -m 300 http://127.0.0.1:8916/v1/chat/completions \
  -H "Authorization: Bearer smoketok" -H "Content-Type: application/json" \
  -d '{"model":"fusion","max_tokens":256,
       "messages":[{"role":"user","content":"Read the file config.py."}],
       "tools":[{"type":"function","function":{"name":"read","description":"Read a file",
         "parameters":{"type":"object","properties":{"path":{"type":"string"}},
                       "required":["path"]}}}]}' | .venv/bin/python -m json.tool
```

Expected: 200, `finish_reason: "tool_calls"`, a `read` call, and
`fusion.path == "tool_fast"`. Record the wall-clock time and compare it with the
~2 s the design predicts.

- [ ] **Step 3: Write-class tool call, timed**

Same command with the tool renamed to `write` (and a `body` parameter). Expected:
`fusion.path == "tool_reviewed"` — the review ran because `write` is not in
`readonly_tools`. Record the latency; the design predicts ~7 s.

- [ ] **Step 4: Verify the ledger per branch**

```bash
.venv/bin/python - <<'EOF'
import sqlite3
c = sqlite3.connect("/tmp/gw_m9.sqlite")
for rid, in c.execute("SELECT DISTINCT request_id FROM ledger WHERE request_id != 'admin'"):
    rows = c.execute("SELECT model, state, usage_source FROM ledger WHERE request_id=?",
                     (rid,)).fetchall()
    print(rid[:8], len(rows), "rows:", rows)
assert not c.execute("SELECT 1 FROM ledger WHERE state='preflight'").fetchone()
print("no stranded rows")
EOF
```

Expected: the read-only request has **2** rows, the write-class request has
**4**, and none is in `preflight`.

- [ ] **Step 5: Record and commit**

Append to `docs/M1_ACCEPTANCE.md`: the two measured latencies, the per-branch
ledger row counts, the `fusion` metadata of each response, and explicitly that
**the plurality branch was not exercised because kimi-k3 has no quota**. Then:

```bash
git add docs/M1_ACCEPTANCE.md
git commit -m "docs: record the M9 tool-call fusion live smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Deploy (separate approval)**

`HOST=vps bash scripts/deploy.sh`, then re-run steps 2–4 against
`https://gateway.cutecookie.xyz`. Fusion is opt-in, so this changes nothing for
existing default traffic — only `model: "fusion"` requests gain tool support.

---

## Self-Review

**Spec coverage.** Canonicalisation, exact-not-semantic, unparseable-as-disagreement, multiset ordering, tool-vs-text never matching, calls-only comparison, plurality, default-deny classification → Task 1 ✓ · `readonly_tools` + validation + real config → Task 2 ✓ · `Candidate`, `_extract_message`, response shape → Task 3 ✓ · prompt rendering + the fuser's action rule + prose byte-identity → Task 4 ✓ · the full decision tree incl. objection-escalates and missing-review-is-not-agreement → Task 5 ✓ · bypass removal, tool-call response, synthesised stream → Task 6 ✓ · degradation rungs → Tasks 5 and 6 ✓ · per-branch ledger assertions → Tasks 5, 6, 7 ✓ · live smoke with the kimi gap recorded → Task 7 ✓ · prose and single-model paths unchanged → Global Constraints + every task's whole-suite step ✓.

**Placeholder scan.** None. Every code step carries complete code; Task 7 carries exact commands and expected output. Task 6's rewrite of the four sibling bypass tests is described by the pattern plus one fully written example, because the four differ only in which request key triggers them.

**Type consistency.** `Candidate(text, tool_calls)` is constructed identically in Tasks 3–6. `canonical_calls` returns `tuple[CanonCall, ...] | None` in Task 1 and is consumed as nullable in Task 5. `plurality` takes `dict[str, tuple | None]` and returns `str | None` in both. `all_readonly(canon, readonly)` argument order matches. `best_candidate` returns `(model, Candidate) | None` in Tasks 3 and 6. `openai_response(candidate, model, meta)` — candidate first — matches Tasks 3 and 6. `decide_tools(candidates, readonly)` returns `(verdict, winner)` in Tasks 5's helper and call site.

**One deliberate asymmetry.** `_extract_text` survives alongside `_extract_message`: the review and fuser calls return prose and have no use for tool calls, so only the candidate path switches. Task 3 states this explicitly so a later reader does not "clean it up".

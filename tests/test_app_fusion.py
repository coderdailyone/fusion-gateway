import httpx, json
from fastapi.testclient import TestClient
from gateway.app import create_app
from tests.helpers import DelayedByteStream, FakeClock

CFG = """
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://example.invalid"
api_key_env = "P_KEY"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[models."b"]
provider = "p"
upstream_model = "b"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 128
stage_timeout_s = 5
[policy]
version = "static-v0"
default_model = "fusion"
"""


def handler(req):
    body = json.loads(req.content)
    prompt = body["messages"][-1]["content"]
    if "VERDICT" in prompt:
        text = "VERDICT a correct ok\nVERDICT b correct ok"
    elif "Produce the single best final answer" in prompt:
        text = "FUSED ANSWER"
    else:
        text = "candidate answer"
    if body.get("stream"):
        chunk = {"choices": [{"index": 0, "delta": {"content": text},
                              "finish_reason": None}]}
        payload = (f"data: {json.dumps(chunk)}\n\n"
                   'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'
                   "data: [DONE]\n\n")
        return httpx.Response(200, content=payload.encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}})


def make_client(tmp_path, monkeypatch, h=handler, cfg=CFG):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    p = tmp_path / "g.toml"
    p.write_text(cfg)
    app = create_app(p, tmp_path / "g.sqlite", clock=FakeClock(),
                     transports={"p": httpx.MockTransport(h)})
    return TestClient(app)


def H(tok="tokA"):
    return {"Authorization": f"Bearer {tok}"}


BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}

# A cap calibrated (empirically, against this exact panel/fuser/mock shape --
# see the fix-round-1 section of task-5-report.md for the derivation) to let
# both candidates and both reviews settle -- 2.8e-05 actual -- while the
# fuser's OWN preflight estimate (~0.000206, dominated by build_fusion_prompt's
# longer text) pushes consumed over a cap of 0.00022. Too low and gather_panel
# itself trips first (already covered); too high (>= 0.000234) and nothing
# trips at all. max_tokens=1 on the request and a small review_max_tokens
# keep the candidate/review preflight ESTIMATES (as opposed to their actual
# settled cost) from swamping this budget before the fuser is ever reached.
BUDGET_CFG = """
[budget]
active = "T"
[budgets.T]
cap_usd = 0.00022
[providers.p]
base_url = "https://example.invalid"
api_key_env = "P_KEY"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[models."b"]
provider = "p"
upstream_model = "b"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 5
stage_timeout_s = 5
[policy]
version = "static-v0"
default_model = "fusion"
"""

BUDGET_BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 1}


def test_auto_takes_the_fusion_path(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fusion"
    assert body["choices"][0]["message"]["content"] == "FUSED ANSWER"
    assert body["fusion"]["path"] == "quorum"
    assert "x-fusion-trace-id" in r.headers


def test_naming_the_pseudo_model_explicitly_also_fuses(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "model": "fusion"}, headers=H())
    assert r.json()["fusion"]["path"] == "quorum"


def test_naming_a_real_model_takes_the_single_model_path(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "model": "a"}, headers=H())
    assert r.status_code == 200 and r.json()["model"] == "a"
    assert "fusion" not in r.json()


def test_fusion_writes_one_ledger_row_per_call_under_one_request_id(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    rid = r.headers["x-fusion-trace-id"]
    st = c.get("/admin/status", headers=H("tokB")).json()
    assert st["ledger"]["consumed_usd"] > 0
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger WHERE request_id=?",
                        (rid,)).fetchall()
    # 2 candidates + 2 reviews + 1 fuser
    assert len(rows) == 5
    assert not any(state == "preflight" for _, state in rows)


# -- Final whole-branch review, finding 1 (CRITICAL): a tool-calling request
# was billed for the whole panel and then handed a 502. Standard OpenAI
# function-call shape (content: null, tool_calls: [...]) makes
# `_extract_text` return "" for every candidate; `collect()`'s `if text:`
# drops them all; zero candidates survive; `_finish_fusion` finds
# `best_candidate` is None -> 502, even though naming the same model
# explicitly works fine.

def _tool_call_response(model):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{}"}}]},
                    "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


TOOLS = [{"type": "function", "function": {"name": "get_weather",
                                           "parameters": {"type": "object"}}}]


def _ledger_row_count(tmp_path, request_id):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    return conn.execute(
        "SELECT COUNT(*) FROM ledger WHERE request_id=?", (request_id,)
    ).fetchone()[0]


def test_a_tool_calling_request_bypasses_fusion_and_keeps_the_tool_call(tmp_path, monkeypatch):
    def h(req):
        return _tool_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "tools": TOOLS}, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] != "fusion"          # single-model chain answered
    assert "fusion" not in body
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    # Re-review residual 2: with the handler answering the tool-call shape
    # for EVERY model, a reverted 1a is silently rescued by 1b's
    # zero-usable-candidates fallback -- status 200 and a real tool_calls
    # payload either way. Only the ledger row count tells "the panel never
    # ran" (1 row: the single bypassed call) apart from "the panel ran,
    # burned money, and then the fallback answered" (5 rows: 2 candidates +
    # 2 reviews + 1 fuser attempt/fallback).
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 1


def test_tool_choice_alone_also_bypasses_fusion(tmp_path, monkeypatch):
    # tool_choice can be sent without a fresh `tools` list (e.g. a
    # multi-turn conversation that already established the tool set) --
    # either key alone must be enough to skip fusion.
    def h(req):
        return _tool_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "tool_choice": "auto"}, headers=H())
    assert r.status_code == 200
    assert r.json()["model"] != "fusion"
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 1


# -- Re-review residual 1: the legacy OpenAI `functions`/`function_call`
# shape (deprecated but still accepted by real clients) bypassed the bypass
# -- app.py only checked `tools`/`tool_choice`, so a `functions` request
# still fused: 5 ledger rows billed and the client got prose back instead of
# a function call, the exact semantics-loss finding 1a exists to prevent.

FUNCTIONS = [{"name": "get_weather", "parameters": {"type": "object"}}]


def _legacy_function_call_response(model):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "function_call": {
            "name": "get_weather", "arguments": "{}"}},
                    "finish_reason": "function_call"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def test_legacy_functions_request_bypasses_fusion(tmp_path, monkeypatch):
    def h(req):
        return _legacy_function_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "functions": FUNCTIONS}, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] != "fusion"
    assert body["choices"][0]["message"]["function_call"]["name"] == "get_weather"
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 1


def test_legacy_function_call_alone_also_bypasses_fusion(tmp_path, monkeypatch):
    def h(req):
        return _legacy_function_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "function_call": "auto"}, headers=H())
    assert r.status_code == 200
    assert r.json()["model"] != "fusion"
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 1


def test_streaming_tool_calling_request_also_bypasses_fusion(tmp_path, monkeypatch):
    # The critical repro was non-streaming (a clean 502), but the routing
    # decision happens before either branch, so the streaming path must be
    # covered too -- otherwise a tool-calling streaming request would have
    # silently gone through the fuser and gotten prose back instead of its
    # tool call, which is worse than a loud error. The handler distinguishes
    # the fuser's own stream call (fuser_body strips `tools`) from the
    # bypassed single-model chain's stream call (the client's body, `tools`
    # and all, forwarded verbatim) by the presence of `tools` on the wire.
    def h(req):
        body = json.loads(req.content)
        if body.get("stream") and body.get("tools"):
            payload = json.dumps({
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{}"}}]},
                            "finish_reason": None}]})
            sse = (f"data: {payload}\n\n"
                  'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
                  "data: [DONE]\n\n")
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})
        if body.get("stream"):
            # The fuser's own stream (no `tools` -- fusion did NOT get
            # bypassed): plain prose, no tool call in sight.
            payload = json.dumps({"choices": [{"index": 0,
                                               "delta": {"content": "FUSED PROSE"},
                                               "finish_reason": None}]})
            sse = (f"data: {payload}\n\n"
                  'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
                  "data: [DONE]\n\n")
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})
        # Non-streaming candidate/review calls the panel makes internally.
        return _tool_call_response(body.get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "tools": TOOLS, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert "get_weather" in raw
    assert "stream_failed" not in raw and '"error"' not in raw


# -- Final whole-branch review, finding 1b: "every candidate returned no
# usable text" used to degrade straight to a 502. Paying for a panel and
# then hard-failing is the worst outcome available -- fall through to the
# single-model chain instead.

def test_zero_usable_candidates_degrades_to_the_single_model_chain_not_502(tmp_path, monkeypatch):
    def h(req):
        # Every upstream call -- panel candidates AND the eventual
        # single-model retry -- answers 200 with unusable (null) content, a
        # refusal-shaped response with no extractable text.
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200          # not 502
    assert r.json()["model"] != "fusion"  # the single-model chain answered


def test_all_upstreams_down_returns_502_not_500(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch, h=lambda req: httpx.Response(500, json={}))
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_exhausted"


def test_a_lone_survivor_is_returned_verbatim(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        if body["model"] == "a":
            return httpx.Response(500, json={})
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})     # fuser also down
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "only b"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "only b"
    assert r.json()["fusion"]["degraded"] is True
    rid = r.headers["x-fusion-trace-id"]
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model FROM ledger WHERE request_id=?", (rid,)).fetchall()
    # Only the two candidate calls: fewer than 2 candidates means
    # _cross_review's own guard skips review entirely, and _finish_fusion's
    # "<2 candidates" rung returns the survivor WITHOUT ever calling the
    # fuser -- the 500 the mock above wires up for the fuser prompt is dead
    # code on this path. A row count of 2 is what tells "no fuser call was
    # made" apart from "the fuser was called and failed" (test below), which
    # would show up here as 3 rows instead.
    assert len(rows) == 2


def test_a_dead_fuser_falls_back_to_the_best_candidate(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"answer from {body['model']}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "answer from a"
    assert r.json()["fusion"]["degraded"] is True


def test_streaming_fusion_emits_keepalives_then_the_fuser_stream(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion" in raw                    # SSE comment keepalive
    assert "FUSED ANSWER" in raw
    assert raw.rstrip().endswith("data: [DONE]")
    # Every keepalive must be an SSE COMMENT, not a data line -- a data line
    # the client cannot parse as a chunk would break a conformant SDK.
    for line in raw.splitlines():
        if line.startswith(": "):
            continue
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            json.loads(line[6:])       # must be valid JSON, or this raises


def test_streaming_fusion_settles_every_ledger_row_including_the_fuser(tmp_path, monkeypatch):
    """The fuser's row is the one call app.py bills by hand instead of
    through call_model -- confirm it actually reaches 'settled' and isn't
    quietly left stranded in 'preflight' by some future refactor."""
    c = make_client(tmp_path, monkeypatch)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert "FUSED ANSWER" in raw
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute(
        "SELECT model, state, usage_source FROM ledger WHERE request_id=? ORDER BY id",
        (rid,)).fetchall()
    assert len(rows) == 5                              # 2 candidates + 2 reviews + 1 fuser
    assert all(state == "settled" for _, state, _ in rows), rows
    # The fuser call specifically: 'reported' (not just 'settled') pins that
    # the explicit settle() in the happy path actually ran, rather than the
    # cancellation-safety-net's estimate quietly papering over a bug that
    # skipped it.
    assert rows[-1] == ("b", "settled", "reported")


def test_streaming_dead_fuser_falls_back_to_the_best_candidate(tmp_path, monkeypatch):
    """Mirrors test_a_dead_fuser_falls_back_to_the_best_candidate, but for
    the streaming path: the fuser 500s before its first byte, and the
    client -- which already has an open stream -- must get a valid chunk
    stream carrying the fallback answer, not an error chunk."""
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"answer from {body['model']}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw        # the panel succeeded before the fuser died
    assert raw.rstrip().endswith("data: [DONE]")
    contents = []
    for line in raw.splitlines():
        if line.startswith(": ") or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        assert "error" not in obj, f"fuser death leaked as an error chunk: {obj}"
        content = obj["choices"][0]["delta"].get("content")
        if content:
            contents.append(content)
    assert contents == ["answer from a"]


# -- M9 Task 3 review, finding 1 (Important, live at the Candidate-carrying-
# -commit): _fuser_gave_nothing used to call _as_chunks(fallback[1].text,
# ...) unconditionally. A tool-calls-only Candidate has empty .text, so the
# client got a silent HTTP 200 with an empty content delta -- worse than the
# 502 this milestone exists to fix, since the panel is fully billed and the
# client is told it succeeded. No `tools`/`tool_choice` on the client
# request (so the tool-call bypass never fires) but a panel member emits
# `content: null` + `tool_calls` unprompted -- a shape a provider with
# server-side tools can produce on its own.

def test_streaming_tool_call_only_fallback_after_a_dead_fuser_is_not_a_silent_empty_200(
        tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})               # fuser dead
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        # Candidate call: tool-calls only, no prose -- unprompted.
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{}"}}]},
                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200          # headers already committed by then
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw          # the panel succeeded before the fuser died
    data_lines = [l for l in raw.splitlines() if l.startswith("data: ")]
    assert len(data_lines) == 1
    err = json.loads(data_lines[0][6:])
    assert err["error"]["type"] == "upstream_exhausted"
    # Must NOT be the old silent-empty-200 shape: no content delta of any
    # kind, and no [DONE] terminator implying a normal stream completed.
    assert '"content"' not in raw
    assert "[DONE]" not in raw


def test_streaming_empty_fuser_stream_falls_back_to_the_best_candidate(tmp_path, monkeypatch):
    """The fuser answers 200 with a genuinely empty body -- no bytes at all,
    distinct from an upstream error. Without an explicit `first_byte` guard
    after the read loop this falls through to ledger.settle + 'succeeded',
    charging the caller for a stream that decodes to zero chunks and no
    [DONE] under a real SSEDecoder. The gateway already holds the right
    answer via best_candidate and must serve it instead."""
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(200, content=b"",
                                  headers={"content-type": "text/event-stream"})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"answer from {body['model']}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw
    assert raw.rstrip().endswith("data: [DONE]")
    contents = []
    for line in raw.splitlines():
        if line.startswith(": ") or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        assert "error" not in obj, f"empty fuser body billed silently: {obj}"
        content = obj["choices"][0]["delta"].get("content")
        if content:
            contents.append(content)
    assert contents == ["answer from a"]
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger WHERE request_id=? ORDER BY id",
                        (rid,)).fetchall()
    assert len(rows) == 5
    # The fuser's own row must be fail()ed -- never settled as a paid
    # success, and never left in 'preflight'.
    assert rows[-1] == ("b", "failed")


# -- Final whole-branch review, finding 7: the streaming fuser's bound was
# wall-clock, not the per-chunk idle bound its own comment claims.

def test_streaming_fuser_deadline_resets_per_chunk_not_wall_clock(tmp_path, monkeypatch):
    """stage_timeout_s=1; the fuser emits a chunk every 0.3s (never idle more
    than 0.3s) for 8 chunks -- ~2.4s of real time, comfortably over the 1s
    budget in wall-clock terms but never idle anywhere near it. Before the
    fix `deadline` was computed once before the loop and never reset, so
    `remaining` hit zero around t=1s regardless of how recently a chunk
    arrived, truncating the stream after 3 chunks with a spurious
    stream_failed -- exactly what app.py's own comment at the wait_for call
    says must NOT happen ('a fuser that is genuinely still producing output
    at the deadline isn't cut off mid-token, only one that goes quiet is').
    """
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            chunks = [
                (f'data: {{"choices":[{{"index":0,"delta":{{"content":"{i}"}},'
                 f'"finish_reason":null}}]}}\n\n').encode()
                for i in range(8)]
            chunks.append(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n')
            chunks.append(b"data: [DONE]\n\n")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=DelayedByteStream(chunks, 0.3))
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "candidate answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    cfg = CFG.replace("stage_timeout_s = 5", "stage_timeout_s = 1")
    c = make_client(tmp_path, monkeypatch, h=h, cfg=cfg)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()

    assert "stream_failed" not in raw, f"stream was cut short:\n{raw}"
    assert raw.rstrip().endswith("data: [DONE]")
    contents = []
    for line in raw.splitlines():
        if line.startswith(": ") or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if obj.get("choices"):
            content = obj["choices"][0]["delta"].get("content")
            if content is not None:
                contents.append(content)
    assert contents == [str(i) for i in range(8)], (
        f"expected all 8 chunks, got {contents!r} -- the stream was cut off "
        f"by a wall-clock deadline instead of an idle one")


class _BrokenMidStream(httpx.AsyncByteStream):
    """Yields one truncated raw chunk (no line terminator, mid-token) then
    blows up -- simulates a real upstream connection dropping mid-flight."""
    def __init__(self, first_chunk: bytes):
        self.first_chunk = first_chunk

    async def __aiter__(self):
        yield self.first_chunk
        raise RuntimeError("upstream connection dropped")

    async def aclose(self) -> None:
        pass


def test_streaming_fuser_mid_stream_error_does_not_glue_onto_a_truncated_chunk(tmp_path, monkeypatch):
    """Final whole-branch review, finding 2 (CRITICAL), fusion path. Same bug
    as test_streaming.py's single-model pin, but for the fuser's own stream
    -- M8 makes streaming fusion the default, so this is now the common
    path. Reproduced end to end against the real openai SDK decoder."""
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            # A COMPLETE, valid data line -- the connection drops before the
            # blank-line terminator that would normally follow it. That gap
            # is exactly what the \n\n-prefix fix closes; a chunk cut
            # mid-token produces unparseable JSON on its own regardless of
            # what follows, which would not isolate this bug.
            truncated = b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"wor"},"finish_reason":null}]}'
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=_BrokenMidStream(truncated))
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "candidate answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes())
    assert b"stream_failed" in raw

    from openai import OpenAI
    client = OpenAI(api_key="x", base_url="http://test",
                    http_client=httpx.Client(transport=httpx.MockTransport(
                        lambda req: httpx.Response(200, content=raw,
                            headers={"content-type": "text/event-stream"}))))
    stream = client.chat.completions.create(
        model="fusion", messages=[{"role": "user", "content": "hi"}], stream=True)
    try:
        list(stream)
    except Exception as e:
        assert not isinstance(e, json.JSONDecodeError), (
            f"the error envelope glued onto a truncated upstream line and "
            f"broke the real SDK's decoder: {e!r}\nraw={raw!r}")


def test_non_streaming_budget_trip_at_the_fuser_returns_503_not_500(tmp_path, monkeypatch):
    """The panel succeeds (2 candidates + 2 reviews settle under the cap);
    only the fuser's OWN preflight -- inside call_model, called bare from
    _finish_fusion -- crosses it. Before the fix this propagated out of the
    handler as an unhandled BudgetTripped -> gateway 500, with the
    `requests` row stranded 'open'."""
    c = make_client(tmp_path, monkeypatch, cfg=BUDGET_CFG)
    r = c.post("/v1/chat/completions", json=BUDGET_BODY, headers=H())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "budget_exhausted"
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 4                       # panel settled; no stray fuser row
    assert all(state == "settled" for _, state in rows), rows
    status = conn.execute(
        "SELECT status FROM requests WHERE id != 'admin'").fetchone()
    assert status[0] == "failed"                # never left 'open'


def test_streaming_budget_trip_at_the_fuser_emits_an_error_not_a_crash(tmp_path, monkeypatch):
    """Same budget shape as the non-streaming test above, but the client
    already has an open SSE stream by the time the fuser's preflight trips:
    status is 200 (headers are already committed) and the trip must surface
    as a single well-formed error data line, not an unhandled exception that
    kills the connection mid-stream."""
    c = make_client(tmp_path, monkeypatch, cfg=BUDGET_CFG)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BUDGET_BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw             # the panel succeeded first
    data_lines = [l for l in raw.splitlines() if l.startswith("data: ")]
    assert len(data_lines) == 1
    assert json.loads(data_lines[0][6:])["error"]["type"] == "budget_exhausted"
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger").fetchall()
    assert len(rows) == 4
    assert all(state == "settled" for _, state in rows), rows

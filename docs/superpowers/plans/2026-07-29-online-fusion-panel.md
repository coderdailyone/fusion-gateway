# Online Fusion Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gateway actually fuse — every unspecified request answered by a panel of three models that cross-review each other, with a quorum short-circuit that cuts expected latency from ~73 s to ~21 s.

**Architecture:** Two new modules. `gateway/fusion_prompts.py` is pure (no IO) — prompt building and verdict parsing, ported from `evaluator/fusion/prompts.py` with the benchmark scaffolding stripped. `gateway/fusion.py` is the async orchestrator: it launches all panel candidates at t=0, waits only for the quorum, cross-reviews them, and either short-circuits (cancelling the slow leg) or waits the slow leg out. `gateway/app.py` gains one branch; the existing single-model path is untouched.

**Tech Stack:** Python 3.10, FastAPI, httpx (`MockTransport` in tests), asyncio, pytest, SQLite.

## Global Constraints

- `gateway/` **must not import `evaluator/` or `router/`** — the evaluator pulls litellm/datasets/scikit-learn and the production venv is 36 MB because those were never installed. Prompt logic is **ported, not imported**. Enforced by a test in Task 6.
- `gateway/providers.py`, `gateway/providers_anthropic.py`, `gateway/ledger.py`, `gateway/db.py`, `gateway/events.py` are **unchanged**. No ledger or event schema change.
- The existing single-model request path in `gateway/app.py` must behave **identically** — a request naming a real model takes it unchanged.
- Fusion **must never return a 5xx the gateway itself produced.** Every failure has a rung on the degradation ladder.
- A ledger row must **never** be left in `preflight` — that is a CONSUMING_STATE cleared only by `_recover_orphans` at startup.
- A **cancelled** slow-leg call is `settle`d with `usage_source="estimated"`, never `fail`ed: the upstream did work and may bill for it, and `fail` posts $0.
- Verdict line format is exactly `VERDICT <target> <correct|wrong|unsure> <reason>`.
- A reviewer **never** sees its own candidate.
- Commit message trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Run `.venv/bin/python -m pytest tests/ -q` (the **whole** suite) after every task. `tests/test_config.py`, `tests/test_policy.py`, `tests/test_app.py` and `tests/test_streaming.py` all read the real `configs/gateway.toml` and assert its contents; a previous milestone broke 5 tests by editing that file and only running a subset.

## File Structure

| file | responsibility |
|---|---|
| `gateway/fusion_prompts.py` (new) | pure: render a conversation, build review/fusion prompts, parse verdicts |
| `gateway/fusion.py` (new) | async orchestration: candidates, reviews, consensus, cancellation, billing, events |
| `gateway/config.py` (modify) | `FusionCfg`, `[fusion]` parsing + validation, `default_model` may name the fusion model |
| `gateway/app.py` (modify) | one branch to the fusion path; streaming keepalives; response shape |
| `configs/gateway.toml` (modify) | `[fusion]` section; `policy.default_model = "fusion"` |
| `tests/test_fusion_prompts.py` (new) | Task 1 |
| `tests/test_fusion_config.py` (new) | Task 2 |
| `tests/test_fusion.py` (new) | Tasks 3–4 |
| `tests/test_app_fusion.py` (new) | Task 5 |
| `tests/test_isolation.py` (new) | Task 6 |

---

### Task 1: Pure prompt layer

**Files:**
- Create: `gateway/fusion_prompts.py`
- Test: `tests/test_fusion_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Verdict(verdict: str, reason: str)` (frozen dataclass); `VALID_VERDICTS: set[str]`; `render_conversation(messages: list[dict]) -> str`; `build_review_prompt(conversation: str, candidates: dict[str, str], reviewer: str) -> str`; `build_fusion_prompt(conversation: str, candidates: dict[str, str], reviews: dict[str, dict[str, Verdict]]) -> str`; `parse_review(text: str, valid_targets: set[str]) -> dict[str, Verdict]`.

**Context:** this is a port of `evaluator/fusion/prompts.py` and the `parse_review`/`Verdict` half of `evaluator/fusion/review.py`. Read those two files first. Everything keyed to a benchmark — `_FORMAT`, `_MCQ_FMT`, `_MATH_FMT`, `_HUMANEVAL_FMT`, `_LIVECODEBENCH_FMT`, `_DEFAULT_FMT`, `_MCQ_SOURCES`, `_MCQ_NO_EARLY_ANSWER_IS_RULE`, `format_instruction()` — exists only to satisfy official graders and **must not be carried over**. There is no grader in production.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fusion_prompts.py
import pytest
from gateway.fusion_prompts import (
    Verdict, render_conversation, build_review_prompt, build_fusion_prompt,
    parse_review,
)

CANDS = {"deepseek-chat": "The answer is 4.", "glm-5.2": "It is 4.", "kimi-k3": "Four."}


def test_render_conversation_keeps_turns_and_roles():
    out = render_conversation([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ])
    assert "Be terse." in out and "What is 2+2?" in out and "And 3+3?" in out
    # Order preserved: the latest turn is what the panel is answering.
    assert out.index("What is 2+2?") < out.index("And 3+3?")


def test_render_conversation_survives_hostile_message_shapes():
    # app.py does request.json() with no validation, so these are client-reachable.
    for messages in ([], [{}], [{"role": "user"}], [{"content": None}],
                     [{"role": 5, "content": ["a", "b"]}], [None], "notalist"):
        out = render_conversation(messages)
        assert isinstance(out, str)


def test_review_prompt_never_shows_the_reviewer_its_own_answer():
    p = build_review_prompt("Q", CANDS, reviewer="glm-5.2")
    assert "It is 4." not in p                      # its own candidate text
    assert "The answer is 4." in p and "Four." in p  # the others
    assert "--- Candidate glm-5.2 ---" not in p      # and no block header for it


def test_review_prompt_states_the_exact_verdict_format():
    p = build_review_prompt("Q", CANDS, reviewer="glm-5.2")
    assert "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>" in p


def test_fusion_prompt_carries_the_majority_copy_rule():
    # This rule is the `break` countermeasure: M5 measured fusion talking
    # itself out of 26 correct answers. Losing it silently loses the fix.
    p = build_fusion_prompt("Q", CANDS, {})
    low = p.lower()
    assert "majority" in low and "copy" in low and "verbatim" in low
    assert "only depart from the majority" in low


def test_fusion_prompt_renders_reviews_and_tolerates_none():
    reviews = {"deepseek-chat": {"glm-5.2": Verdict("correct", "matches")}}
    assert "deepseek-chat says glm-5.2 is correct: matches" in build_fusion_prompt("Q", CANDS, reviews)
    assert "(no reviews available)" in build_fusion_prompt("Q", CANDS, {})


def test_no_benchmark_scaffolding_survives_the_port():
    # These strings exist only to satisfy official graders (MCQ letter
    # extraction, \boxed{} math extraction, per-benchmark format sentences).
    # A chat gateway has no grader; carrying them over would corrupt answers.
    prompts = [build_review_prompt("Q", CANDS, reviewer="glm-5.2"),
               build_fusion_prompt("Q", CANDS, {})]
    import gateway.fusion_prompts as fp
    src = open(fp.__file__).read()
    for banned in ("answer is (X)", "\\boxed", "mmlu_pro", "gpqa_diamond",
                   "humaneval", "livecodebench", "option letter"):
        for p in prompts:
            assert banned not in p, banned
        assert banned not in src, banned


def test_parse_review_extracts_valid_lines():
    text = ("VERDICT glm-5.2 correct matches the others\n"
            "VERDICT kimi-k3 wrong off by one\n")
    got = parse_review(text, {"glm-5.2", "kimi-k3"})
    assert got["glm-5.2"] == Verdict("correct", "matches the others")
    assert got["kimi-k3"].verdict == "wrong"


def test_parse_review_drops_junk_without_raising():
    text = ("hello\n"
            "VERDICT\n"
            "VERDICT glm-5.2\n"
            "VERDICT nobody correct not a target\n"      # unknown target
            "VERDICT glm-5.2 maybe not a valid verdict\n"  # invalid verdict
            "VERDICT kimi-k3 CORRECT case is normalised\n")
    got = parse_review(text, {"glm-5.2", "kimi-k3"})
    assert "nobody" not in got and got.get("glm-5.2") is None
    assert got["kimi-k3"].verdict == "correct"
    assert parse_review(None, {"a"}) == {}
    assert parse_review("", set()) == {}


def test_parse_review_allows_a_verdict_with_no_reason():
    assert parse_review("VERDICT a correct", {"a"}) == {"a": Verdict("correct", "")}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_prompts.py -q`
Expected: collection error — `No module named 'gateway.fusion_prompts'`.

- [ ] **Step 3: Write the implementation**

```python
# gateway/fusion_prompts.py
"""Prompt construction and verdict parsing for the online fusion panel.

Ported from `evaluator/fusion/prompts.py` and the parsing half of
`evaluator/fusion/review.py`. Pure: no IO, no network, no gateway imports, so
every rule below is unit-testable without a server.

DELIBERATELY NOT PORTED: the per-benchmark answer-format machinery (`_FORMAT`,
`_MCQ_SOURCES`, `_MCQ_NO_EARLY_ANSWER_IS_RULE`, `format_instruction`). Those
exist to satisfy the official graders' extractors -- "finish with 'The answer
is (X)'", "put your final answer within \\boxed{}". A chat gateway has no
grader and no benchmark source, so those instructions would only corrupt
ordinary answers.

KEPT VERBATIM, because M5 measured them working: the structured VERDICT line
(which took reviewer agreement from 0.63-0.74 to 0.9157), no self-review, and
the majority-copy rule (the countermeasure to `break` -- fusion talking itself
out of an answer the panel already had right, 26 of 1055 tasks).
"""
from __future__ import annotations

from dataclasses import dataclass

VALID_VERDICTS = {"correct", "wrong", "unsure"}


@dataclass(frozen=True)
class Verdict:
    verdict: str  # "correct" | "wrong" | "unsure"
    reason: str


def _as_text(value) -> str:
    """Coerce an upstream/client-supplied value to text without raising.

    `app.py` calls `request.json()` with no validation, so `messages` can hold
    any JSON shape a client cares to send.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # OpenAI content parts: [{"type": "text", "text": "..."}, ...]
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value)


def render_conversation(messages) -> str:
    """Render OpenAI `messages` as a transcript for the review/fusion prompts.

    The candidates receive the client's `messages` verbatim; only the reviewer
    and the fuser need the conversation as text.
    """
    if not isinstance(messages, list):
        return ""
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _as_text(m.get("role")) or "user"
        content = _as_text(m.get("content"))
        if not content:
            continue
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _candidate_block(candidates: dict[str, str], exclude: str | None = None) -> str:
    parts = []
    for model, text in sorted(candidates.items()):
        if model == exclude:
            continue
        parts.append(f"--- Candidate {model} ---\n{text}")
    return "\n\n".join(parts)


def build_review_prompt(conversation: str, candidates: dict[str, str],
                        reviewer: str) -> str:
    """Ask `reviewer` to judge the OTHER candidates -- never its own answer."""
    return (
        "You are reviewing other models' answers to the conversation below. "
        "For EACH candidate, judge whether its answer is correct and "
        "responsive.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"{_candidate_block(candidates, exclude=reviewer)}\n\n"
        "For each candidate, output one line in exactly this format:\n"
        "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>\n"
        "Judge only correctness, not style. Output nothing else."
    )


def build_fusion_prompt(conversation: str, candidates: dict[str, str],
                        reviews: dict[str, dict[str, Verdict]]) -> str:
    """The fuser sees every candidate plus the cross-review evidence."""
    lines = []
    for reviewer, verdicts in sorted(reviews.items()):
        for target, v in sorted(verdicts.items()):
            lines.append(f"{reviewer} says {target} is {v.verdict}: {v.reason}")
    review_block = "\n".join(lines) if lines else "(no reviews available)"
    rules = [
        "- If a majority of the candidates give the SAME answer and no review "
        "calls it wrong, COPY that answer verbatim. Do not rewrite, reword, or "
        "'improve' it -- copying is the correct action here.",
        "- Only depart from the majority answer when a review identifies a "
        "concrete error in it.",
        "- If they disagree, decide using the specific objections raised, and "
        "write a corrected answer (you may combine correct parts of several).",
        "- Reply with the answer itself. Do not mention the candidates, the "
        "reviews, or that several models were consulted.",
    ]
    return (
        "Several models answered the conversation below, and reviewed each "
        "other. Produce the single best final answer.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"{_candidate_block(candidates)}\n\n"
        f"--- Peer review ---\n{review_block}\n\n"
        "Rules:\n" + "\n".join(rules)
    )


def parse_review(text, valid_targets: set[str]) -> dict[str, Verdict]:
    """Extract VERDICT lines. Malformed lines are dropped, never fatal."""
    out: dict[str, Verdict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "VERDICT":
            continue
        target, verdict = parts[1], parts[2].lower()
        if target not in valid_targets or verdict not in VALID_VERDICTS:
            continue
        out[target] = Verdict(verdict, parts[3] if len(parts) > 3 else "")
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_fusion_prompts.py -q`
Expected: 10 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 344 passed (334 + 10).

- [ ] **Step 6: Commit**

```bash
git add gateway/fusion_prompts.py tests/test_fusion_prompts.py
git commit -m "feat(gateway): pure prompt layer for the online fusion panel

Ported from evaluator/fusion without the benchmark scaffolding: the
per-benchmark format sentences and the MCQ 'answer is' rule exist only to
satisfy official graders, and there is no grader in production. The
structured VERDICT line, no-self-review, and the majority-copy rule are
carried over verbatim -- those are the parts M5 measured working.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `[fusion]` config section

**Files:**
- Modify: `gateway/config.py`
- Test: `tests/test_fusion_config.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `FusionCfg(model, panel, quorum, reviewers, fuser, review_max_tokens, stage_timeout_s)` (frozen dataclass; `panel`/`quorum`/`reviewers` are `tuple[str, ...]`); `Config.fusion: FusionCfg | None`.

**Context:** `gateway/config.py` currently ends `load_config` with a check that raises `ConfigError("policy.default_model not in models")`. Making fusion the default model breaks that check, because `fusion` is a pseudo-model and is deliberately **not** in `[models]`. The check must accept the fusion model name too.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fusion_config.py
import pytest
from pathlib import Path
from gateway.config import load_config, ConfigError

REAL = Path("configs/gateway.toml")

BASE = """
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
[models."c"]
provider = "p"
upstream_model = "c"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b", "c"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 512
stage_timeout_s = 120
[policy]
version = "static-v0"
default_model = "fusion"
"""


def write(tmp_path, text):
    p = tmp_path / "g.toml"
    p.write_text(text)
    return p


def test_fusion_section_loads(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    f = cfg.fusion
    assert f.model == "fusion"
    assert f.panel == ("a", "b", "c") and f.quorum == ("a", "b")
    assert f.reviewers == ("a", "b") and f.fuser == "b"
    assert f.review_max_tokens == 512 and f.stage_timeout_s == 120


def test_default_model_may_name_the_fusion_pseudo_model(tmp_path):
    # `fusion` is deliberately NOT in [models]; the pre-existing
    # "default_model not in models" check must not reject it.
    cfg = load_config(write(tmp_path, BASE))
    assert cfg.default_model == "fusion" and "fusion" not in cfg.models


def test_fusion_section_is_optional(tmp_path):
    text = BASE.split("[fusion]")[0] + '[policy]\nversion = "v"\ndefault_model = "a"\n'
    cfg = load_config(write(tmp_path, text))
    assert cfg.fusion is None


def test_default_model_still_rejected_when_it_names_nothing(tmp_path):
    text = BASE.replace('default_model = "fusion"', 'default_model = "nope"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("old,new", [
    ('panel = ["a", "b", "c"]', 'panel = ["a", "nope"]'),      # panel member unknown
    ('quorum = ["a", "b"]', 'quorum = ["a", "nope"]'),          # quorum member unknown
    ('reviewers = ["a", "b"]', 'reviewers = ["nope"]'),         # reviewer unknown
    ('fuser = "b"', 'fuser = "nope"'),                          # fuser unknown
])
def test_unknown_names_are_rejected(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, BASE.replace(old, new)))


def test_quorum_must_be_a_subset_of_panel(tmp_path):
    text = BASE.replace('panel = ["a", "b", "c"]', 'panel = ["a", "c"]')
    with pytest.raises(ConfigError):    # quorum names "b", not in panel
        load_config(write(tmp_path, text))


def test_panel_smaller_than_two_is_rejected(tmp_path):
    text = (BASE.replace('panel = ["a", "b", "c"]', 'panel = ["a"]')
                .replace('quorum = ["a", "b"]', 'quorum = ["a"]')
                .replace('reviewers = ["a", "b"]', 'reviewers = ["a"]')
                .replace('fuser = "b"', 'fuser = "a"'))
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_fusion_model_may_not_collide_with_a_real_model(tmp_path):
    text = BASE.replace('model = "fusion"', 'model = "a"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_the_real_config_still_loads(tmp_path):
    # Task 6 adds [fusion] to it; until then fusion is None. Either way the
    # real file must load -- several other test modules depend on it.
    load_config(REAL)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion_config.py -q`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'fusion'`.

- [ ] **Step 3: Add `FusionCfg` to `gateway/config.py`**

Insert after the `ModelCfg` dataclass:

```python
@dataclass(frozen=True)
class FusionCfg:
    model: str                      # the pseudo-model clients request
    panel: tuple[str, ...]          # every candidate, in preference order
    quorum: tuple[str, ...]         # agreement here short-circuits the slow leg
    reviewers: tuple[str, ...]      # who cross-reviews (may exclude slow models)
    fuser: str                      # writes the final answer
    review_max_tokens: int
    stage_timeout_s: float
```

Change the `Config` dataclass to carry it:

```python
@dataclass(frozen=True)
class Config:
    providers: dict[str, ProviderCfg]; models: dict[str, ModelCfg]
    policy_version: str; default_model: str
    active_budget: str; budget_caps: dict[str, float | None]   # None == no cap
    fusion: "FusionCfg | None" = None
```

- [ ] **Step 4: Parse and validate `[fusion]` in `load_config`**

Replace the existing default-model check:

```python
    pol = data["policy"]
    if pol["default_model"] not in models:
        raise ConfigError("policy.default_model not in models")
```

with:

```python
    fusion = None
    if "fusion" in data:
        f = data["fusion"]
        name = f["model"]
        if name in models:
            raise ConfigError(
                f"fusion.model {name!r} collides with a real model; the fusion "
                "pseudo-model must not shadow one"
            )
        panel = tuple(f["panel"])
        quorum = tuple(f["quorum"])
        reviewers = tuple(f["reviewers"])
        fuser = f["fuser"]
        if len(panel) < 2:
            raise ConfigError("fusion.panel needs at least 2 models")
        for field, names in (("panel", panel), ("quorum", quorum),
                             ("reviewers", reviewers), ("fuser", (fuser,))):
            for n in names:
                if n not in models:
                    raise ConfigError(f"fusion.{field} names unknown model {n!r}")
        if not set(quorum) <= set(panel):
            raise ConfigError("fusion.quorum must be a subset of fusion.panel")
        fusion = FusionCfg(
            model=name, panel=panel, quorum=quorum, reviewers=reviewers,
            fuser=fuser,
            review_max_tokens=int(f.get("review_max_tokens", 512)),
            stage_timeout_s=float(f.get("stage_timeout_s", 120)),
        )

    pol = data["policy"]
    # The fusion pseudo-model is deliberately absent from [models], so it is a
    # legitimate default even though it resolves to no ModelCfg.
    if pol["default_model"] not in models and not (
        fusion is not None and pol["default_model"] == fusion.model
    ):
        raise ConfigError("policy.default_model not in models")
```

and pass it through the constructor:

```python
    return Config(providers, models, pol["version"], pol["default_model"],
                  data["budget"]["active"], caps, fusion)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_fusion_config.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 344 + the new config tests, 0 failures. `tests/test_config.py` must still pass untouched — `Config` gained a defaulted field, so existing positional construction still works.

- [ ] **Step 7: Commit**

```bash
git add gateway/config.py tests/test_fusion_config.py
git commit -m "feat(gateway): [fusion] config section with load-time validation

The fusion pseudo-model is deliberately not in [models], so the
pre-existing default_model check has to admit it explicitly. Everything
else is rejected at load: unknown names, a quorum that is not a subset of
the panel, a panel under 2, and a fusion.model that shadows a real one.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Orchestrator — candidates, reviews, consensus, cancellation

**Files:**
- Create: `gateway/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: Task 1's `render_conversation`, `build_review_prompt`, `build_fusion_prompt`, `parse_review`, `Verdict`; Task 2's `FusionCfg`.
- Produces:
  - `PanelResult(conversation: str, candidates: dict[str, str], reviews: dict[str, dict[str, Verdict]], path: str, degraded: bool)` — frozen dataclass. `path` is `"quorum"` or `"full"`.
  - `async def gather_panel(*, fcfg, cfg, adapters, ledger, events, clock, request_id, body) -> PanelResult` — runs stages 1–2, does all billing and events. Raises `BudgetTripped` (from `gateway.ledger`) and nothing else.
  - `def is_consensus(candidates, reviews) -> bool`
  - `def fuser_body(fcfg, panel: PanelResult, body: dict) -> dict`
  - `def best_candidate(fcfg, panel: PanelResult) -> tuple[str, str] | None` — `(model, text)`
  - `def openai_response(text: str, model: str, meta: dict) -> dict`
  - `async def call_model(...) -> str | None` — internal, one billed upstream call returning its text.

**Context — the design in one paragraph.** Every panel member's candidate call is launched at `t=0`. Only the quorum members are awaited. When they are in, the reviewers cross-review them; if every reviewer judged every other candidate `correct`, that is consensus — M5's fusion prompt requires the fuser to copy a majority answer verbatim, so the slow leg provably cannot change the outcome, and it is cancelled. Otherwise the slow leg is awaited (it has been running the whole time), reviewed, and everything goes to the fuser.

**Extracting text from an upstream response:** `resp["choices"][0]["message"]["content"]`. The response is upstream-controlled, so index and key access must be defensive — a `TypeError`/`IndexError`/`KeyError` here would 500 the gateway.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fusion.py
import asyncio
import time
import pytest
from gateway.config import FusionCfg, ModelCfg, ProviderCfg
from gateway.db import connect, Store
from gateway.events import EventLog
from gateway.ledger import Ledger
from gateway.fusion_prompts import Verdict
from gateway.fusion import (
    PanelResult, gather_panel, is_consensus, fuser_body, best_candidate,
    openai_response,
)
from gateway.providers import ProviderError
from tests.helpers import FakeClock

FCFG = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                 reviewers=("a", "b"), fuser="b",
                 review_max_tokens=512, stage_timeout_s=5)


class FakeCfg:
    """Minimal stand-in for gateway.config.Config."""
    def __init__(self):
        self.models = {
            n: ModelCfg(provider="p", upstream_model=n, in_usd_per_mtok=1.0,
                        out_usd_per_mtok=1.0, fallback=())
            for n in ("a", "b", "s")
        }


class FakeAdapter:
    """Returns scripted text per upstream model; `delay` models a slow leg."""
    def __init__(self, script, delays=None, errors=()):
        self.script = script          # upstream_model -> text (or callable)
        self.delays = delays or {}
        self.errors = set(errors)
        self.calls = []

    async def chat(self, upstream_model, payload):
        self.calls.append((upstream_model, payload))
        await asyncio.sleep(self.delays.get(upstream_model, 0))
        if upstream_model in self.errors:
            raise ProviderError("p", "http", status=500)
        text = self.script.get(upstream_model, "")
        if callable(text):
            text = text(payload)
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


def make_env(tmp_path, adapter):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(connect(tmp_path / "g.sqlite"))
    clock = FakeClock()
    store.conn.execute(
        "INSERT INTO requests VALUES ('r1','t','prism','fusion','open',NULL)")
    store.conn.commit()
    return dict(fcfg=FCFG, cfg=FakeCfg(), adapters={"p": adapter},
                ledger=Ledger(store, clock, cap_usd=None, budget_name="T"),
                events=EventLog(store, clock), clock=clock,
                request_id="r1"), store


BODY = {"messages": [{"role": "user", "content": "2+2?"}]}


def agree_script(text="4"):
    """a and b answer; both review the other as correct; s is slow."""
    def review_or_answer(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} correct fine" for t in targets)
        return text
    return {"a": review_or_answer, "b": review_or_answer, "s": review_or_answer}


@pytest.mark.anyio
async def test_quorum_agreement_short_circuits_and_cancels_the_slow_leg(tmp_path):
    ad = FakeAdapter(agree_script(), delays={"s": 3})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.path == "quorum"
    assert set(panel.candidates) == {"a", "b"}     # s never contributed
    rows = store.conn.execute(
        "SELECT model, state, usage_source FROM ledger ORDER BY id").fetchall()
    states = {(r["model"], r["state"]) for r in rows}
    # The cancelled leg is SETTLED with an estimate, never failed (that would
    # post $0 for work the upstream may bill) and never left in preflight.
    assert ("s", "settled") in states
    assert [r["usage_source"] for r in rows if r["model"] == "s"] == ["estimated"]
    assert not any(r["state"] == "preflight" for r in rows)


@pytest.mark.anyio
async def test_disagreement_waits_for_the_slow_leg(tmp_path):
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong nope" for t in targets)
        return "an answer"
    ad = FakeAdapter({"a": script, "b": script, "s": script}, delays={"s": 0.05})
    env, store = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.path == "full"
    assert set(panel.candidates) == {"a", "b", "s"}


@pytest.mark.anyio
async def test_a_wrong_verdict_forces_the_full_path_even_if_answers_match(tmp_path):
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            targets = [n for n in ("a", "b", "s") if f"Candidate {n}" in prompt]
            return "\n".join(f"VERDICT {t} wrong disputed" for t in targets)
        return "identical"
    ad = FakeAdapter({"a": script, "b": script, "s": script})
    env, _ = make_env(tmp_path, ad)
    assert (await gather_panel(body=BODY, **env)).path == "full"


@pytest.mark.anyio
async def test_slow_leg_starts_at_t0_not_after_the_quorum(tmp_path):
    # The whole latency argument rests on this. If the slow leg were launched
    # after the quorum decided, the full path would cost quorum + slow instead
    # of max(quorum, slow).
    ad = FakeAdapter(agree_script(), delays={"a": 0.2, "b": 0.2, "s": 0.2})
    env, _ = make_env(tmp_path, ad)
    started = time.monotonic()
    await gather_panel(body=BODY, **env)
    # a, b and s all ran concurrently, then one review round: ~0.4s, not ~0.6s.
    assert time.monotonic() - started < 0.55


def test_is_consensus_requires_every_pairwise_correct():
    c = {"a": "x", "b": "x"}
    assert is_consensus(c, {"a": {"b": Verdict("correct", "")},
                            "b": {"a": Verdict("correct", "")}})
    # a missing review is not agreement -- absence of evidence is not evidence
    assert not is_consensus(c, {"a": {"b": Verdict("correct", "")}})
    assert not is_consensus(c, {"a": {"b": Verdict("unsure", "")},
                                "b": {"a": Verdict("correct", "")}})
    assert not is_consensus({"a": "x"}, {})          # fewer than 2 candidates
    assert not is_consensus(c, {})


def test_fuser_body_drops_client_tools_and_messages():
    panel = PanelResult("Q", {"a": "x", "b": "y"}, {}, "quorum", False)
    body = {"messages": [{"role": "user", "content": "Q"}],
            "tools": [{"type": "function"}], "max_tokens": 99,
            "temperature": 0.3, "stream": True, "user": "someone"}
    out = fuser_body(FCFG, panel, body)
    assert "tools" not in out and "user" not in out
    assert len(out["messages"]) == 1 and out["messages"][0]["role"] == "user"
    assert "Candidate a" in out["messages"][0]["content"]
    assert out["max_tokens"] == 99 and out["temperature"] == 0.3


def test_best_candidate_prefers_panel_order():
    panel = PanelResult("Q", {"b": "second", "a": "first"}, {}, "full", False)
    assert best_candidate(FCFG, panel) == ("a", "first")
    assert best_candidate(FCFG, PanelResult("Q", {}, {}, "full", True)) is None


def test_openai_response_is_well_formed():
    r = openai_response("hi", "fusion", {"path": "quorum"})
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["model"] == "fusion" and r["fusion"]["path"] == "quorum"
```

Add to `tests/test_fusion.py` the anyio backend fixture (the repo has no async tests yet):

```python
@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -q`
Expected: collection error — `No module named 'gateway.fusion'`.

If pytest reports `'anyio' not found in markers`, install the plugin the repo already ships with FastAPI: `.venv/bin/pip install anyio pytest-anyio` is **not** needed — FastAPI depends on `anyio`, and `pytest-anyio` ships as the `anyio.pytest_plugin`. Confirm with `.venv/bin/python -c "import anyio.pytest_plugin"`. If that import fails, use `asyncio.run(...)` inside plain sync tests instead of the `anyio` marker, and drop the fixture.

- [ ] **Step 3: Write the implementation**

```python
# gateway/fusion.py
"""Online fusion panel: candidates -> cross-review -> fuse.

Every panel member's candidate call is launched at t=0. Only the QUORUM is
awaited. If the quorum members review each other as correct, that is
consensus -- and M5's fusion prompt requires the fuser to COPY a majority
answer verbatim, so the slow leg provably cannot change the outcome and is
cancelled. Otherwise the slow leg (running since t=0) is awaited and folded in.

This module talks to upstreams only through the adapters `app.py` already
built, and bills through the existing ledger: one row per upstream call, all
sharing one request_id. It imports nothing from `evaluator/` or `router/`.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from gateway.fusion_prompts import (
    Verdict, build_fusion_prompt, build_review_prompt, parse_review,
    render_conversation,
)
from gateway.ledger import BudgetTripped, estimate_tokens
from gateway.providers import ProviderError

logger = logging.getLogger(__name__)

# Passed through to the fuser; everything else the client sent is dropped.
_FUSER_PASSTHROUGH = ("max_tokens", "temperature", "top_p")


@dataclass(frozen=True)
class PanelResult:
    conversation: str
    candidates: dict[str, str]
    reviews: dict[str, dict[str, Verdict]]
    path: str            # "quorum" | "full"
    degraded: bool


def _extract_text(resp) -> str:
    """Pull the assistant text out of an upstream response, defensively.

    The response is upstream-controlled; a raw index or key access here would
    turn a malformed 200 into a gateway 500.
    """
    try:
        content = resp["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):        # content parts
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


async def call_model(*, model_name, body, cfg, adapters, ledger, events, clock,
                     request_id, kind):
    """One billed upstream call. Returns its text, or None on failure.

    `kind` labels the event ("candidate" | "review" | "fuser"). Raises only
    BudgetTripped and asyncio.CancelledError.
    """
    model_cfg = cfg.models[model_name]
    adapter = adapters[model_cfg.provider]
    est_in, est_out = estimate_tokens(body.get("messages") or [],
                                      body.get("max_tokens"))
    entry_id = ledger.preflight(
        request_id, model_cfg.provider, model_name, est_in, est_out,
        model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok,
    )
    events.append(request_id, "call.attempt", {"model": model_name, "stage": kind})
    start = clock.now()

    def _latency():
        return int((clock.now() - start).total_seconds() * 1000)

    try:
        resp = await adapter.chat(model_cfg.upstream_model, body)
    except asyncio.CancelledError:
        # The quorum agreed and this leg is no longer needed. The upstream has
        # already done work and may bill for it, so settle with the preflight
        # estimate: fail() would post $0 and under-count real spend, and
        # leaving the row in 'preflight' would hold a CONSUMING_STATE that only
        # a restart clears.
        ledger.settle(entry_id, est_in, est_out, "estimated", _latency(),
                      model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
        events.append(request_id, "fusion.candidate",
                      {"model": model_name, "status": "cancelled"})
        raise
    except ProviderError as exc:
        ledger.fail(entry_id)
        events.append(request_id, "call.failed",
                      {"model": model_name, "stage": kind,
                       "kind": exc.kind, "status": exc.status})
        return None
    except Exception:
        logger.exception("fusion %s call failed model=%s request_id=%s",
                         kind, model_name, request_id)
        ledger.fail(entry_id)
        events.append(request_id, "call.failed",
                      {"model": model_name, "stage": kind, "kind": "unknown"})
        return None

    usage = (resp or {}).get("usage") or {}
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        in_tok, out_tok, source = usage["prompt_tokens"], usage["completion_tokens"], "reported"
    else:
        in_tok, out_tok, source = est_in, est_out, "estimated"
    ledger.settle(entry_id, in_tok, out_tok, source, _latency(),
                  model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
    events.append(request_id, "call.succeeded", {"model": model_name, "stage": kind})
    return _extract_text(resp)


def is_consensus(candidates: dict[str, str],
                 reviews: dict[str, dict[str, Verdict]]) -> bool:
    """True only when every candidate was reviewed `correct` by every other.

    Deliberately conservative: a missing review is NOT agreement. Absence of
    evidence sends the request down the slow path, which is the safe direction.
    """
    names = sorted(candidates)
    if len(names) < 2:
        return False
    for reviewer in names:
        verdicts = reviews.get(reviewer)
        if not verdicts:
            return False
        for target in names:
            if target == reviewer:
                continue
            got = verdicts.get(target)
            if got is None or got.verdict != "correct":
                return False
    return True


async def _cross_review(*, candidates, fcfg, cfg, adapters, ledger, events,
                        clock, request_id, conversation):
    """Each configured reviewer that produced a candidate judges the others."""
    reviewers = [r for r in fcfg.reviewers if r in candidates]
    if len(candidates) < 2 or not reviewers:
        return {}

    async def one(reviewer):
        targets = {m for m in candidates if m != reviewer}
        if not targets:
            return reviewer, {}
        body = {"messages": [{"role": "user", "content":
                              build_review_prompt(conversation, candidates, reviewer)}],
                "max_tokens": fcfg.review_max_tokens}
        text = await call_model(model_name=reviewer, body=body, cfg=cfg,
                                adapters=adapters, ledger=ledger, events=events,
                                clock=clock, request_id=request_id, kind="review")
        parsed = parse_review(text, targets)
        events.append(request_id, "fusion.review",
                      {"reviewer": reviewer, "verdicts": len(parsed)})
        return reviewer, parsed

    done = await asyncio.gather(*(one(r) for r in reviewers))
    return {r: v for r, v in done if v}


async def gather_panel(*, fcfg, cfg, adapters, ledger, events, clock,
                       request_id, body) -> PanelResult:
    """Stages 1-2. Raises BudgetTripped; every other failure degrades."""
    conversation = render_conversation(body.get("messages") or [])
    slow = [m for m in fcfg.panel if m not in fcfg.quorum]
    events.append(request_id, "fusion.started",
                  {"panel": list(fcfg.panel), "quorum": list(fcfg.quorum)})

    async def candidate(model_name):
        text = await call_model(model_name=model_name, body=body, cfg=cfg,
                                adapters=adapters, ledger=ledger, events=events,
                                clock=clock, request_id=request_id,
                                kind="candidate")
        events.append(request_id, "fusion.candidate",
                      {"model": model_name, "status": "ok" if text else "failed"})
        return text

    # t=0: launch EVERY panel member, quorum and slow alike. Launching the slow
    # leg later would make the full path cost quorum + slow instead of
    # max(quorum, slow) -- the entire latency argument rests on this line.
    tasks = {m: asyncio.create_task(candidate(m)) for m in fcfg.panel}

    async def collect(names):
        got = {}
        for m in names:
            try:
                text = await asyncio.wait_for(asyncio.shield(tasks[m]),
                                              timeout=fcfg.stage_timeout_s)
            except asyncio.TimeoutError:
                events.append(request_id, "fusion.degraded",
                              {"rung": "candidate_timeout", "model": m})
                continue
            if text:
                got[m] = text
        return got

    async def cancel(names):
        for m in names:
            task = tasks[m]
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        candidates = await collect(fcfg.quorum)
        reviews = await _cross_review(
            candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
            ledger=ledger, events=events, clock=clock, request_id=request_id,
            conversation=conversation)
        agreed = is_consensus(candidates, reviews)
        events.append(request_id, "fusion.consensus",
                      {"agreed": agreed, "candidates": sorted(candidates)})

        if agreed:
            await cancel(slow)
            return PanelResult(conversation, candidates, reviews, "quorum",
                               degraded=len(candidates) < len(fcfg.quorum))

        candidates.update(await collect(slow))
        if len(candidates) >= 2:
            reviews = await _cross_review(
                candidates=candidates, fcfg=fcfg, cfg=cfg, adapters=adapters,
                ledger=ledger, events=events, clock=clock,
                request_id=request_id, conversation=conversation)
        return PanelResult(conversation, candidates, reviews, "full",
                           degraded=len(candidates) < len(fcfg.panel))
    except BaseException:
        # Never leak a running candidate task past this function: an
        # abandoned task would settle its ledger row after the response is
        # gone, or not at all.
        await cancel([m for m in fcfg.panel if not tasks[m].done()])
        raise


def fuser_body(fcfg, panel: PanelResult, body: dict) -> dict:
    """The OpenAI body for the fuser call: one user message, no client tools."""
    out = {k: body[k] for k in _FUSER_PASSTHROUGH if k in body}
    out["messages"] = [{"role": "user", "content": build_fusion_prompt(
        panel.conversation, panel.candidates, panel.reviews)}]
    return out


def best_candidate(fcfg, panel: PanelResult):
    """The answer to fall back on when the fuser itself fails: the first
    surviving member in configured panel order."""
    for m in fcfg.panel:
        if panel.candidates.get(m):
            return m, panel.candidates[m]
    for m in sorted(panel.candidates):
        return m, panel.candidates[m]
    return None


def openai_response(text: str, model: str, meta: dict) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "fusion": meta,
    }
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -q`
Expected: all pass.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add gateway/fusion.py tests/test_fusion.py
git commit -m "feat(gateway): fusion orchestrator with a quorum short-circuit

Every panel member launches at t=0; only the quorum is awaited. When the
quorum members review each other as correct, M5's majority-copy rule means
the fuser must copy that answer verbatim -- the slow leg cannot change the
outcome, so it is cancelled. The cancelled leg settles with an estimate
rather than failing: the upstream did work and \$0 would under-count it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: The degradation ladder

**Files:**
- Modify: `gateway/fusion.py` (only if a rung is missing)
- Test: `tests/test_fusion.py` (append)

**Interfaces:**
- Consumes: everything Task 3 produced.
- Produces: no new names — this task proves the ladder holds and fixes whatever does not.

**Context:** fusion is the default path for all traffic, so a fusion crash is a gateway outage. Every rung below must be a test, and none may produce a 5xx that the gateway itself generated.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_fusion.py
from gateway.ledger import BudgetTripped


@pytest.mark.anyio
async def test_one_dead_panel_member_does_not_stop_fusion(tmp_path):
    ad = FakeAdapter(agree_script(), errors={"a"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert "a" not in panel.candidates and panel.degraded


@pytest.mark.anyio
async def test_a_single_surviving_candidate_yields_no_reviews(tmp_path):
    # Fewer than 2 candidates: there is nothing to cross-review and nothing to
    # fuse. app.py returns the survivor verbatim (Task 5).
    ad = FakeAdapter(agree_script(), errors={"a", "s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert set(panel.candidates) == {"b"} and panel.reviews == {}
    assert panel.degraded


@pytest.mark.anyio
async def test_zero_candidates_returns_an_empty_panel(tmp_path):
    ad = FakeAdapter(agree_script(), errors={"a", "b", "s"})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.candidates == {} and panel.degraded


@pytest.mark.anyio
async def test_all_reviews_failing_still_fuses(tmp_path):
    # Reviews are evidence, not a precondition. The fusion prompt already
    # renders "(no reviews available)".
    def script(payload):
        prompt = payload["messages"][0]["content"]
        if "VERDICT" in prompt:
            raise RuntimeError("reviewer exploded")
        return "an answer"

    ad = FakeAdapter({"a": script, "b": script, "s": script})
    env, _ = make_env(tmp_path, ad)
    panel = await gather_panel(body=BODY, **env)
    assert panel.reviews == {} and panel.path == "full"
    assert len(panel.candidates) >= 2


@pytest.mark.anyio
async def test_a_slow_candidate_past_the_stage_timeout_is_dropped(tmp_path):
    fcfg = FusionCfg(model="fusion", panel=("a", "b", "s"), quorum=("a", "b"),
                     reviewers=("a", "b"), fuser="b",
                     review_max_tokens=512, stage_timeout_s=0.05)
    ad = FakeAdapter(agree_script(), delays={"a": 1.0})
    env, _ = make_env(tmp_path, ad)
    env["fcfg"] = fcfg
    panel = await gather_panel(body=BODY, **env)
    assert "a" not in panel.candidates


@pytest.mark.anyio
async def test_budget_tripped_propagates_and_strands_nothing(tmp_path):
    ad = FakeAdapter(agree_script())
    env, store = make_env(tmp_path, ad)
    env["ledger"].trip()
    with pytest.raises(BudgetTripped):
        await gather_panel(body=BODY, **env)
    rows = store.conn.execute("SELECT state FROM ledger").fetchall()
    assert not any(r["state"] == "preflight" for r in rows)


@pytest.mark.anyio
async def test_no_ledger_row_is_ever_left_in_preflight(tmp_path):
    # The invariant that matters most: 'preflight' is a CONSUMING_STATE cleared
    # only by _recover_orphans at startup, so a stranded row holds budget
    # forever. Exercise every path and assert none strands.
    for errors, delays in (((), {"s": 3}), (("a",), {}), (("a", "b", "s"), {})):
        ad = FakeAdapter(agree_script(), delays=delays, errors=errors)
        env, store = make_env(tmp_path / f"{errors}{delays}", ad)
        await gather_panel(body=BODY, **env)
        rows = store.conn.execute("SELECT state FROM ledger").fetchall()
        assert not any(r["state"] == "preflight" for r in rows), (errors, delays)
```

Note `make_env` writes to `tmp_path / "g.sqlite"`; the loop above passes distinct
paths, so add `path.mkdir(parents=True, exist_ok=True)` at the top of `make_env`.

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -q`
Expected: some fail. Fix `gateway/fusion.py` until they pass — do **not** weaken a test to match the code.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 4: Commit**

```bash
git add gateway/fusion.py tests/test_fusion.py
git commit -m "test(gateway): pin every rung of the fusion degradation ladder

Fusion is the default path, so a fusion crash is a gateway outage. Dead
members, a lone survivor, zero candidates, dead reviewers, a stage timeout
and a tripped budget each have a test -- and every path asserts no ledger
row is left in 'preflight'.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire the fusion path into `app.py`

**Files:**
- Modify: `gateway/app.py`
- Test: `tests/test_app_fusion.py`

**Interfaces:**
- Consumes: `gather_panel`, `fuser_body`, `best_candidate`, `openai_response`, `call_model`, `PanelResult` from Task 3; `FusionCfg` from Task 2.
- Produces: no new public names.

**Context:** this task uses an **inline test config** written to `tmp_path`, exactly as `tests/test_app_anthropic_wire.py:33-64` already does — the real `configs/gateway.toml` is not touched until Task 6, so no existing test changes meaning here. Read that file for the pattern.

`create_app` currently reads `cfg = load_config(config_path)`. The fusion branch goes in `chat_completions`, immediately after `requested_model` is resolved and **before** `plan_route` is called, because the fusion model is not routable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app_fusion.py
import httpx, json, pytest
from fastapi.testclient import TestClient
from gateway.app import create_app
from tests.helpers import FakeClock

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


def make_client(tmp_path, monkeypatch, h=handler):
    monkeypatch.setenv("GATEWAY_TOKENS", "prism:tokA,admin:tokB")
    monkeypatch.setenv("P_KEY", "sk-p")
    p = tmp_path / "g.toml"
    p.write_text(CFG)
    app = create_app(p, tmp_path / "g.sqlite", clock=FakeClock(),
                     transports={"p": httpx.MockTransport(h)})
    return TestClient(app)


def H(tok="tokA"):
    return {"Authorization": f"Bearer {tok}"}


BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app_fusion.py -q`
Expected: FAIL — `auto` resolves to the fusion model, which `plan_route` rejects as `unknown_model`, so the first test gets a 400.

- [ ] **Step 3: Add the fusion imports to `gateway/app.py`**

```python
from gateway.fusion import (
    PanelResult, best_candidate, call_model, fuser_body, gather_panel,
    openai_response,
)
```

- [ ] **Step 4: Add the fusion branch in `chat_completions`**

Insert immediately after `events.append(request_id, "request.received", ...)` and **before** the `try: plan = plan_route(...)` block:

```python
        fcfg = cfg.fusion
        resolved = (cfg.default_model
                    if requested_model in ("", "auto") else requested_model)
        if fcfg is not None and resolved == fcfg.model:
            return await _fusion_request(
                request_id=request_id, body=body, streaming=streaming,
                fcfg=fcfg,
            )
```

Add `_fusion_request` as a nested function inside `create_app` (it needs `cfg`, `adapters`, `ledger`, `events`, `clock`, `store`), placed just above `chat_completions`:

```python
    async def _fusion_request(*, request_id, body, streaming, fcfg):
        common = dict(fcfg=fcfg, cfg=cfg, adapters=adapters, ledger=ledger,
                      events=events, clock=clock, request_id=request_id)

        if not streaming:
            try:
                panel = await gather_panel(body=body, **common)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.model})
                _finish_request(store, request_id, "failed", clock)
                return JSONResponse(status_code=503,
                                    content={"error": {"type": "budget_exhausted"}})

            text, source = await _finish_fusion(panel, body, fcfg, common,
                                                request_id)
            if text is None:
                _finish_request(store, request_id, "failed", clock)
                return JSONResponse(status_code=502,
                                    content={"error": {"type": "upstream_exhausted"}})
            _finish_request(store, request_id, "succeeded", clock)
            meta = {"path": panel.path, "panel": sorted(panel.candidates),
                    "fuser": fcfg.fuser, "degraded": panel.degraded or source != "fuser",
                    "answered_by": source}
            events.append(request_id, "fusion.fused",
                          {"fuser": fcfg.fuser, "path": panel.path, "source": source})
            return JSONResponse(content=openai_response(text, fcfg.model, meta),
                                headers={"x-fusion-trace-id": request_id})

        async def gen():
            # Stages 1-2 produce nothing visible and can take tens of seconds.
            # SSE comment lines are the spec's keepalive and are skipped by
            # conformant parsers (including the OpenAI SDKs), so an idle client
            # timeout does not fire while the panel works.
            yield b": fusion panel\n\n"
            try:
                panel = await gather_panel(body=body, **common)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.model})
                _finish_request(store, request_id, "failed", clock)
                yield b'data: {"error": {"type": "budget_exhausted"}}\n\n'
                return
            yield b": fusion fusing\n\n"

            fbody = dict(fuser_body(fcfg, panel, body), stream=True)
            model_cfg = cfg.models[fcfg.fuser]
            adapter = adapters[model_cfg.provider]
            est_in, est_out = estimate_tokens(fbody["messages"],
                                              fbody.get("max_tokens"))
            try:
                entry_id = ledger.preflight(
                    request_id, model_cfg.provider, fcfg.fuser, est_in, est_out,
                    model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
            except BudgetTripped:
                events.append(request_id, "budget.tripped", {"model": fcfg.fuser})
                _finish_request(store, request_id, "failed", clock)
                yield b'data: {"error": {"type": "budget_exhausted"}}\n\n'
                return

            events.append(request_id, "call.attempt",
                          {"model": fcfg.fuser, "stage": "fuser"})
            start = clock.now()
            accumulated = bytearray()
            first_byte = False
            try:
                async for chunk in adapter.chat_stream(model_cfg.upstream_model, fbody):
                    first_byte = True
                    accumulated.extend(chunk)
                    yield chunk
            except Exception:
                if not first_byte:
                    ledger.fail(entry_id)
                    events.append(request_id, "call.failed",
                                  {"model": fcfg.fuser, "stage": "fuser",
                                   "kind": "unknown"})
                    # The fuser never spoke: fall back to the best candidate.
                    fallback = best_candidate(fcfg, panel)
                    if fallback is None:
                        _finish_request(store, request_id, "failed", clock)
                        yield b'data: {"error": {"type": "upstream_exhausted"}}\n\n'
                        return
                    events.append(request_id, "fusion.degraded",
                                  {"rung": "fuser_failed", "model": fallback[0]})
                    _finish_request(store, request_id, "succeeded", clock)
                    for piece in _as_chunks(fallback[1], fcfg.model):
                        yield piece
                    return
                logger.exception("fusion stream failed request_id=%s", request_id)
                ledger.settle(entry_id, est_in, max(len(accumulated) // 4, 0),
                              "estimated",
                              int((clock.now() - start).total_seconds() * 1000),
                              model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
                _finish_request(store, request_id, "failed", clock)
                yield b'data: {"error": {"type": "stream_failed"}}\n\n'
                return

            latency_ms = int((clock.now() - start).total_seconds() * 1000)
            raw = bytes(accumulated)
            usage = parse_stream_usage(raw)
            if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
                in_tok, out_tok, src = (usage["prompt_tokens"],
                                        usage["completion_tokens"], "reported")
            else:
                in_tok, out_tok, src = est_in, max(len(raw) // 4, 0), "estimated"
            ledger.settle(entry_id, in_tok, out_tok, src, latency_ms,
                          model_cfg.in_usd_per_mtok, model_cfg.out_usd_per_mtok)
            events.append(request_id, "fusion.fused",
                          {"fuser": fcfg.fuser, "path": panel.path, "source": "fuser"})
            _finish_request(store, request_id, "succeeded", clock)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"x-fusion-trace-id": request_id})

    async def _finish_fusion(panel, body, fcfg, common, request_id):
        """Run the fuser. Returns (text, source) with source in
        {"fuser", "candidate"}, or (None, "none") when nothing survived."""
        if len(panel.candidates) < 2:
            fallback = best_candidate(fcfg, panel)
            if fallback is None:
                events.append(request_id, "fusion.degraded",
                              {"rung": "no_candidates"})
                return None, "none"
            events.append(request_id, "fusion.degraded",
                          {"rung": "single_candidate", "model": fallback[0]})
            return fallback[1], "candidate"

        text = await call_model(model_name=fcfg.fuser,
                                body=fuser_body(fcfg, panel, body),
                                kind="fuser",
                                **{k: v for k, v in common.items() if k != "fcfg"})
        if text:
            return text, "fuser"
        fallback = best_candidate(fcfg, panel)
        if fallback is None:
            return None, "none"
        events.append(request_id, "fusion.degraded",
                      {"rung": "fuser_failed", "model": fallback[0]})
        return fallback[1], "candidate"
```

Add the small SSE helper near the other module-level helpers in `app.py`:

```python
def _as_chunks(text: str, model: str) -> list[bytes]:
    """Render a plain answer as a minimal OpenAI chunk stream.

    Used when the fuser dies before its first byte and a candidate's answer is
    served instead: the client already opened a stream, so it must receive one.
    """
    chunk = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": {"content": text},
                          "finish_reason": None}]}
    done = dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    return [f"data: {json.dumps(chunk)}\n\n".encode(),
            f"data: {json.dumps(done)}\n\n".encode(),
            b"data: [DONE]\n\n"]
```

`app.py` already imports `json` and `uuid`; add `import time` if absent.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_app_fusion.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures. The real `configs/gateway.toml` still has no `[fusion]`, so `cfg.fusion is None` and every existing test takes the untouched single-model path.

- [ ] **Step 7: Commit**

```bash
git add gateway/app.py tests/test_app_fusion.py
git commit -m "feat(gateway): serve the fusion path from /v1/chat/completions

One branch before plan_route; the single-model path is untouched.
Streaming emits SSE comment keepalives while the panel works, then
streams the fuser. Every degradation rung answers 200 or a deliberate
502/503 -- never a gateway 500.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Make fusion the default in the real config

**Files:**
- Modify: `configs/gateway.toml`
- Modify: `tests/test_app.py`, `tests/test_streaming.py`, `tests/test_policy.py`
- Create: `tests/test_isolation.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new.

**Context — read this before touching anything.** Five test modules bind the string `"auto"` to deepseek-chat, because `policy.default_model` is `deepseek-chat` today:

| file | what breaks |
|---|---|
| `tests/test_app.py:23` | `BODY = {"model": "auto", ...}`, and tests assert `model == "deepseek-chat"` / the glm fallback / a 502 |
| `tests/test_streaming.py:11` | `BODY = {"model": "auto", "stream": True, ...}` |
| `tests/test_policy.py:9` | `plan_route(CFG, "auto")` asserts `chain == ("deepseek-chat", "glm-4.5-flash")` |
| `tests/test_config.py:11` | `cfg.default_model in cfg.models` — **fusion is not in models** |

Those tests are testing the **single-model path**, not the meaning of `"auto"`. Re-point them to the explicit model name `"deepseek-chat"` so they keep testing what they were written to test. **Do not weaken an assertion to make it pass.** `tests/test_app_anthropic_wire.py` uses its own inline config and needs no change.

- [ ] **Step 1: Add `[fusion]` to `configs/gateway.toml`**

Insert before `[policy]`:

```toml
[fusion]
# Wired 2026-07-29 (M8). The panel and fuser are exactly what M5 measured at
# 0.8901 on the 1063-task standard tier -- see docs/M5_FUSION_REPORT.md, and
# note the win over the best single member was +1.1pt at p = 0.176, i.e. not
# significant, and was measured on benchmark tasks rather than chat.
model = "fusion"
panel = ["deepseek-chat", "glm-5.2", "kimi-k3"]
# If these two agree and neither review objects, M5's own majority-copy rule
# means the fuser must copy that answer -- kimi-k3 cannot change the outcome,
# so it is cancelled. This is what turns ~73s into ~15s on the common path.
quorum = ["deepseek-chat", "glm-5.2"]
# kimi-k3 is deliberately NOT a reviewer: it is a ~34s reasoning model and
# would dominate the review stage. It remains a candidate, so its answer still
# reaches the fuser on the full path.
reviewers = ["deepseek-chat", "glm-5.2"]
fuser = "glm-5.2"
review_max_tokens = 512
stage_timeout_s = 120
```

and change the policy block:

```toml
[policy]
version = "static-v0"
default_model = "fusion"
```

- [ ] **Step 2: Verify the config loads**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path; from gateway.config import load_config
c = load_config(Path('configs/gateway.toml'))
print(c.default_model, c.fusion)"
```
Expected: `fusion FusionCfg(model='fusion', panel=('deepseek-chat', 'glm-5.2', 'kimi-k3'), ...)`.

- [ ] **Step 3: Run the whole suite and see exactly what breaks**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: failures in `tests/test_app.py`, `tests/test_streaming.py`, `tests/test_policy.py`, `tests/test_config.py`. Read each one before editing.

- [ ] **Step 4: Re-point the affected tests**

`tests/test_app.py:23` — change the shared body to name the model explicitly, since these tests exercise the single-model chain:

```python
BODY = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hello"}]}
```

`tests/test_streaming.py:11`:

```python
BODY = {"model": "deepseek-chat", "stream": True, "messages": [{"role":"user","content":"hi"}]}
```

`tests/test_policy.py` — `"auto"` now resolves to a pseudo-model that `plan_route` cannot route, which is correct behaviour. Replace the first test and add one pinning the new meaning:

```python
def test_explicit_default_chain_still_routes():
    rp = plan_route(CFG, "deepseek-chat")
    assert rp.chain == ("deepseek-chat", "glm-4.5-flash")
    assert rp.policy_version == "static-v0"

def test_auto_now_resolves_to_the_fusion_pseudo_model():
    # app.py intercepts this before plan_route; plan_route itself has no
    # route for a pseudo-model, and saying so is correct.
    assert CFG.default_model == "fusion"
    with pytest.raises(UnknownModel):
        plan_route(CFG, "fusion")
```

`tests/test_config.py:11` — `assert cfg.default_model in cfg.models` is now false by design:

```python
    # default_model names the fusion pseudo-model, which is deliberately not
    # a [models] entry; load_config admits it explicitly.
    assert cfg.default_model == cfg.fusion.model
```

- [ ] **Step 5: Add the isolation guard test**

```python
# tests/test_isolation.py
"""gateway/ must not depend on the evaluator or the offline router.

Those packages pull litellm, datasets and scikit-learn. The production venv is
36 MB precisely because they were never installed there, so an import added
here would break the deploy at runtime, not at test time.
"""
import ast
import pathlib

GATEWAY = pathlib.Path("gateway")
BANNED = {"evaluator", "router"}


def _imported_roots(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def test_gateway_never_imports_the_evaluator_or_router():
    offenders = []
    for path in sorted(GATEWAY.rglob("*.py")):
        for root in _imported_roots(path):
            if root in BANNED:
                offenders.append(f"{path}: {root}")
    assert offenders == [], offenders
```

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add configs/gateway.toml tests/test_app.py tests/test_streaming.py \
        tests/test_policy.py tests/test_config.py tests/test_isolation.py
git commit -m "feat(gateway): make fusion the default model

policy.default_model is now the fusion pseudo-model, so a request naming
no model is answered by the panel. Tests that used \"auto\" to reach the
single-model path now name deepseek-chat explicitly -- they were testing
that path, not the meaning of \"auto\".

Adds a guard test that gateway/ imports neither evaluator nor router:
those pull litellm/datasets/sklearn, which the 36 MB production venv does
not have, so such an import would break the deploy and not the suite.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Live smoke — PAID, HUMAN GATE

> **STOP.** Do not run this task without explicit approval. It makes real,
> billed calls, and the deployed gateway has **no budget cap**.

**Files:**
- Modify: `docs/M1_ACCEPTANCE.md`

**Context:** `kimi-k3`'s quota is exhausted (HTTP 403 `access_terminated_error`). Until it is topped up the panel is effectively `deepseek-chat` + `glm-5.2`, both quorum members — so the **quorum path is the only path that can run**, kimi will fail every time, and `degraded` will be true. That is the degradation ladder working, not a bug. Record it as measured, and re-run the full path after the top-up.

Note also that `kimi-k3` rejects `temperature` outright (HTTP 400). Candidate calls forward the client's body verbatim, so a client that sets `temperature` loses kimi from the panel. Confirm this in the smoke and record it.

- [ ] **Step 1: Start a local gateway with real keys**

```bash
cd ~/git_projects/fusion-gateway
set -a; source runs/secrets/.env; set +a
export GATEWAY_TOKENS="smoke:smoketok,admin:admintok" \
       GATEWAY_CONFIG=configs/gateway.toml GATEWAY_DB=/tmp/gw_m8.sqlite
rm -f /tmp/gw_m8.sqlite*
.venv/bin/uvicorn --factory gateway.app:create_app_from_env \
  --host 127.0.0.1 --port 8913 &
until curl -s -o /dev/null http://127.0.0.1:8913/healthz; do sleep 1; done
```

- [ ] **Step 2: Non-streaming fused request, timed**

```bash
time curl -s -m 300 http://127.0.0.1:8913/v1/chat/completions \
  -H "Authorization: Bearer smoketok" -H "Content-Type: application/json" \
  -d '{"model":"auto","max_tokens":256,
       "messages":[{"role":"user","content":"What is 17 * 23? Show your reasoning briefly."}]}' \
  | .venv/bin/python -m json.tool
```

Expected: HTTP 200, `"model": "fusion"`, a correct answer (391), and a `fusion` object naming the path and panel. Record the wall-clock time.

- [ ] **Step 3: Streaming fused request, timed**

```bash
time curl -sN -m 300 http://127.0.0.1:8913/v1/chat/completions \
  -H "Authorization: Bearer smoketok" -H "Content-Type: application/json" \
  -d '{"model":"auto","stream":true,"max_tokens":256,
       "messages":[{"role":"user","content":"Count from 1 to 5."}]}' | head -30
```

Expected: `: fusion panel` and `: fusion fusing` comment lines, then
`chat.completion.chunk` data lines, then `data: [DONE]`.

- [ ] **Step 4: Verify the ledger**

```bash
.venv/bin/python - <<'EOF'
import sqlite3
c = sqlite3.connect("/tmp/gw_m8.sqlite")
for rid, in c.execute("SELECT DISTINCT request_id FROM ledger WHERE request_id != 'admin'"):
    rows = c.execute("SELECT model, state, usage_source, actual_cost_usd "
                     "FROM ledger WHERE request_id=?", (rid,)).fetchall()
    print(rid, len(rows), "rows")
    for r in rows:
        print("   ", r)
assert not c.execute("SELECT 1 FROM ledger WHERE state='preflight'").fetchone(), \
    "a row was stranded in preflight"
print("no stranded rows")
EOF
```

Expected: 4–7 rows per request id (2 candidates + 1 failed kimi + 2 reviews + 1 fuser), none in `preflight`.

- [ ] **Step 5: Record the result and commit**

Append to `docs/M1_ACCEPTANCE.md` a section with: the two measured latencies, the per-request ledger row counts and total cost, the `fusion` metadata of each response, kimi-k3's observed status, and whether `degraded` was true. Then:

```bash
git add docs/M1_ACCEPTANCE.md
git commit -m "docs: record the M8 fusion live smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Deploy (separate approval)**

`HOST=vps bash scripts/deploy.sh`, then re-run steps 2–4 against
`https://gateway.cutecookie.xyz`. Fusion becomes the default for **all**
production traffic at this point, including Prism and crabot, and expected
latency rises from ~1 s to ~15 s on the common path.

---

## Self-Review

**Spec coverage.** `fusion_prompts.py` pure layer + scaffolding removal → Task 1 ✓ · `[fusion]` section with every validation rule → Task 2 ✓ · quorum short-circuit, slow leg at t=0, cancel-and-settle → Task 3 ✓ · consensus definition → Task 3 ✓ · every degradation rung → Task 4 ✓ · app branch, non-streaming response shape with the `fusion` object and `x-fusion-trace-id`, streaming keepalives → Task 5 ✓ · `default_model = "fusion"` → Task 6 ✓ · no-evaluator-import guard → Task 6 ✓ · billing (one row per call under one request_id, cancelled leg settled/estimated, nothing stranded) → Tasks 3, 4, 5, 7 ✓ · events → Tasks 3, 5 ✓ · live smoke incl. the kimi caveat → Task 7 ✓ · regression of the existing 334 → every task's whole-suite step ✓.

**Placeholder scan.** None. Every code step carries complete code; Task 7 carries exact commands and expected output.

**Type consistency.** `FusionCfg(model, panel, quorum, reviewers, fuser, review_max_tokens, stage_timeout_s)` is constructed in Task 2 and consumed identically in Tasks 3–6. `PanelResult(conversation, candidates, reviews, path, degraded)` is defined in Task 3 and constructed positionally in Tasks 3–5 tests in that order. `call_model(...)` is keyword-only with `kind`, and Task 5 passes `**common` minus `fcfg` — matching its signature, which takes no `fcfg`. `gather_panel` is keyword-only and always called with `body=` plus the `common` dict. `best_candidate` returns `(model, text) | None` in Tasks 3 and 5 alike.

**One known gap, deliberate.** `is_consensus` requires a `correct` verdict from every reviewer about every other candidate. With `reviewers` a strict subset of `panel` on the full path, candidates produced by non-reviewers (kimi-k3) are reviewed *by* the reviewers but never review *others* — so `is_consensus` would return False on the full path even in total agreement. That is harmless: consensus is only ever evaluated on the quorum, before the slow leg is folded in. Task 3's `test_disagreement_waits_for_the_slow_leg` pins the behaviour.

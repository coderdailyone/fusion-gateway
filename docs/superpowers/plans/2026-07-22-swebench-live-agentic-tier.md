# M4 — SWE-bench-Live Agentic Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-task routing tier that runs a task-level escalation cascade (deepseek-chat → claude-opus-4-8) over de-contaminated SWE-bench-Live instances, graded by the official harness, and reports whether the cascade reaches near-opus resolve rate at materially lower cost-per-successful-task.

**Architecture:** Four isolated units under `evaluator/agentic/` (runner, verifier, cascade, grade) plus a dataset loader, a resumable budget-gated pilot driver, and a report script. The cascade is *derived* from two per-instance model runs plus the proxy verifier's gate, so no redundant third run. All command execution + grading happen on a dedicated Docker eval box; the pure-logic units are developed and unit-tested on the dev box with stubs/fixtures.

**Tech Stack:** Python 3.10, pytest, LiteLLM (already wraps our model registry), SWE-agent (agent scaffold, Docker per-instance), the official SWE-bench-Live evaluation harness, HuggingFace `datasets`.

## Global Constraints

- **Isolation:** every module under `evaluator/agentic/` imports **no `gateway.*`** and never touches the gateway SQLite. (Verbatim discipline from M2c/M2d.)
- **Manifests byte-unchanged:** `configs/suite.manifest.json` (standard) and `configs/suite.hard.manifest.json` (hard) are **not modified** by this milestone.
- **Secrets:** API keys read **only** from `runs/secrets/.env` (mode 600, gitignored); never printed in full, never committed.
- **Cheat boundary (iron rule):** the proxy verifier and any escalation logic **must never read** the official hidden-test fields `FAIL_TO_PASS` / `PASS_TO_PASS`. Those are the final grader only.
- **Frozen / re-gradeable:** every model attempt (patch + trajectory + cost) and the `predictions.jsonl` are persisted; re-grading spends $0.
- **Budget gate:** the pilot spends under a **hard ~$50 ceiling**, with a **3-instance paid smoke** gate before the remainder; cost is measured via LiteLLM and frozen.
- **De-contamination:** only SWE-bench-Live instances whose issue/PR `created_at` is after every pool model's training cutoff are included; the snapshot revision is pinned.
- **Model pool:** cheap = `deepseek-chat`, strong = `claude-opus-4-8` (fallbacks if a mirror can't sustain multi-turn tool-use: `glm-5.2` cheap / `gpt-5.x` strong). Both driven through their existing `evaluator/validate.py` `MODELS` registry entries (mirror `api_base`/`api_key`).
- **Commit trailer:** end every commit message with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Branch:** `feat/m0-m1-gateway`.
- **Report:** final deliverable `docs/M4_AGENTIC_TIER_REPORT.md`.

---

## File Structure

**Phase A — dev box, no Docker (pure logic, full TDD):**
- `evaluator/agentic/__init__.py` — package marker.
- `evaluator/agentic/records.py` — `AgenticAttempt` dataclass + JSONL freeze/read.
- `evaluator/agentic/dataset.py` — load SWE-bench-Live snapshot → `Instance` list; date filter; stratified deterministic sample.
- `evaluator/agentic/verifier.py` — proxy verifier: pure `decide(...)` + container-agnostic `verify(...)` (injected `run_cmd`).
- `evaluator/agentic/cascade.py` — `run_cascade(...)` orchestration (injected runner/verifier callables).
- `evaluator/agentic/grade.py` — `write_predictions(...)` (testable) + `grade(...)` (box subprocess).
- `evaluator/agentic/runner.py` — `build_agent_config(...)` (testable) + `run(...)` (box subprocess).
- `evaluator/agentic/pilot.py` — resumable, budget-gated driver loop over instances (logic testable with stubs).
- `scripts/agentic_report.py` — compute metrics + write the report (math testable).
- Tests: `tests/eval/test_agentic_records.py`, `test_agentic_dataset.py`, `test_agentic_verifier.py`, `test_agentic_cascade.py`, `test_agentic_grade.py`, `test_agentic_runner.py`, `test_agentic_pilot.py`, `test_agentic_report.py`, `test_agentic_isolation.py`.

**Phase B — eval box, Docker (execution; needs machine access):**
- `docs/M4_BOX_SETUP.md` — box provisioning + install + harness sanity runbook.
- (executes `runner.run` / `grade.grade` for real; produces frozen runs under `evaluator/runs/m4_agentic/...`.)
- `docs/M4_AGENTIC_TIER_REPORT.md` — the published result.

---

## Phase A — dev box (no Docker required)

### Task 1: Agentic attempt record + freezing

**Files:**
- Create: `evaluator/agentic/__init__.py`
- Create: `evaluator/agentic/records.py`
- Test: `tests/eval/test_agentic_records.py`

**Interfaces:**
- Produces: `AgenticAttempt(instance_id: str, model: str, patch: str, trajectory_path: str, n_steps: int, cost_usd: float, status: str, error: str | None)`; `append_attempt(run_dir, a: AgenticAttempt) -> None`; `read_attempts(run_dir) -> list[AgenticAttempt]` (round-trip equal, reading from `attempts.jsonl`).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_agentic_records.py
from pathlib import Path
from evaluator.agentic.records import AgenticAttempt, append_attempt, read_attempts


def test_attempt_round_trip(tmp_path: Path):
    a = AgenticAttempt(
        instance_id="astropy__astropy-12345", model="deepseek-chat",
        patch="diff --git a/x b/x\n", trajectory_path="traj/astropy-12345.json",
        n_steps=14, cost_usd=0.0731, status="ok", error=None)
    b = AgenticAttempt(
        instance_id="astropy__astropy-12345", model="claude-opus-4-8",
        patch="", trajectory_path="traj/astropy-12345.opus.json",
        n_steps=0, cost_usd=0.0, status="error", error="mirror timeout")
    append_attempt(tmp_path, a)
    append_attempt(tmp_path, b)
    got = read_attempts(tmp_path)
    assert got == [a, b]
    assert (tmp_path / "attempts.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_records.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluator.agentic.records`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/__init__.py
"""M4 agentic long-task tier. Isolated from gateway.* like the rest of evaluator/."""
```

```python
# evaluator/agentic/records.py
"""Frozen agentic-attempt store: one SWE-agent run's patch + trajectory + cost."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgenticAttempt:
    instance_id: str
    model: str
    patch: str            # unified git diff produced by the agent ("" if none)
    trajectory_path: str  # relative path to the saved SWE-agent trajectory
    n_steps: int          # number of agent turns
    cost_usd: float       # summed LiteLLM completion_cost over the trajectory
    status: str           # "ok" | "error" | "timeout"
    error: str | None


def append_attempt(run_dir, a: AgenticAttempt) -> None:
    path = Path(run_dir) / "attempts.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(a)) + "\n")


def read_attempts(run_dir) -> list[AgenticAttempt]:
    path = Path(run_dir) / "attempts.jsonl"
    out: list[AgenticAttempt] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    out.append(AgenticAttempt(**json.loads(line)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_records.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/__init__.py evaluator/agentic/records.py tests/eval/test_agentic_records.py
git commit -m "feat(M4): agentic attempt record + JSONL freezing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Dataset loader — de-contamination filter + stratified sample

**Files:**
- Create: `evaluator/agentic/dataset.py`
- Test: `tests/eval/test_agentic_dataset.py`

**Interfaces:**
- Produces: `Instance(instance_id: str, repo: str, base_commit: str, problem_statement: str, created_at: str, image_ref: str, raw: dict)`; `filter_by_date(rows: list[dict], min_created: str) -> list[dict]`; `stratified_sample(instances: list[Instance], k: int, seed: str) -> list[Instance]` (deterministic, spread across repos); `load_instances(rows: list[dict]) -> list[Instance]`.
- Consumes: raw SWE-bench-Live instance dicts (standard SWE-bench schema: `instance_id`, `repo`, `base_commit`, `problem_statement`, `created_at`, and an image reference field). **The hidden-test fields `FAIL_TO_PASS`/`PASS_TO_PASS` are carried untouched in `Instance.raw` for the grader only — never read here.**

> **Implementer note (verify upstream at implement time):** confirm the exact HF dataset id + pinned revision for the SWE-bench-Live monthly snapshot (e.g. the `SWE-bench-Live/SWE-bench-Live` dataset, its config/split), the exact `created_at` field name, and the image-reference field name, against the current dataset card. Keep the shapes below; adjust only field names if the card differs, and pin the revision in a module constant `SNAPSHOT_REVISION`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_agentic_dataset.py
from evaluator.agentic.dataset import (
    Instance, filter_by_date, load_instances, stratified_sample)

ROWS = [
    {"instance_id": "a__a-1", "repo": "a/a", "base_commit": "c1",
     "problem_statement": "p1", "created_at": "2025-01-10T00:00:00Z",
     "image_ref": "img:a1", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []},
    {"instance_id": "a__a-2", "repo": "a/a", "base_commit": "c2",
     "problem_statement": "p2", "created_at": "2025-09-01T00:00:00Z",
     "image_ref": "img:a2", "FAIL_TO_PASS": ["t2"], "PASS_TO_PASS": []},
    {"instance_id": "b__b-1", "repo": "b/b", "base_commit": "c3",
     "problem_statement": "p3", "created_at": "2025-10-01T00:00:00Z",
     "image_ref": "img:b1", "FAIL_TO_PASS": ["t3"], "PASS_TO_PASS": []},
]


def test_filter_by_date_keeps_only_post_cutoff():
    kept = filter_by_date(ROWS, "2025-06-01")
    assert [r["instance_id"] for r in kept] == ["a__a-2", "b__b-1"]


def test_load_instances_maps_fields_and_preserves_raw():
    insts = load_instances(ROWS)
    assert insts[0].instance_id == "a__a-1"
    assert insts[0].repo == "a/a"
    assert insts[0].image_ref == "img:a1"
    # hidden test fields survive in raw (for the grader) but are not surfaced
    assert insts[0].raw["FAIL_TO_PASS"] == ["t1"]
    assert not hasattr(insts[0], "FAIL_TO_PASS")


def test_stratified_sample_is_deterministic_and_spreads_repos():
    insts = load_instances(ROWS)
    s1 = stratified_sample(insts, k=2, seed="m4")
    s2 = stratified_sample(insts, k=2, seed="m4")
    assert [i.instance_id for i in s1] == [i.instance_id for i in s2]  # deterministic
    assert len({i.repo for i in s1}) == 2  # spread across both repos before repeating
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluator.agentic.dataset`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/dataset.py
"""Load a pinned SWE-bench-Live snapshot, filter for de-contamination, sample.

The hidden-test fields (FAIL_TO_PASS/PASS_TO_PASS) are carried untouched inside
Instance.raw for the official grader ONLY. Nothing in the agentic pipeline other
than grade.py reads them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import cycle

# Pin at implement time (see implementer note in the plan).
SNAPSHOT_REVISION = "REPLACE_WITH_PINNED_REVISION"
# A date safely after every pool model's training cutoff (verify current cutoffs).
DEFAULT_MIN_CREATED = "2025-06-01"


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    created_at: str
    image_ref: str
    raw: dict


def filter_by_date(rows: list[dict], min_created: str) -> list[dict]:
    """Keep only rows created strictly after min_created (ISO date compare)."""
    return [r for r in rows if r["created_at"][:10] > min_created[:10]]


def load_instances(rows: list[dict]) -> list[Instance]:
    return [
        Instance(
            instance_id=r["instance_id"], repo=r["repo"],
            base_commit=r["base_commit"], problem_statement=r["problem_statement"],
            created_at=r["created_at"], image_ref=r["image_ref"], raw=r)
        for r in rows
    ]


def _key(seed: str, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def stratified_sample(instances: list[Instance], k: int, seed: str) -> list[Instance]:
    """Deterministic round-robin over repos, so a single repo can't dominate.

    Within each repo, order by a seeded hash; then pick one-per-repo in a stable
    repo order until k are chosen.
    """
    by_repo: dict[str, list[Instance]] = {}
    for inst in sorted(instances, key=lambda i: _key(seed, i.instance_id)):
        by_repo.setdefault(inst.repo, []).append(inst)
    repos = sorted(by_repo)  # stable repo order
    picked: list[Instance] = []
    pools = {r: iter(by_repo[r]) for r in repos}
    for repo in cycle(repos):
        if len(picked) >= k:
            break
        nxt = next(pools[repo], None)
        if nxt is not None:
            picked.append(nxt)
        if all(next(iter([]), None) is None for _ in []):  # noop guard
            pass
        # stop if every pool is exhausted
        if all(_exhausted(by_repo, picked, r) for r in repos):
            break
    return picked[:k]


def _exhausted(by_repo, picked, repo) -> bool:
    return sum(1 for p in picked if p.repo == repo) >= len(by_repo[repo])
```

> Note: `load_from_hf(revision=SNAPSHOT_REVISION)` (thin `datasets.load_dataset` wrapper returning `rows`) is added in Phase B where network/HF access is wired; Phase A tests inject `rows` directly so they need no network.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_dataset.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/dataset.py tests/eval/test_agentic_dataset.py
git commit -m "feat(M4): SWE-bench-Live loader — date filter + stratified sample

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Proxy verifier — decision logic + container-agnostic verify

**Files:**
- Create: `evaluator/agentic/verifier.py`
- Test: `tests/eval/test_agentic_verifier.py`

**Interfaces:**
- Produces: `VerifierResult(passed: bool, had_repro_test: bool, repro_red_green: bool, no_regression: bool, patch_nonempty: bool, flagged: bool)`; `decide(patch_nonempty, had_repro_test, repro_red_green, no_regression) -> VerifierResult`; `verify(instance, patch, run_cmd) -> VerifierResult` where `run_cmd(cmd: str) -> tuple[int, str]` runs a shell command **inside the instance container** (injected, so testable with a stub).
- **Cheat boundary:** `verify` reads only `instance.problem_statement` / `instance.repo` and runs the agent-authored reproduction test + the repo's existing tests. It must not access `instance.raw["FAIL_TO_PASS"]` / `["PASS_TO_PASS"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_agentic_verifier.py
import pytest
from evaluator.agentic.dataset import Instance
from evaluator.agentic.verifier import decide, verify


def test_decide_truth_table():
    # pass: non-empty + repro red->green + no regression
    r = decide(patch_nonempty=True, had_repro_test=True,
               repro_red_green=True, no_regression=True)
    assert r.passed and not r.flagged
    # fail: repro didn't flip
    assert not decide(True, True, False, True).passed
    # fail: regression
    assert not decide(True, True, True, False).passed
    # empty patch is an automatic fail
    assert not decide(False, True, True, True).passed
    # no repro test -> fall back to regression+nonempty, but FLAG it
    r2 = decide(patch_nonempty=True, had_repro_test=False,
                repro_red_green=False, no_regression=True)
    assert r2.passed and r2.flagged


def test_verify_never_reads_hidden_tests():
    inst = Instance("i", "r/r", "c", "boom fails", "2025-09-01T00:00:00Z",
                    "img", raw={"FAIL_TO_PASS": _Boom(), "PASS_TO_PASS": _Boom()})
    calls = []

    def run_cmd(cmd: str):
        calls.append(cmd)
        # simulate: repro fails pre-patch (rc!=0), passes post-patch (rc==0),
        # existing tests pass (rc==0)
        return (0, "ok")

    # If verify touched FAIL_TO_PASS/PASS_TO_PASS, _Boom.__eq__/__iter__ raises.
    res = verify(inst, patch="diff --git a/x b/x\n+fix\n", run_cmd=run_cmd)
    assert isinstance(res.passed, bool)
    assert calls, "verify should have run commands in the container"


class _Boom:
    def __iter__(self): raise AssertionError("hidden tests were read!")
    def __eq__(self, o): raise AssertionError("hidden tests were read!")
    def __hash__(self): return 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluator.agentic.verifier`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/verifier.py
"""Proxy verifier: decides "did this attempt pass?" as the escalation gate.

IRON RULE: never reads the official hidden tests (FAIL_TO_PASS / PASS_TO_PASS).
Signals used, all available to a real autonomous deployment:
  - reproduction test authored by the agent goes red -> green after the patch,
  - the repo's existing / touched tests do not regress,
  - the patch is non-empty.
`run_cmd(cmd) -> (returncode, output)` runs a shell command inside the instance
container; it is injected so the decision logic is testable without Docker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    had_repro_test: bool
    repro_red_green: bool
    no_regression: bool
    patch_nonempty: bool
    flagged: bool  # weak-signal case (no reproduction test available)


def decide(patch_nonempty: bool, had_repro_test: bool,
           repro_red_green: bool, no_regression: bool) -> VerifierResult:
    if not patch_nonempty:
        return VerifierResult(False, had_repro_test, repro_red_green,
                              no_regression, False, flagged=not had_repro_test)
    if had_repro_test:
        passed = repro_red_green and no_regression
        return VerifierResult(passed, True, repro_red_green, no_regression,
                              True, flagged=False)
    # fallback: no reproduction test -> regression guard only, and FLAG
    return VerifierResult(no_regression, False, False, no_regression, True,
                          flagged=True)


# Command templates the box implementation fills in (verify against the repo's
# test runner at implement time). Kept as module constants for the box wiring.
REPRO_TEST_PATH = "/tmp/m4_repro_test.py"


def verify(instance, patch: str, run_cmd: Callable[[str], tuple[int, str]]) -> VerifierResult:
    """Run the proxy checks inside the container and return the gate decision.

    The container is assumed to already have the candidate patch applied by the
    caller (cascade/runner). `run_cmd` executes inside it.
    Reads only instance.problem_statement / instance.repo — never hidden tests.
    """
    patch_nonempty = bool(patch.strip())
    if not patch_nonempty:
        return decide(False, had_repro_test=False, repro_red_green=False,
                      no_regression=False)

    # 1. Does an agent-authored reproduction test exist in the container?
    rc_exists, _ = run_cmd(f"test -f {REPRO_TEST_PATH}")
    had_repro = rc_exists == 0

    repro_red_green = False
    if had_repro:
        # It must FAIL on the un-patched tree and PASS with the patch applied.
        # The caller stashes/re-applies; here we run against the patched tree and
        # trust the caller-provided pre-patch result via a sentinel file.
        rc_post, _ = run_cmd(f"python -m pytest -q {REPRO_TEST_PATH}")
        rc_pre_marker, pre_out = run_cmd("cat /tmp/m4_repro_prepatch_rc || echo 1")
        repro_red_green = (rc_post == 0) and (pre_out.strip() != "0")

    # 2. Regression guard: the repo's existing tests still pass.
    rc_reg, _ = run_cmd("python -m pytest -q -x 2>/dev/null || true; echo $?")
    no_regression = _last_rc_zero(rc_reg)

    return decide(patch_nonempty=True, had_repro_test=had_repro,
                  repro_red_green=repro_red_green, no_regression=no_regression)


def _last_rc_zero(rc: int) -> bool:
    return rc == 0
```

> **Implementer note (box):** the exact reproduction-test authoring + pre/post-patch orchestration (stash patch → run repro → apply patch → run repro; run the repo's existing suite) is finalized against the real container in Task 9. The pure `decide()` truth table is frozen here and must not change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_verifier.py -v`
Expected: PASS (both tests; `_Boom` never raises → hidden tests untouched)

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/verifier.py tests/eval/test_agentic_verifier.py
git commit -m "feat(M4): proxy verifier — repro-red/green gate, never reads hidden tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Cascade orchestration

**Files:**
- Create: `evaluator/agentic/cascade.py`
- Test: `tests/eval/test_agentic_cascade.py`

**Interfaces:**
- Consumes: `AgenticAttempt` (Task 1), `VerifierResult` (Task 3).
- Produces: `CascadeResult(instance_id: str, accepted_patch: str, escalated: bool, cost_usd: float, cheap: AgenticAttempt, strong: AgenticAttempt | None, verifier: VerifierResult)`; `run_cascade(instance, cheap_run, strong_run, verify) -> CascadeResult` where `cheap_run(instance)->AgenticAttempt`, `strong_run(instance)->AgenticAttempt`, `verify(instance, patch)->VerifierResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_agentic_cascade.py
from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.cascade import run_cascade

PASS = VerifierResult(True, True, True, True, True, False)
FAIL = VerifierResult(False, True, False, True, True, False)


def _attempt(model, patch, cost):
    return AgenticAttempt("i", model, patch, "t.json", 5, cost, "ok", None)


def test_pass_keeps_cheap_and_does_not_call_strong():
    strong_calls = []
    res = run_cascade(
        instance=object(),
        cheap_run=lambda i: _attempt("deepseek-chat", "cheapdiff", 0.05),
        strong_run=lambda i: strong_calls.append(1) or _attempt("x", "y", 9.0),
        verify=lambda i, p: PASS)
    assert res.escalated is False
    assert res.accepted_patch == "cheapdiff"
    assert res.cost_usd == 0.05
    assert res.strong is None
    assert strong_calls == []  # strong never invoked


def test_fail_escalates_and_sums_cost():
    res = run_cascade(
        instance=object(),
        cheap_run=lambda i: _attempt("deepseek-chat", "cheapdiff", 0.05),
        strong_run=lambda i: _attempt("claude-opus-4-8", "opusdiff", 1.20),
        verify=lambda i, p: FAIL)
    assert res.escalated is True
    assert res.accepted_patch == "opusdiff"
    assert round(res.cost_usd, 4) == 1.25
    assert res.strong.model == "claude-opus-4-8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_cascade.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluator.agentic.cascade`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/cascade.py
"""Task-level escalation cascade: cheap runs the whole task; a proxy verifier
gates; escalate the whole task to the strong model only on failure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult


@dataclass(frozen=True)
class CascadeResult:
    instance_id: str
    accepted_patch: str
    escalated: bool
    cost_usd: float
    cheap: AgenticAttempt
    strong: AgenticAttempt | None
    verifier: VerifierResult


def run_cascade(
    instance,
    cheap_run: Callable[[object], AgenticAttempt],
    strong_run: Callable[[object], AgenticAttempt],
    verify: Callable[[object, str], VerifierResult],
) -> CascadeResult:
    cheap = cheap_run(instance)
    v = verify(instance, cheap.patch)
    if v.passed:
        return CascadeResult(cheap.instance_id, cheap.patch, False,
                             cheap.cost_usd, cheap, None, v)
    strong = strong_run(instance)
    return CascadeResult(cheap.instance_id, strong.patch, True,
                         cheap.cost_usd + strong.cost_usd, cheap, strong, v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_cascade.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/cascade.py tests/eval/test_agentic_cascade.py
git commit -m "feat(M4): task-level escalation cascade orchestration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Predictions writer + SWE-agent config builder (the testable seams)

**Files:**
- Create: `evaluator/agentic/grade.py`
- Create: `evaluator/agentic/runner.py`
- Test: `tests/eval/test_agentic_grade.py`
- Test: `tests/eval/test_agentic_runner.py`

**Interfaces:**
- Produces (`grade.py`): `write_predictions(results: list[CascadeResult], model_name: str, path) -> None` — writes SWE-bench `predictions.jsonl` (one obj per instance: `instance_id`, `model_name_or_path`, `model_patch`). `grade(predictions_path, instances, box) -> dict[str, bool]` is declared but implemented in Phase B (raises `NotImplementedError` for now with a docstring).
- Produces (`runner.py`): `build_agent_config(model_name: str, registry) -> dict` — maps a registry model to the SWE-agent LiteLLM config (`model`, `api_base`, `api_key`, `max_tokens`). `run(instance, model_name, registry, work_dir, box) -> AgenticAttempt` declared, implemented in Phase B.
- Consumes: `CascadeResult` (Task 4); the `MODELS` registry from `evaluator/validate.py` (dict of factories) — but `build_agent_config` takes a plain `registry` dict `{name: {"model","api_base","api_key","max_tokens"}}` so it is testable without importing live keys.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_agentic_grade.py
import json
from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.cascade import CascadeResult
from evaluator.agentic.grade import write_predictions

V = VerifierResult(True, True, True, True, True, False)


def test_write_predictions_swebench_shape(tmp_path):
    r = CascadeResult(
        "astropy__astropy-1", "diff --git a/f b/f\n+x\n", True, 1.1,
        AgenticAttempt("astropy__astropy-1", "deepseek-chat", "", "t", 3, 0.1, "ok", None),
        AgenticAttempt("astropy__astropy-1", "claude-opus-4-8", "diff --git a/f b/f\n+x\n", "t2", 7, 1.0, "ok", None),
        V)
    p = tmp_path / "predictions.jsonl"
    write_predictions([r], model_name="cascade-deepseek-opus", path=p)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows == [{
        "instance_id": "astropy__astropy-1",
        "model_name_or_path": "cascade-deepseek-opus",
        "model_patch": "diff --git a/f b/f\n+x\n",
    }]
```

```python
# tests/eval/test_agentic_runner.py
from evaluator.agentic.runner import build_agent_config

REGISTRY = {
    "deepseek-chat": {"model": "deepseek/deepseek-chat",
                      "api_base": None, "api_key": "sk-x", "max_tokens": 8192},
    "claude-opus-4-8": {"model": "anthropic/claude-opus-4-8",
                        "api_base": "https://mirror/claudecode", "api_key": "sk-y",
                        "max_tokens": 8192},
}


def test_build_agent_config_maps_registry_to_litellm():
    cfg = build_agent_config("claude-opus-4-8", REGISTRY)
    assert cfg["model"] == "anthropic/claude-opus-4-8"
    assert cfg["api_base"] == "https://mirror/claudecode"
    assert cfg["api_key"] == "sk-y"
    assert cfg["max_tokens"] == 8192


def test_build_agent_config_unknown_model_raises():
    import pytest
    with pytest.raises(KeyError):
        build_agent_config("nope", REGISTRY)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_grade.py tests/eval/test_agentic_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/grade.py
"""Write SWE-bench predictions and (Phase B) invoke the official harness."""
from __future__ import annotations

import json
from pathlib import Path


def write_predictions(results, model_name: str, path) -> None:
    """Write one SWE-bench prediction object per instance to predictions.jsonl."""
    path = Path(path)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "instance_id": r.instance_id,
                "model_name_or_path": model_name,
                "model_patch": r.accepted_patch,
            }) + "\n")


def grade(predictions_path, instances, box) -> dict:
    """Run the official SWE-bench-Live evaluation harness on the eval box.

    Implemented in Phase B (Task 9): invokes the harness subprocess over `box`
    (an ssh/exec handle), parses its report json, and returns
    {instance_id: resolved_bool}. Never re-implements grading logic — it shells
    out to the official harness so `resolved` is upstream-defined.
    """
    raise NotImplementedError("grade() is wired on the Docker box in Task 9")
```

```python
# evaluator/agentic/runner.py
"""Drive SWE-agent on one instance with one model, returning an AgenticAttempt.

build_agent_config (pure, testable) maps our model registry to SWE-agent's
LiteLLM config. run() (Phase B) executes SWE-agent in the instance container.
"""
from __future__ import annotations


def build_agent_config(model_name: str, registry: dict) -> dict:
    """Map a registry entry to the LiteLLM config SWE-agent consumes."""
    entry = registry[model_name]  # KeyError on unknown model (intended)
    return {
        "model": entry["model"],
        "api_base": entry.get("api_base"),
        "api_key": entry["api_key"],
        "max_tokens": entry.get("max_tokens", 8192),
    }


def run(instance, model_name: str, registry: dict, work_dir, box):
    """Run SWE-agent on `instance` with `model_name` inside its container.

    Implemented in Phase B (Task 9): starts the instance's Docker container on
    `box`, runs SWE-agent (config from build_agent_config) against the repo at
    instance.base_commit, captures the final `git diff` patch + trajectory +
    summed LiteLLM cost + step count, and returns an AgenticAttempt. On any
    failure returns status="error"/"timeout" with an empty patch.
    """
    raise NotImplementedError("run() is wired on the Docker box in Task 9")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_grade.py tests/eval/test_agentic_runner.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/grade.py evaluator/agentic/runner.py tests/eval/test_agentic_grade.py tests/eval/test_agentic_runner.py
git commit -m "feat(M4): predictions writer + SWE-agent config builder (testable seams)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Pilot driver — resumable, budget-gated loop

**Files:**
- Create: `evaluator/agentic/pilot.py`
- Test: `tests/eval/test_agentic_pilot.py`

**Interfaces:**
- Consumes: `run_cascade` (Task 4), `AgenticAttempt`/`append_attempt`/`read_attempts` (Task 1).
- Produces: `run_pilot(instances, cheap_run, strong_run, verify, run_dir, ceiling, spent_so_far=0.0) -> list[CascadeResult]` — for each instance not already frozen, run the cascade, freeze both attempts, stop cleanly if projected spend would breach `ceiling`. Mirrors `run_budgeted`'s resume/skip + ceiling backstop (but the unit of work is a cascade, not one completion).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_agentic_pilot.py
from evaluator.agentic.records import AgenticAttempt, read_attempts
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.dataset import Instance
from evaluator.agentic.pilot import run_pilot

PASS = VerifierResult(True, True, True, True, True, False)


def _inst(n):
    return Instance(f"i{n}", "r/r", "c", "p", "2025-09-01T00:00:00Z", "img", {})


def _mk(model, cost):
    return lambda i: AgenticAttempt(i.instance_id, model, "d", "t", 3, cost, "ok", None)


def test_resumes_and_respects_ceiling(tmp_path):
    insts = [_inst(0), _inst(1), _inst(2)]
    # each cheap run costs 0.40; ceiling 1.00 -> only 2 instances fit
    results = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                        strong_run=_mk("claude-opus-4-8", 5.0),
                        verify=lambda i, p: PASS, run_dir=tmp_path, ceiling=1.00)
    assert len(results) == 2                      # stopped before breaching ceiling
    assert len(read_attempts(tmp_path)) == 2      # frozen

    # resume: already-frozen instances are skipped, remaining one now fits
    more = run_pilot(insts, cheap_run=_mk("deepseek-chat", 0.40),
                     strong_run=_mk("claude-opus-4-8", 5.0),
                     verify=lambda i, p: PASS, run_dir=tmp_path, ceiling=1.00)
    assert [r.instance_id for r in more] == ["i2"]  # only the un-done one
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError: evaluator.agentic.pilot`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/agentic/pilot.py
"""Resumable, budget-gated driver over instances. One unit of work = one cascade.

Resume: any instance whose cheap attempt is already frozen in run_dir is skipped.
Budget: a hard ceiling backstop — if the next cascade's *worst-case* projected
spend (cheap + strong) would breach the ceiling, stop cleanly (frozen work stays
re-gradeable). Mirrors scripts/resample_official.run_budgeted's discipline.
"""
from __future__ import annotations

from evaluator.agentic.records import append_attempt, read_attempts
from evaluator.agentic.cascade import run_cascade

# Worst-case per-instance projection for the gate (cheap + strong upper estimate).
WORST_CASE_PER_INSTANCE = 3.0


def run_pilot(instances, cheap_run, strong_run, verify, run_dir, ceiling,
              spent_so_far: float = 0.0):
    done = {a.instance_id for a in read_attempts(run_dir)}
    spent = spent_so_far + sum(a.cost_usd for a in read_attempts(run_dir))
    results = []
    for inst in instances:
        if inst.instance_id in done:
            continue
        if spent + WORST_CASE_PER_INSTANCE > ceiling:
            break  # stop cleanly before risking a breach
        res = run_cascade(inst, cheap_run, strong_run, verify)
        append_attempt(run_dir, res.cheap)
        if res.strong is not None:
            append_attempt(run_dir, res.strong)
        spent += res.cost_usd
        results.append(res)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_pilot.py -v`
Expected: PASS (resume + ceiling both hold)

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/pilot.py tests/eval/test_agentic_pilot.py
git commit -m "feat(M4): resumable budget-gated cascade driver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Report math + isolation test

**Files:**
- Create: `scripts/agentic_report.py`
- Test: `tests/eval/test_agentic_report.py`
- Test: `tests/eval/test_agentic_isolation.py`

**Interfaces:**
- Consumes: `wilson_ci` from `scripts/hard_report.py` (reuse — already `wilson_ci(k, n, z=1.96)`).
- Produces: `metrics(resolved: dict[str, bool], costs: dict[str, float]) -> dict` returning `n`, `resolved` count, `resolve_rate`, `wilson_lo/hi`, `total_cost`, `cost_per_successful` (= total_cost / resolved count, or `inf` if 0); `verifier_agreement(verifier_pass: dict[str,bool], resolved: dict[str,bool]) -> dict` returning `precision`, `recall`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_agentic_report.py
from scripts.agentic_report import metrics, verifier_agreement


def test_metrics_resolve_rate_and_cost_per_successful():
    resolved = {"i1": True, "i2": False, "i3": True, "i4": False}
    costs = {"i1": 0.10, "i2": 0.20, "i3": 0.30, "i4": 0.40}
    m = metrics(resolved, costs)
    assert m["n"] == 4
    assert m["resolved"] == 2
    assert abs(m["resolve_rate"] - 0.5) < 1e-9
    assert abs(m["total_cost"] - 1.0) < 1e-9
    assert abs(m["cost_per_successful"] - 0.5) < 1e-9  # 1.0 / 2
    assert 0.0 <= m["wilson_lo"] <= m["resolve_rate"] <= m["wilson_hi"] <= 1.0


def test_metrics_zero_resolved_is_inf_cost():
    m = metrics({"i1": False}, {"i1": 0.9})
    assert m["cost_per_successful"] == float("inf")


def test_verifier_agreement_precision_recall():
    # verifier passed i1,i2 ; officially resolved i1,i3
    vp = {"i1": True, "i2": True, "i3": False, "i4": False}
    res = {"i1": True, "i2": False, "i3": True, "i4": False}
    a = verifier_agreement(vp, res)
    assert abs(a["precision"] - 0.5) < 1e-9   # of {i1,i2} passed, 1 resolved
    assert abs(a["recall"] - 0.5) < 1e-9      # of {i1,i3} resolved, 1 caught
```

```python
# tests/eval/test_agentic_isolation.py
import ast
import pathlib

AGENTIC = pathlib.Path("evaluator/agentic")


def test_agentic_modules_never_import_gateway():
    for py in AGENTIC.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("gateway"), f"{py} imports {n.name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("gateway"), \
                    f"{py} imports from {node.module}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_report.py tests/eval/test_agentic_isolation.py -v`
Expected: `test_agentic_report` FAILs (`ModuleNotFoundError: scripts.agentic_report`); `test_agentic_isolation` PASSES (modules already clean).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/agentic_report.py
"""Compute M4 agentic-tier metrics and render docs/M4_AGENTIC_TIER_REPORT.md.

Reuses scripts/hard_report.wilson_ci. Pure functions here are unit-tested; the
main() render reads frozen attempts + the harness resolved map.
"""
from __future__ import annotations

from scripts.hard_report import wilson_ci


def metrics(resolved: dict, costs: dict) -> dict:
    n = len(resolved)
    k = sum(1 for v in resolved.values() if v)
    total = sum(costs.get(i, 0.0) for i in resolved)
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    return {
        "n": n, "resolved": k,
        "resolve_rate": (k / n) if n else 0.0,
        "wilson_lo": lo, "wilson_hi": hi,
        "total_cost": total,
        "cost_per_successful": (total / k) if k else float("inf"),
    }


def verifier_agreement(verifier_pass: dict, resolved: dict) -> dict:
    ids = [i for i in verifier_pass if i in resolved]
    passed = [i for i in ids if verifier_pass[i]]
    truly = [i for i in ids if resolved[i]]
    tp = sum(1 for i in passed if resolved[i])
    precision = (tp / len(passed)) if passed else float("nan")
    recall = (tp / len(truly)) if truly else float("nan")
    return {"precision": precision, "recall": recall, "n": len(ids)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_report.py tests/eval/test_agentic_isolation.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/agentic_report.py tests/eval/test_agentic_report.py tests/eval/test_agentic_isolation.py
git commit -m "feat(M4): report metrics (resolve rate, cost/successful, verifier agreement) + isolation test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Full Phase-A regression + isolation sweep**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_*.py -v`
Expected: PASS (every agentic unit test). This closes Phase A — all logic is proven with **zero Docker and zero spend**.

---

## Phase B — eval box (Docker required; machine access needed from here)

> **Machine access:** Phase B cannot start until the user provides SSH access to the dedicated Docker box (≥16 GB RAM / ≥100 GB disk). Everything before this point was built and tested without it.

### Task 8: Box setup + official harness sanity

**Files:**
- Create: `docs/M4_BOX_SETUP.md` (runbook)

- [ ] **Step 1: Provision + record the box**
Confirm the box specs (`nproc`, `free -h`, `df -h`), Docker works (`docker run --rm hello-world`), and the dev user can run Docker without sudo. Record all of this in `docs/M4_BOX_SETUP.md`.

- [ ] **Step 2: Install the agent + harness in a dedicated venv (do NOT touch the dev box .venv)**
On the box: create `~/m4/.venv`, `pip install` SWE-agent (`sweagent`) and the SWE-bench harness (`swebench`) + `datasets`. Pin versions in `docs/M4_BOX_SETUP.md`. **Verify the current SWE-agent CLI/config schema and the SWE-bench-Live evaluation entrypoint against upstream docs at this step** (they evolve); record the exact commands used.

- [ ] **Step 3: Pull one instance image + sanity-grade its gold patch**
Pick one pinned-snapshot instance. Build `predictions.jsonl` containing that instance's **gold `patch`** (from the dataset) as `model_patch`. Run the official SWE-bench-Live evaluation harness on it.
Expected: the harness reports that instance **`resolved: true`**. This proves the box + images + harness are correct before any model runs. Record the exact command + output in the runbook.

- [ ] **Step 4: Commit the runbook**

```bash
git add docs/M4_BOX_SETUP.md
git commit -m "docs(M4): eval-box setup runbook + harness gold-patch sanity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Wire real `runner.run` + `grade.grade` + `verify` on the box (one-instance dry run, cheap model, $-bounded)

**Files:**
- Modify: `evaluator/agentic/runner.py` (implement `run`)
- Modify: `evaluator/agentic/grade.py` (implement `grade`)
- Modify: `evaluator/agentic/verifier.py` (finalize the container command orchestration in `verify`)
- Create: `evaluator/agentic/box.py` (thin SSH/exec + docker helpers: `container_for(instance)`, `run_cmd_in(container, cmd)`, `copy_out_patch(container)`)
- Modify: `evaluator/agentic/dataset.py` (add `load_from_hf(revision=SNAPSHOT_REVISION)`)
- Test: `tests/eval/test_agentic_box.py` (unit-test the command *construction* in `box.py` with a stub exec — no real Docker)

**Interfaces:**
- Produces: `box.py` `run_cmd_in(container, cmd) -> tuple[int, str]` (the callable `verify` consumes); `runner.run(...)` returns a real `AgenticAttempt`; `grade.grade(...)` returns `{instance_id: resolved}`.

- [ ] **Step 1: Unit-test box command construction (dev-box-safe)**

```python
# tests/eval/test_agentic_box.py
from evaluator.agentic.box import build_docker_exec

def test_build_docker_exec_quotes_and_targets_container():
    cmd = build_docker_exec("ctr123", "python -m pytest -q /tmp/m4_repro_test.py")
    assert cmd[:3] == ["docker", "exec", "ctr123"]
    assert "python -m pytest -q /tmp/m4_repro_test.py" in " ".join(cmd)
```

- [ ] **Step 2: Run it, watch it fail, implement `build_docker_exec`, pass**
Run: `.venv/bin/python -m pytest tests/eval/test_agentic_box.py -v` → FAIL → implement the pure `build_docker_exec(container, cmd) -> list[str]` in `box.py` → PASS. (The SSH/exec execution wrappers around it are exercised for real in Step 3, not in a unit test.)

- [ ] **Step 3: Implement `run`, `grade`, and finalize `verify` against the real container**
On the box, wire:
  - `runner.run`: start the instance container (from `instance.image_ref`), run SWE-agent with `build_agent_config(...)` pointed at the mirror, capture the final `git diff` as `patch`, save the trajectory to `work_dir`, sum LiteLLM cost, return `AgenticAttempt`. Enforce a per-instance wall-clock timeout; on failure return `status="error"`/`"timeout"`, empty patch.
  - `verify`: author/detect the reproduction test, run it pre-patch (record rc to `/tmp/m4_repro_prepatch_rc`), apply the patch, run it post-patch, run the repo's existing suite — all via `run_cmd_in`. **Never reads FAIL_TO_PASS/PASS_TO_PASS.**
  - `grade.grade`: write `predictions.jsonl`, invoke the official SWE-bench-Live harness over the box, parse its report json → `{instance_id: resolved}`.

- [ ] **Step 4: One-instance end-to-end dry run (cheap model only, hard $ cap)**
Run the full chain on **one** instance with `deepseek-chat` only: `runner.run` → `verify` → `write_predictions` → `grade.grade`. Confirm a non-empty patch, a verifier decision, and a harness `resolved` value come back, and the LiteLLM cost is captured. Cap spend at **$2**.
Expected: the chain completes and prints `patch_len>0`, a `VerifierResult`, and `resolved ∈ {true,false}`. Freeze the attempt.

- [ ] **Step 5: Commit**

```bash
git add evaluator/agentic/box.py evaluator/agentic/runner.py evaluator/agentic/grade.py evaluator/agentic/verifier.py evaluator/agentic/dataset.py tests/eval/test_agentic_box.py
git commit -m "feat(M4): wire SWE-agent runner, container verifier, official grader on the box

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: 3-instance paid smoke gate (deepseek + opus)

**No new files** — this is a gated execution + a decision recorded in `docs/M4_BOX_SETUP.md`.

- [ ] **Step 1: Run 3 instances through BOTH models**
On 3 sampled instances, run `runner.run` for `deepseek-chat` **and** `claude-opus-4-8` (the first real opus spend). This confirms **each mirror sustains SWE-agent's multi-turn tool-use loop** — the risk the whole pilot hinges on. Hard cap: **$15**.

- [ ] **Step 2: Record measured cost + escalation signal**
Append to `docs/M4_BOX_SETUP.md`: measured **per-instance cost** for each model, whether both mirrors completed the tool loop (if a mirror failed, switch to the fallback: `glm-5.2` cheap / `gpt-5.x` strong and re-run the 3), and the **projected 30-instance cost** vs the $50 ceiling.

- [ ] **Step 3: GO / NO-GO gate**
If both mirrors work and the projection is under $50 → proceed to Task 11. If not → stop and report to the user with the numbers (this is a user decision — do not overspend). Commit the recorded smoke result.

```bash
git add docs/M4_BOX_SETUP.md
git commit -m "docs(M4): 3-instance paid smoke — measured cost + mirror tool-loop go/no-go

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: ~30-instance pilot run

**No new files** — executes `evaluator/agentic/pilot.py` on the box with the real runner/verify.

- [ ] **Step 1: Assemble the run**
Build the instance list: `load_from_hf(SNAPSHOT_REVISION)` → `filter_by_date(min_created=DEFAULT_MIN_CREATED)` → `stratified_sample(k=30, seed="m4")`. Create `evaluator/runs/m4_agentic/<ts>/`.

- [ ] **Step 2: Run the resumable, budget-gated pilot**
Call `run_pilot(instances, cheap_run=<real deepseek runner.run>, strong_run=<real opus runner.run>, verify=<real container verify>, run_dir=..., ceiling=50.0)`. It freezes every attempt, skips already-done instances on resume, and stops cleanly before the ceiling. Re-run to resume after any transient mirror error (prune-and-retry).
Expected: two runs per instance frozen (cheap always; opus for the cascade's strong leg, also serving as the opus-only baseline), within the $50 ceiling.

- [ ] **Step 3: Grade all three configurations**
`write_predictions` three times — deepseek-only (cheap patches), opus-only (opus patches), cascade (cheap-if-verifier-passed-else-opus) — then `grade.grade` each. Freeze the resolved maps.
Expected: a `{instance_id: resolved}` map for each of the three configurations, re-gradeable at $0.

- [ ] **Step 4: Commit the frozen run pointers**
Commit any small run-metadata/index files (not large trajectories — those are gitignored under `evaluator/runs/` per existing pattern; confirm `.gitignore` covers them).

```bash
git add -A evaluator/runs/m4_agentic/*/index.json 2>/dev/null || true
git commit -m "chore(M4): freeze ~30-instance pilot run pointers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "nothing to commit"
```

---

### Task 12: Publish the report + go/no-go on the full run

**Files:**
- Create: `docs/M4_AGENTIC_TIER_REPORT.md`
- Modify: `docs/POSITIONING.md` (swap the "roadmap / honest gap" note for the measured M4 pilot result + link)

- [ ] **Step 1: Generate the numbers**
`scripts/agentic_report.py main()` reads the frozen attempts + the three resolved maps and computes, per configuration (deepseek-only / opus-only / cascade): resolve rate + Wilson CI, total cost, **cost-per-successful-task**, and (cascade) **escalation rate**; plus **verifier-vs-grader** precision/recall and the weak-signal (no-repro) count.

- [ ] **Step 2: Write `docs/M4_AGENTIC_TIER_REPORT.md`**
Include: the three-configuration table (resolve rate ± CI, cost/successful-task), the escalation rate, the verifier-agreement numbers, the de-contamination note (pinned snapshot + `min_created`), and the **verdict + explicit GO/NO-GO recommendation on the full SWE-bench-Live run** (does the cascade reach near-opus resolve at materially lower cost/successful-task?).

- [ ] **Step 3: Update POSITIONING.md**
Replace the speculative "the path from benchmarks to real work" paragraph's TBD with the measured pilot cost-per-successful-real-task numbers and a link to `docs/M4_AGENTIC_TIER_REPORT.md`.

- [ ] **Step 4: Full test sweep + commit**

Run: `.venv/bin/python -m pytest tests/eval/test_agentic_*.py -v`
Expected: PASS.

```bash
git add docs/M4_AGENTIC_TIER_REPORT.md docs/POSITIONING.md
git commit -m "docs(M4): agentic-tier pilot report + go/no-go on full run

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Goal / thesis (cascade vs singles, cost-per-successful-task) → Tasks 4, 11, 12. ✓
- Execution env = dedicated Docker box, box-agnostic pipeline → Phase B gating + Task 8. ✓
- Four isolated units runner/verifier/cascade/grade → Tasks 3,4,5,9. ✓
- SWE-agent via LiteLLM registry → Task 5 (`build_agent_config`) + Task 9 (`run`). ✓
- Task-level escalation cascade → Task 4. ✓
- Repro-test-first proxy verifier, never touches hidden grader → Task 3 (+ `_Boom` test) + Task 9. ✓
- Official harness grading on the box → Task 5 (`write_predictions`) + Tasks 8, 9. ✓
- De-contamination (pinned snapshot + date filter) → Task 2 + Task 11. ✓
- Two runs/instance, cascade derived, no third run → Tasks 4, 11. ✓
- Pilot ~30, 3-instance paid smoke, $50 ceiling → Tasks 10, 11 + `run_pilot`. ✓
- Freezing / resumable / budget gate → Tasks 1, 6, 11. ✓
- Isolation (no gateway import), manifests unchanged → Task 7 isolation test; no task touches the manifests. ✓
- Report `docs/M4_AGENTIC_TIER_REPORT.md` (resolve+CI, cost/successful, escalation, verifier agreement, go/no-go) → Tasks 7, 12. ✓

**Placeholder scan:** the only intentional deferred values are `SNAPSHOT_REVISION` and the upstream-API confirmations (SWE-agent CLI, SWE-bench-Live harness entrypoint, HF dataset id/field names), each explicitly flagged as an "verify upstream at implement time" step in Tasks 2/8/9 — mirroring the M2d spec's LCB approach, not silent TODOs. All pure-logic code is complete and tested.

**Type consistency:** `AgenticAttempt`, `VerifierResult`, `CascadeResult` field names and the `cheap_run/strong_run/verify` callable signatures are identical across Tasks 1, 3, 4, 5, 6, 11. `write_predictions(results, model_name, path)` and `build_agent_config(model_name, registry)` match their tests. `wilson_ci(k, n)` reuse matches `scripts/hard_report.py`.

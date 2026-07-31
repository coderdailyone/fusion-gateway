# M5 — Domestic Fusion Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fusion panel where three cheap domestic models (deepseek-chat, glm-5.2, kimi-k3) answer each task, cross-review each other's answers, and a domestic fuser (glm-5.2) synthesizes one final answer — then measure whether it reaches frontier single-model quality (gpt-5.6-sol 0.894 / opus 0.913) with no frontier call.

**Architecture:** Four isolated units under `evaluator/fusion/` (panel, review, fuse, prompts) plus a report script. Candidates are read from existing frozen M2c runs ($0); only cross-review and fusion make new API calls. Fused answers are graded by the **existing official scorers** and frozen, so re-scoring and post-hoc gate simulation cost $0.

**Tech Stack:** Python 3.10, pytest, LiteLLM via the existing `evaluator/validate.py` model registry, the existing official scorers (`evaluator/scorers/*` → `evaluator/official/*`), frozen JSONL store (`evaluator/store.py`).

## Global Constraints

- **Isolation:** every module under `evaluator/fusion/` imports **no `gateway.*`** and never touches the gateway SQLite.
- **No frontier model in any panel/review/fusion role.** Panel = `deepseek-chat`, `glm-5.2`, `kimi-k3`; fuser = `glm-5.2`. Frontier numbers come only from existing M2c results. Violating this makes the headline claim circular.
- **No LLM judge in scoring.** Cross-review informs fusion only; grading stays objective (`evaluator/scorers/*`).
- **Manifests byte-unchanged:** `configs/suite.manifest.json` and `configs/suite.hard.manifest.json` are not modified.
- **Secrets** only from `runs/secrets/.env` (mode 600, gitignored); never printed, never committed.
- **Frozen / re-scorable:** every new model output is persisted; re-scoring spends $0.
- **Budget gate:** standard-tier ceiling **$25**, preceded by a **20-task paid smoke**; resumable; reuse `scripts/resample_official.py::run_budgeted`.
- **Benchmark bar (M2c, 1063 tasks, official scoring):** opus 0.913 · gpt-5.5 0.905 · sonnet 0.898 · **gpt-5.6-sol 0.894** · deepseek-chat 0.859 · glm-5.2 0.842.
- **Oracle gate:** if the domestic panel's oracle < 0.894, stop and report the pool is insufficient (do not tune prompts).
- **Commit trailer:** every commit message ends with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Branch:** `feat/m0-m1-gateway`. **venv:** `.venv` (Python 3.10). **Tests:** `tests/eval/` (baseline 128 passing).

---

## File Structure

**Phase A — pure logic, no network, full TDD:**
- `evaluator/fusion/__init__.py` — package marker.
- `evaluator/fusion/panel.py` — `PanelCase` + assemble candidates from frozen runs.
- `evaluator/fusion/prompts.py` — review + fusion prompt construction (reuses official answer-format instructions).
- `evaluator/fusion/review.py` — `Verdict` + parse structured review output + `cross_review` driver.
- `evaluator/fusion/fuse.py` — `fuse` (calls the fuser, returns the final answer text).
- `scripts/fusion_report.py` — oracle, fix/break/net-gain, reviewer agreement, gate-curve simulation, report rendering.
- Tests: `tests/eval/test_fusion_panel.py`, `test_fusion_prompts.py`, `test_fusion_review.py`, `test_fusion_fuse.py`, `test_fusion_report.py`, `test_fusion_isolation.py`.

**Phase B — paid execution (needs keys + spend):**
- `scripts/run_fusion.py` — budget-gated, resumable driver (kimi sampling + review + fusion).
- `docs/M5_FUSION_REPORT.md` — the published result.

---

## Phase A — pure logic (no network, no spend)

### Task 1: PanelCase + candidate assembly from frozen runs

**Files:**
- Create: `evaluator/fusion/__init__.py`
- Create: `evaluator/fusion/panel.py`
- Test: `tests/eval/test_fusion_panel.py`

**Interfaces:**
- Produces: `PanelCase(task_id: str, source: str, candidates: dict[str, str])`; `assemble(frozen_by_model: dict[str, dict[str, str]], task_ids: list[str], min_candidates: int = 2) -> tuple[list[PanelCase], list[str]]` returning `(cases, excluded_task_ids)`; `load_frozen_by_model(run_dirs: dict[str, str]) -> dict[str, dict[str, str]]` (model → {task_id: output_text}, `status == "ok"` only).
- Consumes: `evaluator.store.read_frozen(run_dir) -> list[FrozenOutput]` where `FrozenOutput` has `.task_id`, `.output_text`, `.status`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_fusion_panel.py
from evaluator.fusion.panel import PanelCase, assemble


FROZEN = {
    "deepseek-chat": {"t1": "answer A", "t2": "ds2", "t3": "ds3"},
    "glm-5.2":       {"t1": "answer B", "t2": "glm2"},
    "kimi-k3":       {"t1": "answer C"},
}


def test_assemble_builds_cases_with_all_available_candidates():
    cases, excluded = assemble(FROZEN, ["t1", "t2", "t3"])
    by_id = {c.task_id: c for c in cases}
    assert set(by_id["t1"].candidates) == {"deepseek-chat", "glm-5.2", "kimi-k3"}
    assert by_id["t1"].candidates["glm-5.2"] == "answer B"
    # t2 has only two candidates -> still a valid (degraded) panel
    assert set(by_id["t2"].candidates) == {"deepseek-chat", "glm-5.2"}


def test_task_with_fewer_than_min_candidates_is_excluded_not_crashed():
    cases, excluded = assemble(FROZEN, ["t1", "t2", "t3"], min_candidates=2)
    assert "t3" in excluded                      # only deepseek answered
    assert all(c.task_id != "t3" for c in cases)


def test_unknown_task_id_is_excluded():
    cases, excluded = assemble(FROZEN, ["nope"])
    assert excluded == ["nope"] and cases == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/eval/test_fusion_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/fusion/__init__.py
"""M5 domestic fusion panel. Isolated from gateway.* like the rest of evaluator/."""
```

```python
# evaluator/fusion/panel.py
"""Assemble per-task candidate answers from frozen single-model runs ($0).

Candidates come from the frozen M2c/M2d outputs, so building a panel costs
nothing; only cross-review and fusion make new API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PanelCase:
    task_id: str
    source: str
    candidates: dict[str, str]  # model name -> its answer text


def load_frozen_by_model(run_dirs: dict[str, str]) -> dict[str, dict[str, str]]:
    """model -> {task_id: output_text} for status=="ok" rows only."""
    from evaluator.store import read_frozen

    out: dict[str, dict[str, str]] = {}
    for model, run_dir in run_dirs.items():
        got: dict[str, str] = {}
        for fo in read_frozen(Path(run_dir)):
            if fo.status == "ok" and fo.task_id not in got:
                got[fo.task_id] = fo.output_text
        out[model] = got
    return out


def assemble(frozen_by_model: dict[str, dict[str, str]], task_ids: list[str],
             min_candidates: int = 2,
             sources: dict[str, str] | None = None) -> tuple[list[PanelCase], list[str]]:
    """Build one PanelCase per task from whatever candidates exist.

    A task with fewer than `min_candidates` answers is excluded (reported), not
    an error — models legitimately error or run out of quota on some tasks.
    """
    cases: list[PanelCase] = []
    excluded: list[str] = []
    for tid in task_ids:
        cands = {m: texts[tid] for m, texts in frozen_by_model.items() if tid in texts}
        if len(cands) < min_candidates:
            excluded.append(tid)
            continue
        cases.append(PanelCase(task_id=tid,
                               source=(sources or {}).get(tid, ""),
                               candidates=cands))
    return cases, excluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/eval/test_fusion_panel.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full eval suite**

Run: `.venv/bin/pytest tests/eval -q`
Expected: `131 passed` (128 baseline + 3)

- [ ] **Step 6: Commit**

```bash
git add evaluator/fusion/__init__.py evaluator/fusion/panel.py tests/eval/test_fusion_panel.py
git commit -m "feat(M5): panel assembly from frozen candidates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Review + fusion prompts (official answer format preserved)

**Files:**
- Create: `evaluator/fusion/prompts.py`
- Test: `tests/eval/test_fusion_prompts.py`

**Interfaces:**
- Produces: `build_review_prompt(task, case, reviewer: str) -> str`; `build_fusion_prompt(task, case, reviews: dict[str, dict[str, "Verdict"]]) -> str`; `format_instruction(task) -> str`.
- Consumes: `evaluator.suite.types.Task` (fields `id, source, problem, answer, tests, meta`); `evaluator.official.prompts` module-level `_TEMPLATES` is NOT imported directly — instead `format_instruction` derives the per-source answer-format sentence via the mapping below.

> **Why a local mapping:** `evaluator/official/prompts.py` exposes only `build(task)` (full prompt). Fusion needs just the *answer-format* sentence appended to a different prompt body, so this task defines `_FORMAT` with wording copied verbatim from the official templates. Keep them identical — a fused answer must satisfy the same extractor the grader uses.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_fusion_prompts.py
from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.prompts import (build_fusion_prompt, build_review_prompt,
                                      format_instruction)
from evaluator.fusion.review import Verdict

MCQ = Task(id="t1", source="mmlu_pro", problem="What is 2+2?\nA) 3\nB) 4",
           answer="B", tests=(), meta={})
MATH = Task(id="t2", source="math", problem="Compute 2+2.", answer="4",
            tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"deepseek-chat": "I say (A).", "glm-5.2": "I say (B)."})


def test_format_instruction_matches_official_wording():
    assert "The answer is (X)." in format_instruction(MCQ)
    assert "\\boxed{}" in format_instruction(MATH)


def test_review_prompt_excludes_the_reviewers_own_answer():
    p = build_review_prompt(MCQ, CASE, reviewer="deepseek-chat")
    assert "I say (B)." in p          # the other candidate is reviewed
    assert "I say (A)." not in p      # its own answer is NOT shown (no self-review)
    assert "correct" in p and "wrong" in p and "unsure" in p   # verdict vocabulary


def test_fusion_prompt_carries_candidates_reviews_and_format():
    reviews = {"deepseek-chat": {"glm-5.2": Verdict("wrong", "B is not 4")}}
    p = build_fusion_prompt(MCQ, CASE, reviews)
    assert "I say (A)." in p and "I say (B)." in p     # all candidates present
    assert "B is not 4" in p                           # review evidence present
    assert "The answer is (X)." in p                   # official output contract
    assert MCQ.problem in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/eval/test_fusion_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.fusion.prompts'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/fusion/prompts.py
"""Prompts for cross-review and fusion.

The answer-format sentences are copied verbatim from
`evaluator/official/prompts.py` so a fused answer satisfies exactly the
extractor the official grader uses. If those templates change, change these.
"""
from __future__ import annotations

_MCQ_FMT = ("Finish your response with a single line 'The answer is (X).' "
            "where X is the correct option letter.")
_MATH_FMT = "Put your final answer within \\boxed{}."
_CODE_FMT = "Respond with a single Python code block containing the complete solution."
_DEFAULT_FMT = "Put your final answer clearly at the end."

_FORMAT = {"mmlu_pro": _MCQ_FMT, "gpqa_diamond": _MCQ_FMT,
           "math": _MATH_FMT, "aime": _MATH_FMT, "math_l5": _MATH_FMT,
           "humaneval": _CODE_FMT, "livecodebench": _CODE_FMT}


def format_instruction(task) -> str:
    return _FORMAT.get(task.source, _DEFAULT_FMT)


def _candidate_block(candidates: dict[str, str], exclude: str | None = None) -> str:
    parts = []
    for model, text in sorted(candidates.items()):
        if model == exclude:
            continue
        parts.append(f"--- Candidate {model} ---\n{text}")
    return "\n\n".join(parts)


def build_review_prompt(task, case, reviewer: str) -> str:
    """Ask `reviewer` to judge the OTHER candidates (never its own answer)."""
    return (
        "You are reviewing other models' answers to a problem. For EACH candidate "
        "below, judge whether its final answer is correct.\n\n"
        f"Problem:\n{task.problem}\n\n"
        f"{_candidate_block(case.candidates, exclude=reviewer)}\n\n"
        "For each candidate, output one line in exactly this format:\n"
        "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>\n"
        "Judge only correctness of the final answer, not style."
    )


def build_fusion_prompt(task, case, reviews) -> str:
    """Fuser sees every candidate plus the cross-review evidence."""
    lines = []
    for reviewer, verdicts in sorted(reviews.items()):
        for target, v in sorted(verdicts.items()):
            lines.append(f"{reviewer} says {target} is {v.verdict}: {v.reason}")
    review_block = "\n".join(lines) if lines else "(no reviews available)"
    return (
        "Several models answered the problem below, and reviewed each other. "
        "Produce the single best final answer.\n\n"
        f"Problem:\n{task.problem}\n\n"
        f"{_candidate_block(case.candidates)}\n\n"
        f"--- Peer review ---\n{review_block}\n\n"
        "Rules:\n"
        "- If the candidates agree and no review objects, adopt that answer.\n"
        "- If they disagree, decide using the specific objections raised, and "
        "write a corrected answer (you may combine correct parts of several).\n"
        f"- {format_instruction(task)}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/eval/test_fusion_prompts.py -v`
Expected: PASS (3 tests). Note this test imports `Verdict` from Task 3 — if Task 3 is not yet implemented, implement `review.Verdict` first (it is a 2-field dataclass; see Task 3 Step 3).

- [ ] **Step 5: Commit**

```bash
git add evaluator/fusion/prompts.py tests/eval/test_fusion_prompts.py
git commit -m "feat(M5): review + fusion prompts preserving official answer format

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Structured cross-review

**Files:**
- Create: `evaluator/fusion/review.py`
- Test: `tests/eval/test_fusion_review.py`

**Interfaces:**
- Produces: `Verdict(verdict: str, reason: str)` (verdict ∈ `"correct" | "wrong" | "unsure"`); `parse_review(text: str, valid_targets: set[str]) -> dict[str, Verdict]`; `cross_review(task, case, completion_fn) -> dict[str, dict[str, Verdict]]` (reviewer → target → Verdict).
- Consumes: `build_review_prompt` (Task 2); `completion_fn(model, prompt) -> {"text", "in_tokens", "out_tokens", "cost_usd"}` — the same shape `evaluator/validate.py::make_completion_fn` returns.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_fusion_review.py
from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.review import Verdict, cross_review, parse_review

TASK = Task(id="t1", source="mmlu_pro", problem="2+2?", answer="B", tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"a-model": "says A", "b-model": "says B"})


def test_parse_review_reads_verdict_lines():
    out = parse_review(
        "Some preamble\n"
        "VERDICT a-model wrong picked A but 4 is right\n"
        "VERDICT b-model correct matches 4\n",
        valid_targets={"a-model", "b-model"})
    assert out["a-model"] == Verdict("wrong", "picked A but 4 is right")
    assert out["b-model"].verdict == "correct"


def test_parse_review_drops_malformed_and_unknown_lines():
    out = parse_review(
        "VERDICT a-model banana nonsense verdict\n"     # invalid verdict word
        "VERDICT ghost-model correct not in panel\n"     # unknown target
        "totally unrelated line\n"
        "VERDICT b-model unsure cannot tell\n",
        valid_targets={"a-model", "b-model"})
    assert set(out) == {"b-model"}
    assert out["b-model"].verdict == "unsure"


def test_cross_review_never_asks_a_model_about_itself():
    seen = {}

    def fake_completion(model, prompt):
        seen[model] = prompt
        other = "b-model" if model == "a-model" else "a-model"
        return {"text": f"VERDICT {other} correct looks right",
                "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

    reviews = cross_review(TASK, CASE, fake_completion)
    assert set(reviews) == {"a-model", "b-model"}
    assert set(reviews["a-model"]) == {"b-model"}     # reviewed only the other
    assert "says A" not in seen["a-model"]            # own answer not in its prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/eval/test_fusion_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.fusion.review'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/fusion/review.py
"""Cross-review: each panel model judges the OTHER candidates.

Output is structured (VERDICT lines), not prose, so the fuser receives evidence
and reviewer agreement is measurable. Reviews inform fusion only — grading stays
objective (evaluator/scorers/*). Malformed lines are dropped, never fatal.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.fusion.prompts import build_review_prompt

VALID = {"correct", "wrong", "unsure"}


@dataclass(frozen=True)
class Verdict:
    verdict: str  # "correct" | "wrong" | "unsure"
    reason: str


def parse_review(text: str, valid_targets: set[str]) -> dict[str, Verdict]:
    out: dict[str, Verdict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3 or parts[0] != "VERDICT":
            continue
        target, verdict = parts[1], parts[2].lower()
        if target not in valid_targets or verdict not in VALID:
            continue
        out[target] = Verdict(verdict, parts[3] if len(parts) > 3 else "")
    return out


def cross_review(task, case, completion_fn) -> dict[str, dict[str, Verdict]]:
    """reviewer -> {target: Verdict}. Each model reviews only the others."""
    reviews: dict[str, dict[str, Verdict]] = {}
    for reviewer in sorted(case.candidates):
        targets = {m for m in case.candidates if m != reviewer}
        if not targets:
            continue
        prompt = build_review_prompt(task, case, reviewer=reviewer)
        try:
            text = completion_fn(reviewer, prompt)["text"]
        except Exception:
            reviews[reviewer] = {}
            continue
        reviews[reviewer] = parse_review(text, targets)
    return reviews
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/eval/test_fusion_review.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluator/fusion/review.py tests/eval/test_fusion_review.py
git commit -m "feat(M5): structured cross-review (no self-review, malformed-safe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Fuser

**Files:**
- Create: `evaluator/fusion/fuse.py`
- Test: `tests/eval/test_fusion_fuse.py`

**Interfaces:**
- Produces: `FusionResult(task_id: str, answer: str, fuser: str, cost_usd: float, status: str, error: str | None)`; `fuse(task, case, reviews, completion_fn, fuser: str = "glm-5.2") -> FusionResult`.
- Consumes: `build_fusion_prompt` (Task 2), `Verdict` (Task 3), the same `completion_fn` shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_fusion_fuse.py
from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.review import Verdict
from evaluator.fusion.fuse import FusionResult, fuse

TASK = Task(id="t1", source="mmlu_pro", problem="2+2?", answer="B", tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"a-model": "says A", "b-model": "says B"})
REVIEWS = {"a-model": {"b-model": Verdict("correct", "4 is right")}}


def test_fuse_calls_the_named_fuser_and_returns_its_answer():
    calls = []

    def fake_completion(model, prompt):
        calls.append((model, prompt))
        return {"text": "The answer is (B).", "in_tokens": 5, "out_tokens": 3,
                "cost_usd": 0.002}

    res = fuse(TASK, CASE, REVIEWS, fake_completion, fuser="glm-5.2")
    assert isinstance(res, FusionResult)
    assert res.answer == "The answer is (B)."
    assert res.fuser == "glm-5.2" and calls[0][0] == "glm-5.2"
    assert res.cost_usd == 0.002 and res.status == "ok"
    assert "4 is right" in calls[0][1]        # review evidence reached the fuser


def test_fuse_records_error_instead_of_raising():
    def boom(model, prompt):
        raise RuntimeError("mirror 503")

    res = fuse(TASK, CASE, REVIEWS, boom, fuser="glm-5.2")
    assert res.status == "error" and res.answer == "" and res.cost_usd == 0.0
    assert "mirror 503" in (res.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/eval/test_fusion_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluator.fusion.fuse'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluator/fusion/fuse.py
"""The fuser: reads task + all candidates + cross-review, writes the final answer.

The fuser is a DOMESTIC model (default glm-5.2) — deliberately not the strongest
candidate, so fusion cannot degenerate into "always echo deepseek". Using a
frontier model here would make the headline claim circular.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluator.fusion.prompts import build_fusion_prompt


@dataclass(frozen=True)
class FusionResult:
    task_id: str
    answer: str
    fuser: str
    cost_usd: float
    status: str          # "ok" | "error"
    error: str | None


def fuse(task, case, reviews, completion_fn, fuser: str = "glm-5.2") -> FusionResult:
    prompt = build_fusion_prompt(task, case, reviews)
    try:
        got = completion_fn(fuser, prompt)
    except Exception as exc:  # never crash a batch on one task
        return FusionResult(case.task_id, "", fuser, 0.0, "error", str(exc))
    return FusionResult(case.task_id, got.get("text", "") or "", fuser,
                        float(got.get("cost_usd", 0.0) or 0.0), "ok", None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/eval/test_fusion_fuse.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluator/fusion/fuse.py tests/eval/test_fusion_fuse.py
git commit -m "feat(M5): domestic fuser (synthesize final answer from panel + reviews)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Report math — oracle, fix/break, agreement, gate curve + isolation test

**Files:**
- Create: `scripts/fusion_report.py`
- Test: `tests/eval/test_fusion_report.py`
- Test: `tests/eval/test_fusion_isolation.py`

**Interfaces:**
- Produces: `oracle(correct_by_model: dict[str, dict[str, bool]]) -> float`; `fix_break(baseline: dict[str, bool], fused: dict[str, bool]) -> dict` returning keys `fix`, `break_`, `net`, `n`; `reviewer_agreement(reviews_by_task: dict[str, dict[str, dict[str, str]]]) -> float`; `gate_curve(candidates_by_task: dict[str, dict[str, str]], fused_correct: dict[str, bool], baseline_correct: dict[str, bool], fusion_cost: dict[str, float]) -> list[dict]`.
- Consumes: `wilson_ci` from `scripts/hard_report.py` (signature `wilson_ci(k, n, z=1.96)`) and `mcnemar_p(b, c)` — reuse, do not reimplement. (`scripts` is importable because `pyproject.toml` sets `pythonpath=["."]`.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_fusion_report.py
from scripts.fusion_report import fix_break, gate_curve, oracle, reviewer_agreement


def test_oracle_is_fraction_where_any_model_is_correct():
    correct = {
        "m1": {"t1": True,  "t2": False, "t3": False},
        "m2": {"t1": False, "t2": True,  "t3": False},
    }
    assert oracle(correct) == 2 / 3        # t1 and t2 covered, t3 by nobody


def test_fix_break_counts_both_directions():
    base  = {"t1": False, "t2": True,  "t3": True,  "t4": False}
    fused = {"t1": True,  "t2": False, "t3": True,  "t4": False}
    r = fix_break(base, fused)
    assert r["fix"] == 1 and r["break_"] == 1 and r["net"] == 0 and r["n"] == 4


def test_reviewer_agreement_fraction_of_matching_verdicts_on_same_target():
    # task t1: two reviewers both call m3 "wrong" (agree);
    # task t2: reviewers disagree about m3
    reviews = {
        "t1": {"m1": {"m3": "wrong"}, "m2": {"m3": "wrong"}},
        "t2": {"m1": {"m3": "correct"}, "m2": {"m3": "wrong"}},
    }
    assert reviewer_agreement(reviews) == 0.5


def test_gate_curve_reports_agreement_gate_cheaper_than_always_fuse():
    cands = {"t1": {"a": "X", "b": "X"},     # agree -> gate can skip fusion
             "t2": {"a": "X", "b": "Y"}}     # disagree -> must fuse
    fused = {"t1": True, "t2": True}
    base  = {"t1": True, "t2": False}
    cost  = {"t1": 0.01, "t2": 0.01}
    rows = {r["policy"]: r for r in gate_curve(cands, fused, base, cost)}
    assert rows["always"]["cost"] == 0.02 and rows["always"]["correct"] == 2
    # agreement gate: t1 adopts the (correct) agreed answer free, t2 pays
    assert rows["on_disagreement"]["cost"] == 0.01
    assert rows["on_disagreement"]["correct"] == 2
```

```python
# tests/eval/test_fusion_isolation.py
import ast
import pathlib

FUSION = pathlib.Path("evaluator/fusion")


def test_fusion_modules_never_import_gateway():
    for py in FUSION.glob("*.py"):
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

Run: `.venv/bin/pytest tests/eval/test_fusion_report.py tests/eval/test_fusion_isolation.py -v`
Expected: `test_fusion_report` FAILs (`ModuleNotFoundError: scripts.fusion_report`); `test_fusion_isolation` PASSES already (fusion modules are clean).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fusion_report.py
"""M5 fusion metrics: oracle ceiling, fix/break, reviewer agreement, gate curve.

Reuses scripts/hard_report.wilson_ci and mcnemar_p for the report rendering.
"""
from __future__ import annotations

from scripts.hard_report import mcnemar_p, wilson_ci  # noqa: F401  (used by main)


def oracle(correct_by_model: dict[str, dict[str, bool]]) -> float:
    """Fraction of tasks where AT LEAST ONE model is correct (fusion's ceiling)."""
    task_ids: set[str] = set()
    for per_task in correct_by_model.values():
        task_ids |= set(per_task)
    if not task_ids:
        return 0.0
    hit = sum(1 for t in task_ids
              if any(per_task.get(t, False) for per_task in correct_by_model.values()))
    return hit / len(task_ids)


def fix_break(baseline: dict[str, bool], fused: dict[str, bool]) -> dict:
    """fix = baseline wrong -> fused right; break_ = baseline right -> fused wrong."""
    ids = [t for t in baseline if t in fused]
    fix = sum(1 for t in ids if not baseline[t] and fused[t])
    brk = sum(1 for t in ids if baseline[t] and not fused[t])
    return {"fix": fix, "break_": brk, "net": fix - brk, "n": len(ids)}


def reviewer_agreement(reviews_by_task: dict[str, dict[str, dict[str, str]]]) -> float:
    """Of (task, target) pairs judged by >=2 reviewers, fraction where all agree."""
    pairs = 0
    agree = 0
    for _tid, by_reviewer in reviews_by_task.items():
        targets: dict[str, list[str]] = {}
        for _reviewer, verdicts in by_reviewer.items():
            for target, verdict in verdicts.items():
                targets.setdefault(target, []).append(verdict)
        for _target, verdicts in targets.items():
            if len(verdicts) < 2:
                continue
            pairs += 1
            if len(set(verdicts)) == 1:
                agree += 1
    return (agree / pairs) if pairs else float("nan")


def gate_curve(candidates_by_task: dict[str, dict[str, str]],
               fused_correct: dict[str, bool],
               baseline_correct: dict[str, bool],
               fusion_cost: dict[str, float]) -> list[dict]:
    """Simulate gate policies over frozen results ($0).

    - "always": fuse every task.
    - "on_disagreement": if all candidate answers are identical, adopt that
      answer for free (its correctness equals the baseline's on that task);
      otherwise pay for fusion.
    """
    ids = sorted(fused_correct)
    rows = []
    always_cost = sum(fusion_cost.get(t, 0.0) for t in ids)
    rows.append({"policy": "always",
                 "cost": always_cost,
                 "correct": sum(1 for t in ids if fused_correct[t]),
                 "n": len(ids)})
    cost = 0.0
    correct = 0
    for t in ids:
        answers = set(candidates_by_task.get(t, {}).values())
        if len(answers) <= 1:                     # unanimous -> free adopt
            correct += 1 if baseline_correct.get(t, False) else 0
        else:
            cost += fusion_cost.get(t, 0.0)
            correct += 1 if fused_correct[t] else 0
    rows.append({"policy": "on_disagreement", "cost": cost, "correct": correct,
                 "n": len(ids)})
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/eval/test_fusion_report.py tests/eval/test_fusion_isolation.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Full Phase-A sweep + commit**

Run: `.venv/bin/pytest tests/eval -q`
Expected: `141 passed` (128 baseline + 13 new). Phase A is proven with **zero network and zero spend**.

```bash
git add scripts/fusion_report.py tests/eval/test_fusion_report.py tests/eval/test_fusion_isolation.py
git commit -m "feat(M5): fusion metrics (oracle, fix/break, agreement, gate curve) + isolation test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase B — paid execution (keys + spend; user-gated)

### Task 6: Sample kimi-k3 candidates on the standard tier

**Files:**
- Create: `scripts/run_fusion.py` (subcommand `sample-kimi`)

**Interfaces:**
- Produces: frozen kimi-k3 outputs at `evaluator/runs/m5_fusion/kimi-k3/` covering the standard tier.
- Consumes: `evaluator.validate.MODELS["kimi-k3"]`, `scripts.resample_official.run_budgeted(models, tasks, run_dir, ceiling, cost_fn)`, `evaluator.suite.manifest.load` + `evaluator.suite.loader.load_suite`.

- [ ] **Step 1: Verify kimi quota with a 5-task probe**
Run kimi-k3 on 5 standard-tier tasks through `run_budgeted` with a $0.50 ceiling. Expected: 5 rows with `status="ok"` and non-empty `output_text`. If quota is exhausted, STOP and report to the user (the panel's third member is unavailable).

- [ ] **Step 2: Sample the full standard tier**
Run kimi-k3 over all 1063 standard-tier tasks, resumable, budget-gated at **$5**, into `evaluator/runs/m5_fusion/kimi-k3/`.
Expected: ≥95% rows `ok`; report the count. Errors are retried by re-running (resume skips completed tasks).

- [ ] **Step 3: Report panel completeness + commit the driver**

```bash
git add scripts/run_fusion.py
git commit -m "feat(M5): fusion driver — kimi-k3 standard-tier sampling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: The free oracle gate (GO/NO-GO)

**Files:**
- Modify: `scripts/run_fusion.py` (subcommand `oracle`)

- [ ] **Step 1: Score all three domestic models from frozen outputs ($0)**
Load frozen deepseek-chat + glm-5.2 (from `evaluator/runs/m2c_full/<model>/`) and kimi-k3 (Task 6), score each with the official scorers (`evaluator/validate.py::SCORERS` maps source → scorer), producing `correct_by_model: dict[model, dict[task_id, bool]]` over the common set.

- [ ] **Step 2: Compute and report the oracle**
Run `oracle(correct_by_model)` plus each model's individual accuracy.
Expected output: the oracle value, per-model accuracies, and the common-set size.

- [ ] **Step 3: GO/NO-GO decision**
- **oracle ≥ 0.894** → the pool can in principle reach gpt-5.6-sol; proceed to Task 8.
- **oracle < 0.894** → STOP. Write the finding to `docs/M5_FUSION_REPORT.md` (pool insufficient; changing the pool, not the prompts, is the fix) and report to the user. Do not spend on review/fusion.

- [ ] **Step 4: Commit the oracle result**

```bash
git add scripts/run_fusion.py docs/M5_FUSION_REPORT.md
git commit -m "feat(M5): free oracle gate over the domestic panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 20-task paid smoke

**Files:**
- Modify: `scripts/run_fusion.py` (subcommand `run`, with `--limit`)

- [ ] **Step 1: Run cross-review + fusion on 20 tasks**
Stratified across sources (mmlu_pro / math / humaneval), ceiling **$2**, frozen to `evaluator/runs/m5_fusion/fused_smoke/`.
Expected: 20 fused answers, `status="ok"`, measured per-task cost.

- [ ] **Step 2: Score the smoke + first diagnostics**
Score fused answers with the official scorers; report fused accuracy, `fix_break` vs the best domestic single model, and the **format failure rate** (fused answers the extractor cannot parse).
Expected: format failure rate near zero. If it is high, fix the fusion prompt's format instruction before the full run — a formatting bug would corrupt the whole result.

- [ ] **Step 3: Report measured cost and project the full run**
Append to `docs/M5_FUSION_REPORT.md`: measured $/task and the projected 1063-task cost vs the $25 ceiling. If the projection exceeds the ceiling, report to the user before proceeding.

---

### Task 9: Full standard-tier fusion run

**Files:** (no new files — executes `scripts/run_fusion.py run`)

- [ ] **Step 1: Run cross-review + fusion over the full common set**
Ceiling **$25**, resumable, frozen to `evaluator/runs/m5_fusion/fused_full/`. Re-run to resume after transient mirror errors.
Expected: fused answers for the full common set; error rows retried to near-zero.

- [ ] **Step 2: Score everything and freeze**
Score fused answers + all three domestic models + read the M2c frontier numbers. All re-scorable at $0 afterwards.

---

### Task 10: Publish the report

**Files:**
- Create/replace: `docs/M5_FUSION_REPORT.md`
- Modify: `scripts/fusion_report.py` (add `main()` that renders the report from frozen data)
- Modify: `README.md` (roadmap row for M5)

- [ ] **Step 1: Compute the headline table**
Fusion accuracy with **Wilson 95% CI**, versus best domestic single (deepseek-chat 0.859), gpt-5.6-sol (0.894), and claude-opus-4-8 (0.913); **McNemar** p-values vs the best domestic single and vs gpt-5.6-sol.

- [ ] **Step 2: Compute the diagnostics**
`fix_break` vs the best domestic single (fix / break / net), `reviewer_agreement`, format failure rate, panel completeness (tasks excluded for <2 candidates).

- [ ] **Step 3: Simulate the gate curve ($0)**
`gate_curve(...)` over the frozen results → the cost–quality operating points (`always` vs `on_disagreement`).

- [ ] **Step 4: Write the verdict**
Does the domestic panel reach frontier single-model quality? Report cost per correct answer, the net gain, and an explicit recommendation on (a) the hard tier and (b) whether fusion should become a gateway strategy.

- [ ] **Step 5: Full test sweep + commit**

Run: `.venv/bin/pytest tests/eval -q`
Expected: PASS.

```bash
git add docs/M5_FUSION_REPORT.md scripts/fusion_report.py README.md
git commit -m "docs(M5): domestic fusion panel report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Hard tier (conditional)

- [ ] **Step 1: Decide from the standard-tier result**
If fusion showed net gain > 0 on the standard tier, repeat Tasks 6–10 against `configs/suite.hard.manifest.json` (657 tasks; kimi hard-tier candidates must be sampled first), ceiling **$25**.
If net gain ≤ 0, **explicitly defer** the hard tier and record the standard-tier rationale in `docs/M5_FUSION_REPORT.md` — do not spend.

- [ ] **Step 2: Commit the hard-tier section or the deferral note**

```bash
git add docs/M5_FUSION_REPORT.md
git commit -m "docs(M5): hard-tier fusion result (or documented deferral)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Mechanism (candidates → cross-review → fuser synthesizes) → Tasks 1–4. ✓
- Domestic-only panel + fuser glm-5.2, no frontier anywhere → Global Constraints + Task 4 default. ✓
- Candidates from frozen runs ($0) → Task 1 (`load_frozen_by_model`). ✓
- Official answer format preserved in fusion → Task 2 (`format_instruction`, verbatim wording). ✓
- No LLM judge in scoring → Global Constraints; scoring reuses `evaluator/scorers/*` (Tasks 7–9). ✓
- Structured verdicts, no self-review, malformed-safe → Task 3. ✓
- kimi sampling first (quota refreshed) → Task 6. ✓
- Free oracle gate with stop rule → Task 7. ✓
- 20-task paid smoke → Task 8. ✓
- Full run, $25 ceiling, resumable, frozen → Task 9. ✓
- fix/break/net gain, reviewer agreement, format failure rate → Tasks 5, 8, 10. ✓
- Post-hoc $0 gate simulation → Tasks 5, 10 (`gate_curve`). ✓
- Report with Wilson CI + McNemar vs best domestic and vs sol → Task 10. ✓
- Hard tier repeated or explicitly deferred → Task 11. ✓
- Isolation, manifests unchanged, budget gate, secrets → Global Constraints + Task 5 isolation test. ✓

**Placeholder scan:** No TBD/TODO. All Phase-A code and tests are complete and runnable. Phase B tasks are execution steps with exact ceilings, run dirs, and stop rules (they invoke code built in Phase A plus the existing `run_budgeted`).

**Type consistency:** `PanelCase(task_id, source, candidates)` used identically in Tasks 1–4; `Verdict(verdict, reason)` defined in Task 3 and consumed in Tasks 2 (test), 4; `FusionResult` fields match its test; `fix_break` returns `fix`/`break_`/`net`/`n` as asserted; `wilson_ci(k, n)` / `mcnemar_p(b, c)` match `scripts/hard_report.py`. Task 2's test imports `Verdict` from Task 3 — flagged inline in Task 2 Step 4.

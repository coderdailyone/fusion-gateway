# M2d Hard-Tier Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate, locked "hard tier" benchmark (LiveCodeBench + GPQA-Diamond + AIME 2024/2025 + MATH level-5) that de-ties the frontier and makes contamination a measurable fresh-vs-public signal, scored with the same official-alignment discipline as M2c.

**Architecture:** Four new sources feed a separate `configs/suite.hard.manifest.json`. The only new scorer is LiveCodeBench test normalization (`evaluator/official/livecodebench_exec.py`) — GPQA reuses the MCQ chain, AIME/MATH-L5 reuse the math grader, and LCB execution reuses `evaluator/scorers/code.py` once its tests are normalized into the existing `{"kind":"stdin"...}` / `{"kind":"pyfunc"...}` shapes. Fetchers, `parse_task`, prompt templates, the manifest builder, the reference self-test, and a report script extend to the four sources. Sampling reuses the M2c per-model-parallel + budget-gated infrastructure.

**Tech Stack:** Python 3, `datasets` (lazy, fetch only), existing `evaluator.sandbox`, `sklearn`/`scipy`-free Wilson CI (hand-rolled), `pytest`. No new third-party deps.

## Global Constraints

- **Isolation:** nothing under `evaluator/` or `scripts/` may `import gateway.*` or touch the gateway SQLite.
- **Standard tier untouched:** `configs/suite.manifest.json` and the M2c results must not change. The hard tier is a *separate* manifest + separate run dirs + separate report.
- **Determinism:** scorers and the GPQA option shuffle are pure/seed-stable; re-scoring a frozen output twice is identical; no `random` at score time.
- **Locked suite:** `configs/suite.hard.manifest.json` pins each source's dataset revision + task ids + `content_sha`; `load_suite` re-verifies the hash. Never mutate the frozen records after building.
- **Official scoring:** LCB normalization carries upstream provenance (repo + real commit + license, **verified live at implement time** — M2c found fabricated SHAs/wrong licenses). AIME/MATH-L5 reuse `evaluator/official/math_grade`; GPQA reuses `evaluator/official/mmlu_extract`.
- **Source→scorer map** (canonical, used by scoring + self-test): `mmlu_pro`/`gpqa_diamond`→`mcq.score`; `math`/`aime`/`math_l5`→`math.score`; `humaneval`/`livecodebench`→`code.score`.
- **Test command:** `.venv/bin/pytest <paths> -q` (repo root on `pythonpath`, works for `scripts.*` imports). Baseline before this plan: 79 in tests/eval.
- **Sampling models (paid step):** `deepseek-chat, claude-sonnet-5, claude-opus-4-8, gpt-5.6-sol, gpt-5.5, glm-5.2`. **Kimi excluded** (account quota).
- **Dataset ids/revisions** are resolved + pinned at build time via `dataset_info(...).sha`; verify each dataset is reachable before building and pin an equivalent if a named id is unavailable.

---

## Existing interfaces this plan builds on (do not change their signatures)

- `evaluator/hf_fetchers.py`: `extract(source_name, row) -> (task_id, record)`, `stratum(source_name, row) -> str`, `load_id_map(name, hf_dataset, revision, split) -> (id_map, strata)`, `make_fetcher(name)`.
- `evaluator/suite/loader.py`: `parse_task(source, record) -> Task`; `Task(id, source, problem, answer: str|None, tests: tuple[dict,...], meta: dict)`.
- `evaluator/suite/manifest.py`: `SourceSpec(name, hf_dataset, hf_revision, split, task_ids, content_sha)`, `Manifest(version, sources)`, `content_sha(records)`, `save(m, path)`, `load(path)`.
- `evaluator/build_manifest.py`: `SOURCES`, `stratified_sample(ids, strata, n, seed)`, `build(seed)`, `main()`.
- `evaluator/scorers/code.py`: `score(task, output_text, runner=run_code) -> Score`; `_run_case` handles `{"entry_point","test"}` (pyfunc) and `{"stdin","expected_stdout"}` (stdin).
- `evaluator/scorers/mcq.py`: `score(task, output_text) -> Score` (via `mmlu_extract.extract_answer`). `evaluator/scorers/math.py`: `score(task, output_text) -> Score` (via `math_grade.math_equiv`).
- `evaluator/official/prompts.py`: `build(task) -> str`, `_TEMPLATES` dict keyed by `task.source`, `_DEFAULT`.
- `evaluator/sandbox.py`: `run_code(code, stdin="", timeout_s=5.0, mem_mb=256, cpu_s=5) -> SandboxResult(status, stdout, stderr, returncode)`.
- `evaluator/audit/references.py`: `Reference(...)`, `build_reference_index(rows_by_source)`, `load_references(manifest, cache_path, fetch)`. `evaluator/audit/reference_selftest.py`: `_synth_output`, `_SCORERS`, `selftest_one`, `selftest`.
- `scripts/resample_official.py`: `run_budgeted(models, tasks, run_dir, ceiling, cost_fn, deps=None)`, `budget_gate`, `estimate_cost`.

---

## File structure

Create:
- `evaluator/official/livecodebench_exec.py` — LCB test-case normalization → generic test shapes (the only new scorer logic).
- `scripts/hard_report.py` — per-model hard-tier accuracy + Wilson CI + pairwise significance + fresh-vs-public contamination.
- `docs/HARD_TIER_REPORT.md` — generated report skeleton (Task 8 fills numbers).
- Tests: `tests/eval/test_official_livecodebench.py`, `test_fetchers_hard.py`, `test_hard_manifest.py`, `test_reference_selftest_hard.py`, `test_hard_report.py`.

Modify:
- `evaluator/hf_fetchers.py` — `extract`/`stratum` for `livecodebench`, `gpqa_diamond`, `aime`, `math_l5`.
- `evaluator/suite/loader.py` — `parse_task` for `gpqa_diamond`, `aime`, `math_l5` (livecodebench stub already present; align it).
- `evaluator/official/prompts.py` — templates for the four new sources.
- `evaluator/build_manifest.py` — `HARD_SOURCES` + filter predicates + `build_hard()` + CLI.
- `evaluator/audit/references.py` — `build_reference_index` for the four sources.
- `evaluator/audit/reference_selftest.py` — `_SCORERS` + `_synth_output` for the four sources.
- `configs/suite.hard.manifest.json` — built in Task 5 (manual, network).

---

### Task 1: LiveCodeBench test normalization (`official/livecodebench_exec.py`)

**Files:**
- Create: `evaluator/official/livecodebench_exec.py`, `tests/eval/test_official_livecodebench.py`

**Interfaces:**
- Produces: `normalize_tests(row: dict) -> tuple[dict, ...]` — turns one LCB problem's test cases into the generic shapes `evaluator/scorers/code.py` executes: stdin cases → `{"kind":"stdin","stdin":<in>,"expected_stdout":<out>}`; functional cases → one `{"kind":"pyfunc","test":<check-src>,"entry_point":<fn_name>}` whose `check(candidate)` asserts `candidate(*parsed_args)==expected` for each case.
- Consumes: nothing (pure). Executed later by `code.score` via `evaluator.sandbox.run_code`.

**Design note:** LCB's `code_generation_lite` schema is not fully known ahead of time and `private_test_cases` may be zlib+base64 encoded. **Step 0: fetch one real row and print its keys + a sample test case** before writing the parser, and adapt field names to what you observe (record the observed schema in the module docstring). The functional-case parser must turn LCB's input representation into Python call args — LCB functional inputs are typically newline-joined literals; parse each line with `ast.literal_eval`, fall back to the raw string.

- [ ] **Step 0: Inspect a real LCB row** (network)

Run:
```bash
.venv/bin/python -c "from datasets import load_dataset; d=load_dataset('livecodebench/code_generation_lite', split='test', trust_remote_code=True); r=d[0]; print(sorted(r.keys())); print({k:(str(r[k])[:200]) for k in ['question_id','difficulty','starter_code','public_test_cases','metadata'] if k in r})"
```
Record the actual field names (question id, test-case fields, functional fn_name location, release/contest date field) in the module docstring. If the id/split differs, note it for Task 2's fetcher.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_official_livecodebench.py`

```python
from evaluator.official.livecodebench_exec import normalize_tests, build_functional_check

def test_stdin_case_shape():
    row = {"test_type": "stdin",
           "cases": [{"input": "3\n", "output": "9\n"}]}
    out = normalize_tests(row)
    assert out == ({"kind": "stdin", "stdin": "3\n", "expected_stdout": "9\n"},)

def test_functional_check_asserts_each_case():
    src = build_functional_check("sq", [("3", "9"), ("4", "16")])
    assert "def check(candidate):" in src
    assert "candidate(3) == 9" in src
    assert "candidate(4) == 16" in src

def test_functional_case_shape():
    row = {"test_type": "functional", "fn_name": "sq",
           "cases": [{"input": "3", "output": "9"}]}
    out = normalize_tests(row)
    assert len(out) == 1 and out[0]["kind"] == "pyfunc"
    assert out[0]["entry_point"] == "sq"
    assert "candidate(3) == 9" in out[0]["test"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_official_livecodebench.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `evaluator/official/livecodebench_exec.py`**

```python
"""LiveCodeBench test-case normalization into the generic test shapes the
project's code scorer already executes.

Ported to match the LiveCodeBench `code_generation` evaluation protocol
(repo LiveCodeBench/LiveCodeBench — VERIFY real commit + license live and pin
them here at implement time). Only the LCB-specific input parsing + check
assembly live here; execution is delegated to evaluator.scorers.code /
evaluator.sandbox (isolation discipline), not LCB's in-process exec.

`row` is the pre-parsed intermediate shape produced by the fetcher (Task 2):
    {"test_type": "stdin"|"functional", "fn_name": <str|None>,
     "cases": [{"input": <str>, "output": <str>}, ...]}
"""
from __future__ import annotations

import ast


def _parse_args(inp: str):
    """LCB functional inputs are newline-joined Python literals; parse each
    line with literal_eval, falling back to the raw line."""
    args = []
    for line in inp.split("\n"):
        line = line.strip()
        if line == "":
            continue
        try:
            args.append(repr(ast.literal_eval(line)))
        except Exception:
            args.append(repr(line))
    return ", ".join(args)


def build_functional_check(fn_name: str, cases: list[tuple[str, str]]) -> str:
    lines = ["def check(candidate):"]
    for inp, out in cases:
        args = _parse_args(inp)
        try:
            expected = repr(ast.literal_eval(out.strip()))
        except Exception:
            expected = repr(out.strip())
        lines.append(f"    assert candidate({args}) == {expected}")
    lines.append("    return True")
    return "\n".join(lines)


def normalize_tests(row: dict) -> tuple[dict, ...]:
    cases = row.get("cases", [])
    if row.get("test_type") == "functional":
        fn = row["fn_name"]
        check = build_functional_check(fn, [(c["input"], c["output"]) for c in cases])
        return ({"kind": "pyfunc", "test": check, "entry_point": fn},)
    return tuple(
        {"kind": "stdin", "stdin": c["input"], "expected_stdout": c["output"]}
        for c in cases
    )
```

- [ ] **Step 4: Run tests to green**

Run: `.venv/bin/pytest tests/eval/test_official_livecodebench.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluator/official/livecodebench_exec.py tests/eval/test_official_livecodebench.py
git commit -m "feat(eval): LiveCodeBench test normalization (stdin + functional)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: LiveCodeBench fetcher + parse_task + prompt + scorer wiring

**Files:**
- Modify: `evaluator/hf_fetchers.py`, `evaluator/suite/loader.py`, `evaluator/official/prompts.py`
- Test: `tests/eval/test_fetchers_hard.py`

**Interfaces:**
- Consumes: `normalize_tests` (Task 1).
- Produces: an `extract("livecodebench", row)` that returns `(id, {"id","prompt","tests","difficulty","release_date"})` with `tests` already normalized; `parse_task("livecodebench", rec)` returning a `Task` with those `tests`; `SCORERS["livecodebench"] = code.score` wherever the canonical map lives.

**Design note:** the raw LCB row must first be reduced to Task 1's intermediate `{"test_type","fn_name","cases"}` shape inside the fetcher (using the schema you recorded in Task 1 Step 0), then passed through `normalize_tests`. The release-date field is carried for the manifest's date-window filter (Task 5).

- [ ] **Step 1: Write failing test** — add to `tests/eval/test_fetchers_hard.py`

```python
from evaluator.hf_fetchers import extract
from evaluator.suite.loader import parse_task

def test_livecodebench_extract_and_parse(monkeypatch):
    # a minimal LCB-shaped raw row (fields per the schema recorded in Task 1)
    raw = {"question_id": "lcb_1", "question_content": "square it",
           "public_test_cases": '[{"input":"3","output":"9","testtype":"functional"}]',
           "private_test_cases": "[]", "metadata": '{"func_name":"sq"}',
           "starter_code": "", "difficulty": "easy", "contest_date": "2024-09-01"}
    tid, rec = extract("livecodebench", raw)
    assert tid == "lcb_1"
    t = parse_task("livecodebench", rec)
    assert t.source == "livecodebench"
    assert t.tests and t.tests[0]["kind"] == "pyfunc"
    assert t.tests[0]["entry_point"] == "sq"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py -q`
Expected: FAIL (KeyError/ValueError — livecodebench not handled by `extract`).

- [ ] **Step 3: Implement the fetcher** — in `evaluator/hf_fetchers.py::extract`, add before the final `raise`:

```python
    if source_name == "livecodebench":
        import json
        from evaluator.official.livecodebench_exec import normalize_tests
        tid = str(row["question_id"])
        meta = json.loads(row.get("metadata") or "{}")
        fn = meta.get("func_name")
        raw_cases = json.loads(row.get("public_test_cases") or "[]")
        priv = row.get("private_test_cases") or "[]"
        try:
            raw_cases += json.loads(priv)
        except Exception:
            pass  # private may be encoded; public-only is acceptable, note in report
        test_type = "functional" if fn else "stdin"
        cases = [{"input": c["input"], "output": c["output"]} for c in raw_cases]
        tests = normalize_tests({"test_type": test_type, "fn_name": fn, "cases": cases})
        return tid, {"id": tid, "prompt": row["question_content"],
                     "tests": [dict(t) for t in tests],
                     "difficulty": row.get("difficulty", "?"),
                     "release_date": str(row.get("contest_date", ""))}
```

Add to `stratum`: `if source_name == "livecodebench": return str(row.get("difficulty", "?"))`.

- [ ] **Step 4: Align `parse_task`** — the existing `livecodebench` branch in `evaluator/suite/loader.py` already does `tests = tuple(rec.pop("tests"))`. Confirm it also pops nothing required-but-missing; it returns `Task(id, "livecodebench", problem, answer=None, tests=tuple(...), meta=rec)`. If the branch expects a `question` key, make it accept `prompt` (our record uses `prompt`): the branch should read `problem = rec.pop("prompt", None) or rec.pop("question")`.

- [ ] **Step 5: Add the prompt template** — in `evaluator/official/prompts.py` add `_LIVECODEBENCH` and map it:

```python
_LIVECODEBENCH = ("Solve the following programming problem. Respond with a single "
                  "Python code block containing the complete solution.\n\n{problem}")
# in _TEMPLATES:
    "livecodebench": _LIVECODEBENCH,
```

- [ ] **Step 6: Wire the scorer** — add `"livecodebench": code.score` to the canonical `SCORERS` map in `evaluator/sampler.py` (and mirror in `evaluator/validate.py`'s `SCORERS`). Grep `SCORERS = {` to find both.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py tests/eval -q`
Expected: PASS (baseline + new).

- [ ] **Step 8: Commit**

```bash
git add evaluator/hf_fetchers.py evaluator/suite/loader.py evaluator/official/prompts.py evaluator/sampler.py evaluator/validate.py tests/eval/test_fetchers_hard.py
git commit -m "feat(eval): LiveCodeBench fetcher/parse/prompt + code-scorer wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: GPQA-Diamond fetcher (seeded shuffle) + parse + prompt + MCQ wiring

**Files:**
- Modify: `evaluator/hf_fetchers.py`, `evaluator/suite/loader.py`, `evaluator/official/prompts.py`
- Test: add to `tests/eval/test_fetchers_hard.py`

**Interfaces:**
- Produces: `extract("gpqa_diamond", row)` → `(id, {"id","question","options":[4],"answer":<letter>,"subject"})` with options built as **[correct, incorrect1..3] shuffled by a seed derived from the question id** and `answer` = the letter the correct option landed on; `parse_task("gpqa_diamond", rec)` → MCQ `Task` (problem = question + lettered options, `answer` = gold letter). `SCORERS["gpqa_diamond"] = mcq.score`.

**Design note:** GPQA rows have `Question`, `Correct Answer`, `Incorrect Answer 1..3` (verify exact field names in Step 0 via a real row). The shuffle MUST be seeded off the stable question id (not global `random`) so the manifest's `content_sha` pins the exact letter↔option mapping and re-runs are identical.

- [ ] **Step 0: Inspect a real GPQA row** (network)

Run:
```bash
.venv/bin/python -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); print(sorted(d[0].keys())); print({k:str(d[0][k])[:80] for k in d[0]})"
```
Record the exact field names for question / correct / incorrect / subject and adjust the code below.

- [ ] **Step 1: Write failing test**

```python
import hashlib
from evaluator.hf_fetchers import extract
from evaluator.suite.loader import parse_task

def test_gpqa_seeded_shuffle_tracks_correct_letter():
    raw = {"Question": "Q?", "Correct Answer": "RIGHT",
           "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2",
           "Incorrect Answer 3": "w3", "Subdomain": "Physics",
           "Record ID": "gpqa_7"}
    tid, rec = extract("gpqa_diamond", raw)
    # the option at rec["answer"] letter must be the correct text
    letters = "ABCD"
    idx = letters.index(rec["answer"])
    assert rec["options"][idx] == "RIGHT"
    # deterministic: same row -> same letter
    _, rec2 = extract("gpqa_diamond", raw)
    assert rec2["answer"] == rec["answer"]
    t = parse_task("gpqa_diamond", rec)
    assert t.source == "gpqa_diamond" and t.answer == rec["answer"]
    assert "RIGHT" in t.problem
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py::test_gpqa_seeded_shuffle_tracks_correct_letter -q`
Expected: FAIL.

- [ ] **Step 3: Implement fetcher** — in `extract`, add:

```python
    if source_name == "gpqa_diamond":
        import random as _random
        tid = str(row.get("Record ID") or row.get("id") or hash(row["Question"]))
        correct = row["Correct Answer"].strip()
        options = [correct,
                   row["Incorrect Answer 1"].strip(),
                   row["Incorrect Answer 2"].strip(),
                   row["Incorrect Answer 3"].strip()]
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        order = list(range(4))
        _random.Random(seed).shuffle(order)
        shuffled = [options[i] for i in order]
        answer_letter = "ABCD"[shuffled.index(correct)]
        return tid, {"id": tid, "question": row["Question"],
                     "options": shuffled, "answer": answer_letter,
                     "subject": str(row.get("Subdomain", "?"))}
```

Add `import hashlib` at the top of `hf_fetchers.py` if absent. Add to `stratum`: `if source_name == "gpqa_diamond": return str(row.get("Subdomain", "?"))`.

- [ ] **Step 4: Implement parse_task** — in `evaluator/suite/loader.py`, add a branch modeled on `mmlu_pro` (letters A–D):

```python
    if source == "gpqa_diamond":
        question = rec.pop("question")
        options = rec.pop("options")
        answer = rec.pop("answer")
        lines = [question, ""]
        for letter, opt in zip(_letters(), options):
            lines.append(f"{letter}. {opt}")
        return Task(id=task_id, source=source, problem="\n".join(lines),
                    answer=answer, tests=(), meta=rec)
```

- [ ] **Step 5: Prompt + scorer wiring** — in `prompts.py` map `"gpqa_diamond"` to the SAME MMLU template (`_MMLU`), since the "The answer is (X)." sink matches the MCQ extractor. Add `"gpqa_diamond": mcq.score` to the `SCORERS` maps (sampler.py, validate.py).

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py tests/eval -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluator/hf_fetchers.py evaluator/suite/loader.py evaluator/official/prompts.py evaluator/sampler.py evaluator/validate.py tests/eval/test_fetchers_hard.py
git commit -m "feat(eval): GPQA-Diamond fetcher (seeded shuffle) + MCQ wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: AIME + MATH-L5 fetchers + parse + prompt + math wiring

**Files:**
- Modify: `evaluator/hf_fetchers.py`, `evaluator/suite/loader.py`, `evaluator/official/prompts.py`
- Test: add to `tests/eval/test_fetchers_hard.py`

**Interfaces:**
- Produces: `extract("aime", row)` → `(id, {"id","problem","answer","year"})`; `extract("math_l5", row)` → `(id, {"id","problem","answer","subject"})`; both `parse_task` → math-shaped `Task` (`answer` set, `tests=()`); `SCORERS["aime"]=SCORERS["math_l5"]=math.score`.

**Design note:** verify field names on real rows (Step 0). AIME answer is an integer string; MATH-L5 answer is the boxed content. Both reuse the math grader unchanged. MATH-L5 uses `hendrycks/competition_math` filtered to `level=="Level 5"` — the *filter* happens in the manifest builder (Task 5), but `extract` must still read the row's fields; `stratum` for math_l5 = subject/type, for aime = year.

- [ ] **Step 0: Inspect real rows** (network)

Run:
```bash
.venv/bin/python -c "from datasets import load_dataset; a=load_dataset('Maxwell-Jia/AIME_2024',split='train'); print('AIME', sorted(a[0].keys()), {k:str(a[0][k])[:60] for k in a[0]})"
.venv/bin/python -c "from datasets import load_dataset; m=load_dataset('hendrycks/competition_math',split='test',trust_remote_code=True); print('MATH', sorted(m[0].keys()), m[0].get('level'))"
```
Record actual field names (AIME problem/answer/id fields; MATH problem/solution/answer/level/type). If a named dataset is unavailable, pin an equivalent and note it.

- [ ] **Step 1: Write failing test**

```python
from evaluator.hf_fetchers import extract
from evaluator.suite.loader import parse_task

def test_aime_extract_parse():
    raw = {"ID": "2024-I-1", "Problem": "Find n.", "Answer": "204", "Year": "2024"}
    tid, rec = extract("aime", raw)
    t = parse_task("aime", rec)
    assert t.source == "aime" and t.answer == "204" and "Find n." in t.problem

def test_math_l5_extract_parse():
    raw = {"problem": "Compute.", "solution": "... \\boxed{3}", "level": "Level 5",
           "type": "Algebra"}
    tid, rec = extract("math_l5", raw)
    t = parse_task("math_l5", rec)
    assert t.source == "math_l5" and "Compute." in t.problem
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement fetchers** — in `extract`, add (adjust field names to Step 0):

```python
    if source_name == "aime":
        tid = str(row.get("ID") or row.get("id"))
        return tid, {"id": tid, "problem": row["Problem"],
                     "answer": str(row["Answer"]).strip(), "year": str(row.get("Year", "?"))}
    if source_name == "math_l5":
        import hashlib
        tid = str(row.get("unique_id") or hashlib.sha256(row["problem"].encode()).hexdigest()[:16])
        # gold answer = boxed content of the solution (reuse the math extractor)
        from evaluator.scorers.math import _find_last_boxed
        gold = _find_last_boxed(row.get("solution", "")) or row.get("answer", "")
        return tid, {"id": tid, "problem": row["problem"],
                     "answer": (gold or "").strip(), "subject": str(row.get("type", "?")),
                     "level": str(row.get("level", ""))}  # carried for the build-time L5 filter
```

`level` stays in `meta` after `parse_task` (which only pops `problem`/`answer`) — harmless.

Add to `stratum`: `if source_name == "aime": return str(row.get("Year","?"))` and `if source_name == "math_l5": return str(row.get("type","?"))`.

- [ ] **Step 4: parse_task** — both are math-shaped; add branches:

```python
    if source in ("aime", "math_l5"):
        problem = rec.pop("problem")
        answer = rec.pop("answer")
        return Task(id=task_id, source=source, problem=problem, answer=answer,
                    tests=(), meta=rec)
```

- [ ] **Step 5: Prompt + scorer wiring** — map `"aime"` and `"math_l5"` to the SAME math template (`_MATH`, the `\boxed{}` sink) in `prompts.py`. Add `"aime": math_scorer.score, "math_l5": math_scorer.score` to `SCORERS` (sampler.py, validate.py — note math is imported as `math_scorer`/`math`).

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/eval/test_fetchers_hard.py tests/eval -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluator/hf_fetchers.py evaluator/suite/loader.py evaluator/official/prompts.py evaluator/sampler.py evaluator/validate.py tests/eval/test_fetchers_hard.py
git commit -m "feat(eval): AIME + MATH-L5 fetchers + math-scorer wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Hard manifest builder + build the locked manifest

**Files:**
- Modify: `evaluator/build_manifest.py`
- Test: `tests/eval/test_hard_manifest.py`
- Create (Step 7, manual/network): `configs/suite.hard.manifest.json`

**Interfaces:**
- Produces: `HARD_SOURCES` (name, hf_dataset, split, size, optional `filter` predicate) and `build_hard(seed=DEFAULT_SEED) -> Manifest`, plus a `--hard` CLI flag in `main()`.

**Design note:** `build_hard` mirrors `build()` but (a) applies a per-source filter predicate to the raw id set before sampling (LCB: `release_date >= "2024-08-01"`; MATH: `level == "Level 5"`), and (b) writes `configs/suite.hard.manifest.json`. The filter runs on `id_map`/rows, so only surviving ids are sampled and hashed.

- [ ] **Step 1: Write failing test** — `tests/eval/test_hard_manifest.py`

```python
from evaluator.build_manifest import HARD_SOURCES, stratified_sample

def test_hard_sources_declared():
    names = {s[0] for s in HARD_SOURCES}
    assert names == {"livecodebench", "gpqa_diamond", "aime", "math_l5"}

def test_stratified_sample_still_pure():
    ids = [f"t{i}" for i in range(20)]
    strata = {i: ("a" if int(i[1:]) % 2 else "b") for i in ids}
    picked = stratified_sample(ids, strata, 10, seed=1)
    assert len(picked) <= 10 and picked == sorted(picked)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_hard_manifest.py -q`
Expected: FAIL — `HARD_SOURCES` undefined.

- [ ] **Step 3: Implement** — in `evaluator/build_manifest.py`:

```python
# (name, hf_dataset, split, size, filter_predicate|None)
# filter_predicate: (PARSED record from extract) -> bool, applied before sampling.
# load_id_map runs extract, so predicates read the record fields extract carries
# (livecodebench -> "release_date"; math_l5 -> "level").
HARD_SOURCES = [
    ("livecodebench", "livecodebench/code_generation_lite", "test", 150,
     lambda rec: str(rec.get("release_date", "")) >= "2024-08-01"),
    ("gpqa_diamond", "Idavidrein/gpqa", "train", 198, None),  # config gpqa_diamond
    ("aime", "Maxwell-Jia/AIME_2024", "train", 60, None),     # + AIME 2025, see build_hard
    ("math_l5", "hendrycks/competition_math", "test", 250,
     lambda rec: rec.get("level") == "Level 5"),
]


def build_hard(seed: int = DEFAULT_SEED) -> Manifest:
    from huggingface_hub import dataset_info
    sources = []
    for name, hf, split, n, pred in HARD_SOURCES:
        revision = dataset_info(hf).sha
        id_map, strata = load_id_map(name, hf, revision, split)  # runs extract
        ids = [tid for tid in id_map if pred is None or pred(id_map[tid])]
        sampled = stratified_sample(ids, strata, min(n, len(ids)), seed)
        records = [id_map[tid] for tid in sampled]
        sources.append(SourceSpec(name, hf, revision, split, tuple(sampled),
                                  content_sha(records)))
    return Manifest(version=1, sources=tuple(sources))
```

The predicate reads the parsed record (Task 4 makes `math_l5`'s `extract` carry
`"level"`, Task 2 makes `livecodebench`'s carry `"release_date"`), so no separate
`_passes` helper is needed. **`hf_fetchers.load_id_map`'s `load_dataset` call
likely needs per-source args for the hard datasets:** GPQA needs the config name
(`load_dataset("Idavidrein/gpqa", "gpqa_diamond", ...)`); LiveCodeBench and
`hendrycks/competition_math` need `trust_remote_code=True`. Extend `load_id_map`
with an optional per-source `config`/`trust_remote_code` (default off, so the
standard tier is unaffected) and set them for these three sources.

Extend `main()` with `--hard`: `if "--hard" in sys.argv: save(build_hard(), "configs/suite.hard.manifest.json")`.

- [ ] **Step 4: Run tests to green**

Run: `.venv/bin/pytest tests/eval/test_hard_manifest.py -q`
Expected: PASS.

- [ ] **Step 5: Commit code**

```bash
git add evaluator/build_manifest.py evaluator/hf_fetchers.py evaluator/suite/loader.py tests/eval/test_hard_manifest.py
git commit -m "feat(eval): hard-tier manifest builder (build_hard + filters)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6 (manual, network): AIME 2025** — verify/pin a 2025 AIME dataset (e.g. `opencompass/AIME2025` or equivalent); add it as a fifth entry or merge into the `aime` source by concatenating both years' ids in `build_hard` (record both dataset revisions if merged — if merging is awkward, keep `aime_2025` as its own source, scored the same). Adjust `HARD_SOURCES`/`build_hard` accordingly and re-run the test.

- [ ] **Step 7 (manual, network): build the locked manifest**

Run: `.venv/bin/python -m evaluator.build_manifest --hard`
Then sanity-load: `.venv/bin/python -c "from evaluator.suite.manifest import load; m=load('configs/suite.hard.manifest.json'); print([(s.name,len(s.task_ids)) for s in m.sources])"`
Commit `configs/suite.hard.manifest.json`.

---

### Task 6: Reference self-test extension to hard sources

**Files:**
- Modify: `evaluator/audit/references.py`, `evaluator/audit/reference_selftest.py`
- Test: `tests/eval/test_reference_selftest_hard.py`

**Interfaces:**
- Consumes: `Reference`, `selftest_one`. Produces: `_SCORERS` + `_synth_output` handling `gpqa_diamond` (synth `The answer is (GOLD).`), `aime`/`math_l5` (synth `\boxed{GOLD}`), `livecodebench` (run the gold/canonical solution if the dataset provides one, else skip with a documented note — LCB lite may not ship reference solutions).

**Design note:** GPQA/AIME/MATH-L5 self-tests mirror the M2c mmlu/math patterns exactly. LCB has no gold *completion* in `code_generation_lite`; the reference self-test for LCB instead asserts that the **normalized tests are non-empty and executable** (a smoke that the test harness runs), not that a gold solution passes.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_reference_selftest_hard.py`

```python
from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.audit.reference_selftest import selftest_one

def _t(source, answer=None):
    return Task(id="q", source=source, problem="P", answer=answer, tests=(), meta={})

def test_gpqa_gold_recognized():
    assert selftest_one(_t("gpqa_diamond", "C"), Reference("q","gpqa_diamond", gold="C")) is None

def test_aime_gold_recognized():
    assert selftest_one(_t("aime", "204"), Reference("q","aime", gold="204", solution="\\boxed{204}")) is None

def test_math_l5_gold_recognized():
    assert selftest_one(_t("math_l5", "3"), Reference("q","math_l5", gold="3", solution="so \\boxed{3}")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_reference_selftest_hard.py -q`
Expected: FAIL (KeyError — sources not in `_SCORERS`).

- [ ] **Step 3: Implement** — in `evaluator/audit/reference_selftest.py`, extend `_SCORERS` and `_synth_output`:

```python
from evaluator.scorers import mcq, code
from evaluator.scorers import math as math_scorer
_SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score,
            "gpqa_diamond": mcq.score, "aime": math_scorer.score,
            "math_l5": math_scorer.score, "livecodebench": code.score}
```

In `_synth_output`, treat `gpqa_diamond` like `mmlu_pro`, and `aime`/`math_l5` like `math`:

```python
    if task.source in ("mmlu_pro", "gpqa_diamond"):
        return f"Reasoning about the options. The answer is ({ref.gold})."
    if task.source in ("math", "aime", "math_l5"):
        if ref.solution:
            return ref.solution
        return f"Therefore the answer is \\boxed{{{ref.gold}}}."
    if task.source in ("humaneval", "livecodebench"):
        body = (ref.prompt or "") + (ref.canonical_solution or "")
        return f"```python\n{body}\n```"
    return ref.gold or ""
```

In `evaluator/audit/references.py::build_reference_index`, add branches so gpqa/aime/math_l5 records map to `Reference(gold=..., solution=...)` like mmlu/math (mirror the existing per-source mapping; use the record's `answer`/`solution` fields).

- [ ] **Step 4: Run tests to green**

Run: `.venv/bin/pytest tests/eval/test_reference_selftest_hard.py tests/eval -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluator/audit/references.py evaluator/audit/reference_selftest.py tests/eval/test_reference_selftest_hard.py
git commit -m "feat(audit): reference self-test covers the four hard sources

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Hard-tier report — Wilson CIs + pairwise significance + contamination

**Files:**
- Create: `scripts/hard_report.py`, `tests/eval/test_hard_report.py`

**Interfaces:**
- Produces: `wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]` (lower, upper); `mcnemar_p(b: int, c: int) -> float` (discordant counts → two-sided p, normal approx); `main()` scoring a hard run dir set and printing per-model accuracy+CI, pairwise significance for the top pair, and the fresh-vs-public table.
- Consumes: the M2c scoring pattern (read frozen, official scorers), `evaluator.suite` loaders.

**Design note:** the report's job is to answer "does the hard tier de-tie the frontier?" — so CIs and a pairwise test are the point, not decoration. Wilson CI and a normal-approx McNemar are enough (no scipy). `main()` mirrors `scripts/final_numbers.py` but reads `configs/suite.hard.manifest.json` and the hard run dirs, and computes the fresh {aime, livecodebench} vs public {math_l5, gpqa_diamond} per-model delta.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_hard_report.py`

```python
from scripts.hard_report import wilson_ci, mcnemar_p

def test_wilson_ci_basic():
    lo, hi = wilson_ci(90, 100)
    assert 0.0 < lo < 0.90 < hi < 1.0
    assert lo < hi

def test_wilson_ci_degenerate():
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0

def test_mcnemar_symmetric_is_insignificant():
    assert mcnemar_p(10, 10) > 0.5   # equal discordant -> not significant

def test_mcnemar_lopsided_is_significant():
    assert mcnemar_p(20, 2) < 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/eval/test_hard_report.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/hard_report.py`**

```python
"""Hard-tier report: per-model accuracy with Wilson CIs, a pairwise McNemar
significance test for the top pair, and a fresh-vs-public contamination table.
Offline (no model calls). Reads configs/suite.hard.manifest.json + hard run dirs.
"""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar via normal approx on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)      # continuity-corrected
    # two-sided normal tail
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    import sys
    from pathlib import Path
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher
    from evaluator.store import read_frozen
    from evaluator.sampler import SCORERS

    run_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluator/runs/m2d_hard")
    manifest = load("configs/suite.hard.manifest.json")
    tasks = load_suite(manifest, {s.name: make_fetcher(s.name) for s in manifest.sources})
    tb = {t.id: t for t in tasks}
    models = sorted(p.name for p in run_root.glob("*") if p.is_dir())

    # score each model over its ok frozen outputs
    scored = {}
    for mdl in models:
        d = {}
        for fo in read_frozen(run_root / mdl):
            if fo.status == "ok" and fo.task_id in tb:
                t = tb[fo.task_id]
                d[fo.task_id] = SCORERS[t.source](t, fo.output_text).correct
        scored[mdl] = d
    common = set(tb) & set.intersection(*[set(scored[m]) for m in models]) if models else set()
    src = {t.id: t.source for t in tasks}

    print(f"common hard tasks: {len(common)}")
    accs = {}
    for mdl in models:
        k = sum(scored[mdl][t] for t in common)
        lo, hi = wilson_ci(k, len(common))
        accs[mdl] = k / len(common) if common else float("nan")
        print(f"  {mdl:18} {accs[mdl]:.4f}  95%CI[{lo:.3f},{hi:.3f}]")

    # pairwise significance for the top two
    top = sorted(accs, key=lambda m: -accs[m])[:2]
    if len(top) == 2:
        a, b = top
        bb = sum(1 for t in common if scored[a][t] and not scored[b][t])
        cc = sum(1 for t in common if scored[b][t] and not scored[a][t])
        print(f"top pair {a} vs {b}: discordant {bb}/{cc}, McNemar p={mcnemar_p(bb,cc):.3f}")

    # fresh vs public contamination
    fresh = {"aime", "livecodebench"}; public = {"math_l5", "gpqa_diamond"}
    print("\nfresh-vs-public (public-minus-fresh accuracy delta; large + = suspect):")
    for mdl in models:
        def acc_of(group):
            ids = [t for t in common if src[t] in group]
            return sum(scored[mdl][t] for t in ids) / len(ids) if ids else float("nan")
        f, p = acc_of(fresh), acc_of(public)
        print(f"  {mdl:18} fresh={f:.3f} public={p:.3f} delta={p-f:+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to green**

Run: `.venv/bin/pytest tests/eval/test_hard_report.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/hard_report.py tests/eval/test_hard_report.py
git commit -m "feat(eval): hard-tier report (Wilson CI + McNemar + contamination)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8 (manual, PAID): sample the hard tier + publish the report

This step makes real, paid model calls — run it manually after Tasks 1–7 are merged and the reference self-test passes on the built hard manifest. Do NOT auto-run it in SDD.

- [ ] **Step 1: Free reference self-test on the hard manifest** — extend the M2c self-test runner to `configs/suite.hard.manifest.json`; require ~100% gold recognition (or documented exceptions) before spending. Fix any grader/fetcher gap first.
- [ ] **Step 2: Paid smoke** (~$1–2): a few tasks/source × the 6 models via the sharded wrapper; eyeball parseability (especially LCB execution and GPQA letter extraction) exactly as M2c's smoke did.
- [ ] **Step 3: Full hard-tier sample** — reuse the M2c per-model-parallel sharded driver into `evaluator/runs/m2d_hard/<model>/`, 6 models, budget-gated ~$25, resumable, prune-and-retry transient errors.
- [ ] **Step 4: Generate + write the report** — run `.venv/bin/python -m scripts.hard_report evaluator/runs/m2d_hard`, paste numbers into `docs/HARD_TIER_REPORT.md` (per-model + CIs, pairwise significance, fresh-vs-public), and state whether the frontier de-ties and which models are memorization suspects.

---

## Final gate (after Tasks 1–7)

- [ ] `.venv/bin/pytest tests/eval tests/router -q` — all green.
- [ ] Isolation: `grep -rn "import gateway" evaluator/ scripts/` returns nothing.
- [ ] Standard tier unchanged: `git diff --stat origin/master -- configs/suite.manifest.json` is empty (only the *hard* manifest is new).
- [ ] Every new `SCORERS` map entry present in both `evaluator/sampler.py` and `evaluator/validate.py`.

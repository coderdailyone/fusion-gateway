"""k-sample sampler for self-consistency voting (M6).

One (task, model) pair -> k independent samples of the SAME official prompt
(`evaluator.runner.run_one` builds it, so the answer never leaks). Every sample
is FROZEN: the whole milestone rests on being able to re-vote, re-tally and
re-score the same ballots later at $0, so a sample that is not written to disk
is money burned.

Layout on disk -- one run dir PER SAMPLE INDEX under a single root:

    <root>/k0/frozen.jsonl        sample #1 of every (task, model)
    <root>/k1/frozen.jsonl        sample #2 of every (task, model)
    ...

Why not one flat frozen.jsonl? Because the store's resume key -- and
`run_budgeted`'s -- is (task_id, model), which admits exactly one row. Splitting
by sample index keeps that proven skip-logic intact, gives k independent
resumable checkpoints, and makes the k ballots for a pair trivial to reassemble
(`frozen_by_pair`). It also fills breadth-first: an interrupted run leaves every
task with the SAME number of ballots rather than a few tasks with k and the rest
with none, so a partial run is still a usable (smaller-k) experiment.

Resumability rule: a (task, model) pair that already has k frozen samples is
skipped and costs nothing. Error rows count toward k (they are frozen like any
other sample, and vote.py treats an unparseable text as a spoiled ballot); rerun
with a fresh `tag` if a batch of transport errors needs re-sampling.

Sharding: pass a `tag` to write into `<root>/k<i>_<tag>/`, so several processes
can sample DISJOINT slices (by model or by task) in parallel without appending
to the same file. `frozen_by_pair` globs every `k*` dir, so the reader does not
care how the work was split. Slices must be disjoint: two shards given the same
(task, model) pair will each pay for it.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from evaluator.runner import FrozenOutput, run_one
from evaluator.store import append_frozen, read_frozen

_DIR_PREFIX = "k"


def accepts_kwarg(fn: Callable, name: str) -> bool:
    """True when `fn` can be called with keyword `name` (or takes **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):      # builtins / C callables
        return False
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return True
    p = params.get(name)
    return p is not None and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)


def sample_task(
    task,
    model: str,
    k: int,
    completion_fn: Callable,
    temperature: float | None = None,
    *,
    on_frozen: Callable[[FrozenOutput], None] | None = None,
) -> list[str]:
    """Draw k independent samples of `task` from `model`; return their texts.

    `completion_fn(model, prompt) -> {text,in_tokens,out_tokens,cost_usd}` --
    the same shape run_one/sample expect.

    `temperature` is forwarded as a keyword to `completion_fn` ONLY when that
    function accepts one. The usual wiring binds it earlier instead
    (`validate.make_model_fn("glm-5.2", temperature=0.8)` forwards it straight
    to litellm.completion), so callers normally leave this None. Passing a
    temperature to a fn that cannot take one raises rather than silently
    dropping it: k samples at an unintended temperature look exactly like k
    samples at the intended one -- identical ballots, a collapsed tally, and a
    paid run that has to be redone.

    Each sample is turned into a FrozenOutput by run_one (so an API failure
    becomes an error row, never an exception) and handed to `on_frozen` before
    the next call, so an interrupted run keeps everything already paid for.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    fn = completion_fn
    if temperature is not None:
        if not accepts_kwarg(completion_fn, "temperature"):
            raise ValueError(
                f"sample_task: completion_fn for {model!r} does not accept a "
                "`temperature` keyword -- bind it when building the fn "
                "(validate.make_model_fn(name, temperature=...)) or pass "
                "temperature=None. Refusing to silently sample at the "
                "provider default.")

        def fn(m: str, prompt: str, _f=completion_fn, _t=temperature) -> dict:
            return _f(m, prompt, temperature=_t)

    texts: list[str] = []
    for _ in range(k):
        fo = run_one(task, model, fn)
        if on_frozen is not None:
            on_frozen(fo)
        texts.append(fo.output_text)
    return texts


def sample_dir(root, index: int, tag: str = "") -> Path:
    """Run dir holding sample #index (0-based) -- see the module docstring."""
    name = f"{_DIR_PREFIX}{index}" + (f"_{tag}" if tag else "")
    return Path(root) / name


def sample_dirs(root) -> list[Path]:
    """Every per-sample-index dir under `root`, sorted by (index, tag)."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(_DIR_PREFIX):
            continue
        head, _, tag = d.name.partition("_")
        try:
            index = int(head[len(_DIR_PREFIX):])
        except ValueError:
            continue
        out.append((index, tag, d))
    return [d for _i, _t, d in sorted(out)]


def frozen_by_pair(root) -> dict[tuple[str, str], list[FrozenOutput]]:
    """(task_id, model) -> its frozen samples, in sample-index order."""
    pairs: dict[tuple[str, str], list[FrozenOutput]] = {}
    for d in sample_dirs(root):
        for fo in read_frozen(d):
            pairs.setdefault((fo.task_id, fo.model), []).append(fo)
    return pairs


def ballot_counts(root) -> dict[tuple[str, str], int]:
    """(task_id, model) -> how many samples are already frozen."""
    return {pair: len(rows) for pair, rows in frozen_by_pair(root).items()}


def sample_many(
    models: dict[str, Callable],
    tasks: list,
    root,
    k: int,
    ceiling: float,
    cost_fn: Callable[[str, int, int], float],
    *,
    tag: str = "",
    log: Callable[[str], None] = print,
    run_budgeted: Callable | None = None,
) -> dict:
    """Resumable, budget-gated k-sampling of every (task, model) pair.

    Runs k breadth-first passes; each (pass, model) slice is delegated to
    `scripts.resample_official.run_budgeted`, which owns the preflight budget
    gate, the hard-ceiling backstop and the within-dir (task_id, model) skip.
    `ceiling` is the TOTAL for this root across all passes: each delegated call
    gets `ceiling` minus what the other sample dirs already cost, so the k
    passes cannot each spend the full budget.

    Pairs that already hold >= k samples anywhere under `root` are never
    re-called. The pending snapshot is taken once, up front: a stale snapshot
    can only skip work (fixed by rerunning), never double-pay, because
    run_budgeted re-checks the target dir before every call.

    Returns {"spent", "completed", "stopped"} -- `spent` covers everything
    frozen under `root`, not just this call's new rows.
    """
    if run_budgeted is None:
        from scripts.resample_official import run_budgeted  # noqa: PLC0415

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    counts = ballot_counts(root)
    spent_by_dir: dict[Path, float] = {
        d: sum(cost_fn(fo.model, fo.in_tokens, fo.out_tokens) for fo in read_frozen(d))
        for d in sample_dirs(root)
    }

    completed = 0
    stopped = False
    for index in range(k):
        d = sample_dir(root, index, tag)
        for name, fn in models.items():
            todo = [t for t in tasks if counts.get((t.id, name), 0) <= index]
            if not todo:
                continue
            d.mkdir(parents=True, exist_ok=True)
            others = sum(v for path, v in spent_by_dir.items() if path != d)
            res = run_budgeted({name: fn}, todo, d, ceiling - others, cost_fn)
            spent_by_dir[d] = res["spent"]
            completed += res["completed"]
            log(f"  [sample {index + 1}/{k}] {name:14} new={res['completed']:4d} "
                f"todo={len(todo):4d} spent=${sum(spent_by_dir.values()):.4f}")
            if res["stopped"]:
                stopped = True
                break
        if stopped:
            break

    return {"spent": sum(spent_by_dir.values()), "completed": completed,
            "stopped": stopped}


def freezer(run_dir):
    """`on_frozen` callback that appends into `run_dir`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def append(fo: FrozenOutput) -> None:
        append_frozen(run_dir, fo)

    return append

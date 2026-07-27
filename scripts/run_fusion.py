"""M5 driver: oracle gate, then cross-review + fusion over the domestic panel.

Subcommands:
    oracle              free go/no-go gate ($0) — scores the frozen domestic
                        candidates and reports the panel's oracle ceiling
    run [--limit N]     cross-review + fusion (PAID, budget-gated, resumable)

Discipline (from the M5 spec + the Phase-A whole-branch review):
  * Panel and fuser are DOMESTIC only. The model list here is an explicit dict,
    never a glob over evaluator/runs/* — a glob would sweep in the frontier
    models and make the headline claim circular.
  * `completion_fn` MUST dispatch on its model-name argument. This repo's
    `make_completion_fn` ignores that argument and always calls the model it
    was bound to, so we build one fn per model and route by name here.
  * Scoring uses the official scorers only; cross-review never touches scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# panel members (domestic) and where their frozen candidates live. kimi was
# sampled in parallel shards, so a model may map to several run dirs; they are
# merged first-ok-wins, matching load_frozen_by_model's semantics.
PANEL_DIRS = {
    "deepseek-chat": ["evaluator/runs/m2c_full/deepseek-chat"],
    "glm-5.2": ["evaluator/runs/m2c_full/glm-5.2"],
    "kimi-k3": [f"evaluator/runs/m5_fusion/kimi-k3-s{s}" for s in range(8)],
}
PANEL = list(PANEL_DIRS)


def _load_panel_frozen() -> dict[str, dict[str, str]]:
    """model -> {task_id: text}, merging each model's (possibly sharded) dirs."""
    from evaluator.fusion.panel import load_frozen_by_model

    merged: dict[str, dict[str, str]] = {}
    for model, dirs in PANEL_DIRS.items():
        texts: dict[str, str] = {}
        for d in dirs:
            if not Path(d).exists():
                continue
            for tid, text in load_frozen_by_model({model: d})[model].items():
                texts.setdefault(tid, text)
        merged[model] = texts
    return merged
FUSER = "glm-5.2"
FRONTIER_BAR = 0.894  # gpt-5.6-sol, M2c official scoring
RUN_ROOT = Path("evaluator/runs/m5_fusion")


def _load_tasks(manifest_path: str = "configs/suite.manifest.json"):
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher

    m = load(manifest_path)
    fetchers = {s.name: make_fetcher(s.name) for s in m.sources}
    return sorted(load_suite(m, fetchers), key=lambda t: (t.source, t.id))


def _score(task, text: str) -> bool:
    from evaluator.validate import SCORERS

    try:
        return bool(SCORERS[task.source](task, text).correct)
    except Exception:
        return False


def cmd_oracle() -> int:
    """Free gate: can the domestic panel even reach the frontier bar?"""
    from evaluator.fusion.panel import assemble
    from scripts.fusion_report import oracle

    tasks = _load_tasks()
    by_id = {t.id: t for t in tasks}
    frozen = _load_panel_frozen()
    for model, texts in frozen.items():
        print(f"  {model:16} frozen ok rows: {len(texts)}")

    cases, excluded = assemble(frozen, [t.id for t in tasks], min_candidates=2)
    # the oracle denominator must equal the set fusion is evaluated on
    evaluated = [c.task_id for c in cases]
    print(f"  panel cases: {len(cases)}  excluded (<2 candidates): {len(excluded)}")

    correct_by_model: dict[str, dict[str, bool]] = {m: {} for m in PANEL}
    for case in cases:
        task = by_id[case.task_id]
        for model, text in case.candidates.items():
            correct_by_model[model][case.task_id] = _score(task, text)

    print("\n=== per-model accuracy (on the panel set) ===")
    for model, per_task in correct_by_model.items():
        n = len(per_task)
        k = sum(per_task.values())
        print(f"  {model:16} {k}/{n} = {k / n:.4f}" if n else f"  {model:16} no rows")

    orc = oracle(correct_by_model)
    print(f"\n=== ORACLE (at least one domestic model correct) = {orc:.4f} "
          f"over {len(evaluated)} tasks ===")
    print(f"  frontier bar (gpt-5.6-sol) = {FRONTIER_BAR}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "oracle.json").write_text(json.dumps(
        {"oracle": orc, "n": len(evaluated), "bar": FRONTIER_BAR,
         "per_model": {m: {"n": len(d), "correct": sum(d.values())}
                       for m, d in correct_by_model.items()},
         "excluded": len(excluded)}, indent=2))
    if orc < FRONTIER_BAR:
        print("  VERDICT: NO-GO — the pool's ceiling is below the bar. "
              "Changing the pool, not the prompts, is the fix.")
        return 1
    print("  VERDICT: GO — headroom exists; paid review+fusion is justified.")
    return 0


def _dispatch():
    """Build a name-routing completion fn (see module docstring, finding C1)."""
    from evaluator import validate

    validate.load_secrets()
    fns = {m: validate.MODELS[m]() for m in set(PANEL) | {FUSER}}

    def dispatch(model: str, prompt: str) -> dict:
        return fns[model](model, prompt)

    return dispatch


def cmd_run(limit: int | None, ceiling: float, out_name: str) -> int:
    """PAID: cross-review + fusion, resumable, budget-gated."""
    from evaluator.fusion.panel import assemble
    from evaluator.fusion.review import cross_review
    from evaluator.fusion.fuse import fuse

    tasks = _load_tasks()
    by_id = {t.id: t for t in tasks}
    frozen = _load_panel_frozen()
    cases, _ = assemble(frozen, [t.id for t in tasks], min_candidates=2)
    if limit:
        # stratify the smoke across sources
        per_source: dict[str, list] = {}
        for c in cases:
            per_source.setdefault(by_id[c.task_id].source, []).append(c)
        picked, i = [], 0
        while len(picked) < limit and any(v[i:] for v in per_source.values()):
            for src in sorted(per_source):
                if len(picked) >= limit:
                    break
                if i < len(per_source[src]):
                    picked.append(per_source[src][i])
            i += 1
        cases = picked

    out_dir = RUN_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fused.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(json.loads(line)["task_id"])
    todo = [c for c in cases if c.task_id not in done]
    print(f"  cases={len(cases)} done={len(done)} todo={len(todo)} ceiling=${ceiling}")

    dispatch = _dispatch()
    spent = 0.0
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                row = json.loads(line)
                spent += row.get("review_cost", 0.0) + row.get("fusion_cost", 0.0)
    for n, case in enumerate(todo, 1):
        if spent >= ceiling:
            print(f"  STOP: spent ${spent:.4f} reached ceiling ${ceiling}")
            break
        task = by_id[case.task_id]
        reviews, review_cost = cross_review(task, case, dispatch)
        res = fuse(task, case, reviews, dispatch, fuser=FUSER)
        spent += review_cost + res.cost_usd
        with out_path.open("a") as f:
            f.write(json.dumps({
                "task_id": case.task_id, "source": task.source,
                "answer": res.answer, "status": res.status, "error": res.error,
                "review_cost": review_cost, "fusion_cost": res.cost_usd,
                "reviews": {r: {t: [v.verdict, v.reason] for t, v in d.items()}
                            for r, d in reviews.items()},
            }) + "\n")
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] spent=${spent:.4f}", flush=True)
    print(f"  DONE spent=${spent:.4f} -> {out_path}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    if cmd == "oracle":
        return cmd_oracle()
    if cmd == "run":
        limit = None
        ceiling = 25.0
        out = "fused_full"
        args = sys.argv[2:]
        for i, a in enumerate(args):
            if a == "--limit":
                limit = int(args[i + 1])
                out = "fused_smoke"
                ceiling = 2.0
            elif a == "--ceiling":
                ceiling = float(args[i + 1])
            elif a == "--out":
                out = args[i + 1]
        return cmd_run(limit, ceiling, out)
    print(f"unknown command {cmd!r}; use 'oracle' or 'run'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

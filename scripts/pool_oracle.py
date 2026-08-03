#!/usr/bin/env python3
"""Does a different-lineage model decorrelate the pool? ($0 — frozen samples only)

M5 and M6 both measured a ~+0.7pt ceiling over the best pool member while the
panel oracle sat 4-6pt higher, and the M6 report named the cause: the pool's
models fail *together*, so no aggregation rule can recover an answer nobody
produced. It also named the only lever that moves that ceiling — add a model
with a different pretraining lineage, not another Chinese chat model tuned on
similar data.

This script pulls that lever offline. Every model below was already sampled and
frozen during M2c, so computing a new panel's oracle costs nothing: it re-scores
existing outputs with the same official scorers and the same `oracle()` the
published M5 numbers came from, and reports how the ceiling moves as the pool
composition changes.

Two comparison families, because kimi-k3 ran out of quota at 492 of 1063 tasks:

  full   — the 1063-task suite, over the models that completed it
  kimi   — the 492-task subset where kimi-k3 exists, so the M5 panel is included

Usage:  .venv/bin/python scripts/pool_oracle.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

RUNS = Path("evaluator/runs/m2c_full")

DOMESTIC = ["deepseek-chat", "glm-5.2", "kimi-k3"]
FOREIGN = ["claude-sonnet-5", "claude-opus-4-8", "gpt-5.6-sol"]


def load_frozen(model: str) -> dict[str, str]:
    """task_id -> output_text, for status=="ok" rows only (first wins)."""
    path = RUNS / model / "frozen.jsonl"
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok" and rec["task_id"] not in out:
            out[rec["task_id"]] = rec.get("output_text") or ""
    return out


def main() -> int:
    from scripts.run_fusion import _load_tasks, _score
    from scripts.fusion_report import oracle

    tasks = _load_tasks()
    by_id = {t.id: t for t in tasks}
    models = DOMESTIC + FOREIGN

    frozen = {m: load_frozen(m) for m in models}
    print("frozen ok rows per model:")
    for m in models:
        print(f"  {m:18} {len(frozen[m])}")

    # Score once per (model, task); every panel below reuses these.
    print("\nscoring (official scorers, same as the published numbers)...")
    correct: dict[str, dict[str, bool]] = {}
    for m in models:
        correct[m] = {tid: _score(by_id[tid], text)
                      for tid, text in frozen[m].items() if tid in by_id}
        n, k = len(correct[m]), sum(correct[m].values())
        print(f"  {m:18} {k}/{n} = {k / n:.4f}" if n else f"  {m:18} none")

    families = {
        "full (1063-task suite, models that completed it)":
            (set(by_id) & set.intersection(
                *[set(correct[m]) for m in models if m != "kimi-k3"]),
             [m for m in models if m != "kimi-k3"]),
        "kimi subset (492 tasks where kimi-k3 exists)":
            (set(correct["kimi-k3"]), models),
    }

    for label, (task_ids, avail) in families.items():
        print(f"\n{'=' * 72}\n{label} — n={len(task_ids)}\n{'=' * 72}")
        dom = [m for m in DOMESTIC if m in avail]

        def orc(panel: list[str]) -> float:
            return oracle({m: {t: correct[m][t] for t in task_ids if t in correct[m]}
                           for m in panel})

        # CONTROL: is this family's task set simply easier? Without this,
        # a subset that happens to be easy inflates every oracle on it and
        # makes any added model look redundant.
        print("\n  per-model accuracy ON THIS TASK SET (the control):")
        for m in avail:
            ids = [t for t in task_ids if t in correct[m]]
            k = sum(correct[m][t] for t in ids)
            print(f"    {m:18} {k}/{len(ids)} = {k / len(ids):.4f}")

        base = orc(dom)
        print(f"\n  domestic pool {dom}")
        print(f"    oracle = {base:.4f}   <- the ceiling M5/M6 could not beat")
        best_single = max(dom, key=lambda m: sum(
            correct[m][t] for t in task_ids if t in correct[m]))
        bs = sum(correct[best_single][t] for t in task_ids if t in correct[best_single])
        bn = len([t for t in task_ids if t in correct[best_single]])
        print(f"    best single member: {best_single} = {bs / bn:.4f}")

        print("\n  adding ONE different-lineage model:")
        for f in FOREIGN:
            if f not in avail:
                continue
            lifted = orc(dom + [f])
            print(f"    + {f:18} oracle = {lifted:.4f}   "
                  f"({lifted - base:+.4f} vs domestic-only)")

        print("\n  for contrast — adding a same-lineage model instead:")
        for m in avail:
            if m in dom or m in FOREIGN:
                continue
            lifted = orc(dom + [m])
            print(f"    + {m:18} oracle = {lifted:.4f}   ({lifted - base:+.4f})")

        # How often does the whole domestic pool fail together? That number is
        # the headroom any pool change has to attack.
        all_wrong = [t for t in task_ids
                     if not any(correct[m].get(t, False) for m in dom)]
        print(f"\n  domestic pool ALL wrong on {len(all_wrong)} tasks "
              f"({len(all_wrong) / len(task_ids):.1%})")
        for f in FOREIGN:
            if f not in avail:
                continue
            rescued = sum(1 for t in all_wrong if correct[f].get(t, False))
            print(f"    {f:18} is right on {rescued:4d} of them "
                  f"({rescued / len(all_wrong):.1%} of the pool's blind spot)"
                  if all_wrong else "")

        # Best 3-model pool overall, ignoring provenance.
        print("\n  best 3-model pool by oracle, any provenance:")
        combos = sorted(((orc(list(c)), c) for c in itertools.combinations(avail, 3)),
                        reverse=True)[:3]
        for score, c in combos:
            print(f"    {score:.4f}  {list(c)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recompute true per-model + router numbers from an official-pipeline run.

Offline: scores frozen outputs with the (now official) scorers, builds the
result matrix, and reruns the exact M3b pipeline (learnability gate, OOF
classifiers, lambda-swept policy, code verify-cascade, Pareto envelope) on the
corrected labels. Prints the numbers for the benchmark report.
"""
from __future__ import annotations


def sota_verdict(points: dict[str, tuple[float, float]]) -> dict:
    best = max(points.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
    return {"best_single": best[0], "best_acc": best[1][0]}


def per_model_table(rows) -> dict:
    from evaluator.report import aggregate
    return aggregate(rows)


def main() -> None:
    import sys
    from evaluator import validate
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher
    from evaluator.sampler import sample
    from router.matrix import ResultMatrix
    from router.learnability import per_model_cv_auc, gate
    from router.train import fit_oof
    from router.policy import sweep_lambda
    from router import cascade
    from router.pareto import static_points, envelopes, render_report

    validate.load_secrets()
    run_dir = sys.argv[1]
    manifest = load("configs/suite.manifest.json")
    tasks = load_suite(manifest, {s.name: make_fetcher(s.name) for s in manifest.sources})
    models = {}  # scoring only over frozen outputs; no models -> no API calls
    # score frozen outputs offline (sample() skips already-frozen pairs; with an
    # empty models dict it does no sampling and only scores what's frozen)
    rows = sample(models, tasks, run_dir)

    print("=== per-model (official scoring) ===")
    for m, a in sorted(per_model_table(rows).items()):
        print(f"  {m:18} acc={a['accuracy']:.4f} mean_cost=${a['mean_cost_usd']:.6f} n={a['n']}")

    matrix = ResultMatrix.from_rows(rows)
    task_ids = [t.id for t in tasks]
    non_code = [t.id for t in tasks if t.source != "humaneval"]
    code_ids = [t.id for t in tasks if t.source == "humaneval"]

    aucs = per_model_cv_auc([t for t in tasks if t.source != "humaneval"], matrix)
    print("learnability:", gate(aucs), aucs)

    oof = fit_oof([t for t in tasks if t.source != "humaneval"], matrix)
    dyn = sweep_lambda(oof, matrix, non_code, [0.0, 1.0, 3.0, 10.0, 1e6])
    sp = static_points(matrix, task_ids)
    print("SOTA verdict:", sota_verdict(sp))
    print(render_report(dyn, sp, envelopes(dyn, sp)))
    # code cascade over cheapest->dearest order (by mean cost)
    order = sorted(matrix.models, key=lambda m: sum(matrix.cost[m].values()))
    print("code cascade:", cascade.evaluate(code_ids, order, matrix))


if __name__ == "__main__":
    main()

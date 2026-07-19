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

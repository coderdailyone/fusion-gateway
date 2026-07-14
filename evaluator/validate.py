"""Small end-to-end validation: run a few tasks per source on a real model
through the runner, score them, freeze outputs, and report accuracy + cost.

This is the M2a acceptance check. It makes real API calls, so it is run
manually with keys (never in CI). Code-track tasks execute model-generated
code in the sandbox — run this on an isolated machine, not a production host.

Usage:
    .venv/bin/python -m evaluator.validate            # default: deepseek, 5/source
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from evaluator.suite.manifest import load
from evaluator.suite.loader import load_suite
from evaluator.hf_fetchers import make_fetcher
from evaluator.runner import run_one
from evaluator.store import new_run_dir, append_frozen, read_frozen
from evaluator.report import ResultRow, aggregate
from evaluator.scorers import mcq, code
from evaluator.scorers import math as math_scorer

SCORERS = {"mmlu_pro": mcq.score, "math": math_scorer.score, "humaneval": code.score}


def load_secrets(path: str = "runs/secrets/.env") -> None:
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def make_completion_fn(litellm_model: str, **overrides):
    """Return completion_fn(model, prompt) -> {text,in_tokens,out_tokens,cost_usd}
    that calls the given litellm model. `overrides` (api_base/api_key) are passed through."""
    import litellm
    litellm.suppress_debug_info = True

    def fn(model: str, prompt: str) -> dict:
        resp = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            **overrides,
        )
        try:
            cost = litellm.completion_cost(resp)
        except Exception:
            cost = 0.0
        return {
            "text": resp.choices[0].message.content or "",
            "in_tokens": resp.usage.prompt_tokens,
            "out_tokens": resp.usage.completion_tokens,
            "cost_usd": cost or 0.0,
        }

    return fn


def first_n_per_source(tasks, n):
    seen: dict[str, int] = {}
    out = []
    for t in sorted(tasks, key=lambda t: (t.source, t.id)):
        if seen.get(t.source, 0) < n:
            out.append(t)
            seen[t.source] = seen.get(t.source, 0) + 1
    return out


def validate(gateway_model_name: str, completion_fn, n_per_source: int = 5,
             manifest_path: str = "configs/suite.manifest.json"):
    load_secrets()
    m = load(manifest_path)
    fetchers = {s.name: make_fetcher(s.name) for s in m.sources}
    tasks = first_n_per_source(load_suite(m, fetchers), n_per_source)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir("evaluator", f"validate_{gateway_model_name}", ts)

    rows = []
    for t in tasks:
        fo = run_one(t, gateway_model_name, completion_fn)
        append_frozen(run_dir, fo)
        correct = False
        if fo.status == "ok":
            correct = SCORERS[t.source](t, fo.output_text).correct
        rows.append(ResultRow(t.id, t.source, gateway_model_name, correct, fo.cost_usd))
        print(f"  [{t.source:10}] {t.id:22} status={fo.status:5} correct={correct} "
              f"cost=${fo.cost_usd:.6f}")

    # prove frozen outputs are re-loadable (re-scorable with zero new calls)
    reload_n = len(read_frozen(run_dir))
    print(f"\nfrozen outputs written + reloaded: {reload_n} (run_dir={run_dir})")

    agg = aggregate(rows)
    print("\n=== aggregate ===")
    for model, a in agg.items():
        print(f"  {model}: n={a['n']} accuracy={a['accuracy']:.2f} "
              f"total_cost=${a['total_cost_usd']:.6f}")
    # per-source accuracy
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r.source, []).append(r.correct)
    print("=== per-source accuracy ===")
    for src, cs in sorted(by_src.items()):
        print(f"  {src}: {sum(cs)}/{len(cs)}")
    return rows, agg, run_dir


def main() -> None:
    fn = make_completion_fn("deepseek/deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"))
    validate("deepseek-chat", fn, n_per_source=5)


if __name__ == "__main__":
    load_secrets()
    main()

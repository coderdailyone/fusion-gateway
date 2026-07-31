"""M6 driver: k-sample the domestic panel for self-consistency voting.

Subcommands:
    probe    PAID (pennies) — sample ONE task k times per model and report
             whether the provider actually honours `temperature`, i.e. whether
             the k samples differ at all. A model that returns k byte-identical
             texts contributes ONE effective vote, not k.
    sample   PAID, budget-gated, resumable — k samples of every (task, model)
             over a stratified subset, all frozen.
    report   FREE ($0) — read the frozen samples back and report cost/task,
             the spoiled-ballot rate (evaluator.consistency.normalize.ballot_key)
             and the projected cost of the full run.

Discipline:
  * DOMESTIC panel only (deepseek-chat / glm-5.2 / kimi-k3). No frontier model
    takes part in voting — the headline claim would be circular.
  * `make_completion_fn` ignores its model-name argument and always calls the
    model it was bound to, so one fn is built per model and `sample_many`
    routes by name (it hands each fn only its own slice).
  * Secrets come from runs/secrets/.env; provider errors are redacted before
    printing (litellm error strings can echo the request, key included).
  * Code ballots are keyed by EXECUTING the problem's public doctests, never by
    task.tests — that field is HumanEval's official grader.

Typical run:
    .venv/bin/python -m scripts.run_consistency probe
    .venv/bin/python -m scripts.run_consistency sample --out smoke --tasks 20 \
        --k 3 --ceiling 1.0
    .venv/bin/python -m scripts.run_consistency report --out smoke
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PANEL = ["deepseek-chat", "glm-5.2", "kimi-k3"]        # domestic only
RUN_ROOT = Path("evaluator/runs/m6_consistency")
TEMPERATURE = 0.8
# Models whose provider REJECTS `temperature`, established by `probe`:
# kimi-k3's endpoint answers HTTP 400 "invalid temperature: only 1 is allowed
# for this model" — so it is sampled at its own default (which the probe showed
# already yields 3/3 distinct outputs; its ballots are real votes, not clones).
NO_TEMPERATURE: set[str] = {"kimi-k3"}
FULL_K = 10          # planned k for the full run (projection only)
FULL_CEILING = 40.0  # M6 budget backstop for the full run
SECRET_ENV = ("DEEPSEEK_API_KEY", "GLM_API_KEY", "MOONSHOT_API_KEY",
              "OPENAI_MIRROR_KEY", "CLAUDE_MIRROR_KEY", "HF_TOKEN")


def _redact(text: str) -> str:
    """Never let a provider error string leak an API key into the log."""
    out = str(text)
    for name in SECRET_ENV:
        value = os.environ.get(name)
        if value and len(value) > 6:
            out = out.replace(value, f"<{name}>")
    return out


def _load_tasks(manifest_path: str = "configs/suite.manifest.json"):
    from evaluator.suite.manifest import load
    from evaluator.suite.loader import load_suite
    from evaluator.hf_fetchers import make_fetcher

    m = load(manifest_path)
    fetchers = {s.name: make_fetcher(s.name) for s in m.sources}
    return sorted(load_suite(m, fetchers), key=lambda t: (t.source, t.id))


def _fns(models: list[str], temperature: float | None):
    """model -> completion_fn, with temperature bound per model."""
    from evaluator import validate

    validate.load_secrets()
    out = {}
    for name in models:
        extra = {}
        if temperature is not None and name not in NO_TEMPERATURE:
            extra["temperature"] = temperature
        out[name] = validate.make_model_fn(name, **extra)
    return out


def _shard(tasks: list, spec: str | None) -> list:
    """`--shard i/n`: keep every n-th task starting at i (disjoint slices)."""
    if not spec:
        return tasks
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {spec}: need 0 <= i < n")
    return [t for j, t in enumerate(tasks) if j % n == i]


# --------------------------------------------------------------------------
# probe


def _probe_once(task, model: str, k: int, temperature: float | None, out_dir):
    """k samples of one (task, model); returns (frozen rows, finding dict)."""
    from evaluator.consistency.sampler import sample_task, freezer

    append = freezer(out_dir)
    rows = []

    def collect(fo):
        append(fo)              # freeze first — a probe sample is paid data too
        rows.append(fo)

    texts = sample_task(task, model, k, _fns([model], temperature)[model],
                        on_frozen=collect)
    errors = [r.error for r in rows if r.status != "ok"]
    ok_texts = [r.output_text for r in rows if r.status == "ok"]
    distinct = len(set(ok_texts))
    return rows, {
        "temperature": temperature,
        "k": k,
        "ok": len(ok_texts),
        "distinct_outputs": distinct,
        "identical": bool(ok_texts) and distinct == 1,
        "lengths": [len(t) for t in texts],
        "errors": [_redact(e)[:300] for e in errors],
        "provider_cost_usd": sum(r.cost_usd for r in rows),
        "latency_ms": [r.latency_ms for r in rows],
    }


def cmd_probe(models: list[str], task_id: str | None, k: int,
              temperature: float | None) -> int:
    tasks = _load_tasks()
    if task_id:
        picked = next((t for t in tasks if t.id == task_id), None)
        if picked is None:
            raise SystemExit(f"no task {task_id!r} in the suite")
    else:                                    # cheap, short, high-variance
        picked = next(t for t in tasks if t.source == "math")
    print(f"probe task: {picked.source}/{picked.id}  k={k}  "
          f"temperature={temperature}", flush=True)

    out_dir = RUN_ROOT / "probe"
    findings = {}
    for name in models:
        rows, finding = _probe_once(picked, name, k, temperature, out_dir)
        if finding["ok"] == 0 and temperature is not None:
            # A reasoning model may REJECT the parameter outright. Drop it and
            # retry once — its own sampling variance is then the only source
            # of ballot diversity, which the report must state plainly.
            print(f"  {name:14} temp={temperature} all {k} calls failed: "
                  f"{_redact(finding['errors'][0] if finding['errors'] else '')[:200]}")
            print(f"  {name:14} retrying WITHOUT temperature ...", flush=True)
            rows, retry = _probe_once(picked, name, k, None, out_dir)
            retry["rejected_temperature"] = True
            retry["temperature_error"] = finding["errors"][:1]
            finding = retry
        findings[name] = finding
        verdict = ("ERROR (see errors[])" if finding["ok"] == 0 else
                   "IDENTICAL -> k ballots collapse to 1 vote"
                   if finding["identical"] else
                   f"varies ({finding['distinct_outputs']}/{finding['ok']} distinct)")
        print(f"  {name:14} temp={finding['temperature']} "
              f"ok={finding['ok']}/{k} lens={finding['lengths']} "
              f"-> {verdict}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(
        {"task": {"id": picked.id, "source": picked.source},
         "k": k, "requested_temperature": temperature,
         "findings": findings}, indent=2))
    print(f"\nwrote {out_dir / 'probe.json'}")
    return 0


# --------------------------------------------------------------------------
# sample


def cmd_sample(models: list[str], out: str, n_tasks: int, k: int,
               ceiling: float, temperature: float | None, seed: int,
               shard: str | None, tag: str) -> int:
    from evaluator import pricing
    from evaluator.consistency.sampler import sample_many
    from evaluator.pilot import stratified_subset

    all_tasks = _load_tasks()
    tasks = (all_tasks if n_tasks >= len(all_tasks)
             else stratified_subset(all_tasks, n_tasks, seed=seed))
    tasks = _shard(tasks, shard)
    by_source: dict[str, int] = {}
    for t in tasks:
        by_source[t.source] = by_source.get(t.source, 0) + 1

    root = RUN_ROOT / out
    root.mkdir(parents=True, exist_ok=True)
    print(f"sampling {len(tasks)} tasks {by_source} x {len(models)} models "
          f"x k={k} -> {root}")
    print(f"  models={models} temperature={temperature} "
          f"(no-temp: {sorted(NO_TEMPERATURE) or 'none'}) ceiling=${ceiling}")

    (root / "config.json").write_text(json.dumps(
        {"tasks": [t.id for t in tasks], "by_source": by_source, "k": k,
         "models": models, "temperature": temperature,
         "no_temperature": sorted(NO_TEMPERATURE), "seed": seed,
         "ceiling_usd": ceiling, "shard": shard, "tag": tag}, indent=2))

    res = sample_many(_fns(models, temperature), tasks, root, k, ceiling,
                      pricing.cost, tag=tag)
    print(f"DONE new_samples={res['completed']} spent=${res['spent']:.4f} "
          f"stopped_early={res['stopped']}")
    return 0


# --------------------------------------------------------------------------
# report (free)


def cmd_report(out: str, k: int) -> int:
    from evaluator import pricing
    from evaluator.consistency.normalize import ballot_key
    from evaluator.consistency.sampler import frozen_by_pair
    from evaluator.sandbox import run_code

    root = RUN_ROOT / out
    tasks = _load_tasks()
    by_id = {t.id: t for t in tasks}
    full_by_source: dict[str, int] = {}
    for t in tasks:
        full_by_source[t.source] = full_by_source.get(t.source, 0) + 1

    pairs = frozen_by_pair(root)
    if not pairs:
        raise SystemExit(f"no frozen samples under {root}")

    rows = [fo for samples in pairs.values() for fo in samples]
    task_ids = sorted({fo.task_id for fo in rows})
    models = sorted({fo.model for fo in rows})
    total_cost = sum(pricing.cost(fo.model, fo.in_tokens, fo.out_tokens)
                     for fo in rows)
    n_err = sum(1 for fo in rows if fo.status != "ok")

    print(f"=== {root} ===")
    print(f"  frozen samples: {len(rows)}  pairs: {len(pairs)}  "
          f"tasks: {len(task_ids)}  models: {models}")
    print(f"  error rows: {n_err} ({n_err / len(rows):.1%})")
    short = {p: len(v) for p, v in pairs.items() if len(v) < k}
    if short:
        print(f"  pairs with < k={k} samples: {len(short)}")

    # --- spoiled ballots ---------------------------------------------------
    spoiled = 0
    per_source: dict[str, list[int]] = {}
    for fo in rows:
        task = by_id[fo.task_id]
        # runner is MANDATORY for code tasks — ballot_key raises without it,
        # deliberately, so a forgotten runner cannot silently degrade the whole
        # code tier to raw-text plurality.
        bad = int(ballot_key(task, fo.output_text, runner=run_code) is None)
        spoiled += bad
        per_source.setdefault(task.source, []).append(bad)
    print("\n=== spoiled ballots (ballot_key -> None) ===")
    print(f"  overall: {spoiled}/{len(rows)} = {spoiled / len(rows):.2%}")
    for src in sorted(per_source):
        v = per_source[src]
        print(f"    {src:12} {sum(v)}/{len(v)} = {sum(v) / len(v):.2%}")

    # --- cost --------------------------------------------------------------
    print("\n=== cost (configs/pricing.toml) ===")
    per_model_source: dict[tuple[str, str], list[float]] = {}
    for fo in rows:
        c = pricing.cost(fo.model, fo.in_tokens, fo.out_tokens)
        per_model_source.setdefault((fo.model, by_id[fo.task_id].source), []).append(c)
    for model in models:
        mrows = [fo for fo in rows if fo.model == model]
        mcost = sum(pricing.cost(fo.model, fo.in_tokens, fo.out_tokens) for fo in mrows)
        out_tok = sum(fo.out_tokens for fo in mrows)
        print(f"  {model:14} samples={len(mrows):4d} cost=${mcost:.4f} "
              f"${mcost / len(mrows):.5f}/sample  mean_out_tok="
              f"{out_tok / len(mrows):.0f}")
    print(f"  TOTAL ${total_cost:.4f} over {len(task_ids)} tasks "
          f"= ${total_cost / len(task_ids):.5f}/task "
          f"(all {len(models)} models, k~{len(rows) / max(1, len(pairs)):.1f})")

    # --- projection --------------------------------------------------------
    # per (model, source) mean cost per sample, weighted by the FULL suite's
    # source mix, times FULL_K samples.
    projected = 0.0
    missing = []
    for model in models:
        for src, n_full in full_by_source.items():
            got = per_model_source.get((model, src))
            if not got:
                missing.append((model, src))
                continue
            projected += (sum(got) / len(got)) * FULL_K * n_full
    print(f"\n=== projected full run ({len(models)} models x k={FULL_K} x "
          f"{sum(full_by_source.values())} tasks) ===")
    print(f"  projected cost = ${projected:.2f}   ceiling = ${FULL_CEILING:.2f} "
          f"-> {'WITHIN' if projected <= FULL_CEILING else 'OVER'} budget")
    if missing:
        print(f"  (no smoke samples for {missing} — projection extrapolates "
              "only over the observed cells)")
    print(f"  scale factor to fit ${FULL_CEILING:.0f}: "
          f"k_max ~ {FULL_CEILING / (projected / FULL_K):.1f} samples/model")

    summary = {
        "root": str(root), "samples": len(rows), "tasks": len(task_ids),
        "models": models, "error_rows": n_err,
        "spoiled": spoiled, "spoiled_rate": spoiled / len(rows),
        "spoiled_by_source": {s: {"n": len(v), "spoiled": sum(v)}
                              for s, v in per_source.items()},
        "total_cost_usd": total_cost,
        "cost_per_task_usd": total_cost / len(task_ids),
        "projected_full_run_usd": projected,
        "full_k": FULL_K, "full_ceiling_usd": FULL_CEILING,
    }
    (root / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {root / 'report.json'}")
    return 0


# --------------------------------------------------------------------------


def _arg(args: list[str], flag: str, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    args = sys.argv[2:]
    models = (_arg(args, "--models") or ",".join(PANEL)).split(",")
    bad = [m for m in models if m not in PANEL]
    if bad:
        raise SystemExit(f"non-domestic model(s) {bad}; panel is {PANEL}")
    temperature = float(_arg(args, "--temperature", TEMPERATURE))
    if "--no-temperature" in args:
        temperature = None

    if cmd == "probe":
        return cmd_probe(models, _arg(args, "--task-id"),
                         int(_arg(args, "--k", 3)), temperature)
    if cmd == "sample":
        return cmd_sample(
            models,
            out=_arg(args, "--out", "smoke"),
            n_tasks=int(_arg(args, "--tasks", 20)),
            k=int(_arg(args, "--k", 3)),
            ceiling=float(_arg(args, "--ceiling", 1.0)),
            temperature=temperature,
            seed=int(_arg(args, "--seed", 1234)),
            shard=_arg(args, "--shard"),
            tag=_arg(args, "--tag", ""),
        )
    if cmd == "report":
        return cmd_report(_arg(args, "--out", "smoke"), int(_arg(args, "--k", 3)))
    print(f"unknown command {cmd!r}; use 'probe', 'sample' or 'report'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Budget-gated official-pipeline re-sample over all feasible models.

Real API calls -> run manually on an isolated host with keys in
runs/secrets/.env. build_prompt already emits official 0-shot-CoT prompts, so
run_pilot/sample re-sample the locked suite through the official pipeline.
Writes a NEW run dir; already-frozen (task, model) pairs are skipped.
"""
from __future__ import annotations

FEASIBLE_MODELS = [
    "deepseek-chat", "claude-sonnet-5", "claude-opus-4-8",
    "gpt-5.6-sol", "gpt-5.5", "glm-5.2", "kimi-k2",
]
BUDGET_CEILING_USD = 35.0
WARN_FRACTION = 0.80


def estimate_cost(rows_est, cost_fn) -> float:
    return sum(cost_fn(m, i, o) for (m, i, o) in rows_est)


def budget_gate(spent: float, next_cost: float, ceiling: float) -> str:
    if spent + next_cost > ceiling:
        return "stop"
    if spent >= WARN_FRACTION * ceiling:
        return "warn"
    return "ok"


def main() -> None:
    import sys
    from datetime import datetime, timezone
    from evaluator import validate
    from evaluator.store import new_run_dir

    validate.load_secrets()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1063  # full locked suite
    models = [m for m in FEASIBLE_MODELS if m in validate.MODELS]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir("evaluator", "official_resample", ts)
    print(f"re-sampling {n} tasks x {len(models)} models -> {run_dir}")
    print(f"budget ceiling ${BUDGET_CEILING_USD}; warn at {int(WARN_FRACTION*100)}%")
    from evaluator.pilot import run_pilot
    run_pilot(n=n, run_dir=run_dir, model_names=models)


if __name__ == "__main__":
    main()

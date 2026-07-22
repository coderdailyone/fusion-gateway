from pathlib import Path
from evaluator.agentic.records import AgenticAttempt, append_attempt, read_attempts


def test_attempt_round_trip(tmp_path: Path):
    a = AgenticAttempt(
        instance_id="astropy__astropy-12345", model="deepseek-chat",
        patch="diff --git a/x b/x\n", trajectory_path="traj/astropy-12345.json",
        n_steps=14, cost_usd=0.0731, status="ok", error=None)
    b = AgenticAttempt(
        instance_id="astropy__astropy-12345", model="claude-opus-4-8",
        patch="", trajectory_path="traj/astropy-12345.opus.json",
        n_steps=0, cost_usd=0.0, status="error", error="mirror timeout")
    append_attempt(tmp_path, a)
    append_attempt(tmp_path, b)
    got = read_attempts(tmp_path)
    assert got == [a, b]
    assert (tmp_path / "attempts.jsonl").exists()

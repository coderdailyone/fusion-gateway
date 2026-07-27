from evaluator.fusion.panel import PanelCase, assemble, load_frozen_by_model
from evaluator.runner import FrozenOutput
from evaluator.store import append_frozen


FROZEN = {
    "deepseek-chat": {"t1": "answer A", "t2": "ds2", "t3": "ds3"},
    "glm-5.2":       {"t1": "answer B", "t2": "glm2"},
    "kimi-k3":       {"t1": "answer C"},
}


def test_assemble_builds_cases_with_all_available_candidates():
    cases, excluded = assemble(FROZEN, ["t1", "t2", "t3"])
    by_id = {c.task_id: c for c in cases}
    assert set(by_id["t1"].candidates) == {"deepseek-chat", "glm-5.2", "kimi-k3"}
    assert by_id["t1"].candidates["glm-5.2"] == "answer B"
    # t2 has only two candidates -> still a valid (degraded) panel
    assert set(by_id["t2"].candidates) == {"deepseek-chat", "glm-5.2"}


def test_task_with_fewer_than_min_candidates_is_excluded_not_crashed():
    cases, excluded = assemble(FROZEN, ["t1", "t2", "t3"], min_candidates=2)
    assert "t3" in excluded                      # only deepseek answered
    assert all(c.task_id != "t3" for c in cases)


def test_unknown_task_id_is_excluded():
    cases, excluded = assemble(FROZEN, ["nope"])
    assert excluded == ["nope"] and cases == []


def test_load_frozen_by_model_keeps_ok_rows_and_first_of_duplicates(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    rows = [
        FrozenOutput(task_id="t1", source="mmlu_pro", model="deepseek-chat",
                     prompt="p1", output_text="first answer", in_tokens=1,
                     out_tokens=1, cost_usd=0.0, latency_ms=10, status="ok",
                     error=None),
        FrozenOutput(task_id="t2", source="mmlu_pro", model="deepseek-chat",
                     prompt="p2", output_text="", in_tokens=1, out_tokens=0,
                     cost_usd=0.0, latency_ms=10, status="error", error="timeout"),
        # duplicate task_id "t1": the FIRST row must win, not the last.
        FrozenOutput(task_id="t1", source="mmlu_pro", model="deepseek-chat",
                     prompt="p1-retry", output_text="second answer (should be ignored)",
                     in_tokens=1, out_tokens=1, cost_usd=0.0, latency_ms=10,
                     status="ok", error=None),
    ]
    for row in rows:
        append_frozen(run_dir, row)

    out = load_frozen_by_model({"deepseek-chat": str(run_dir)})
    # error row (t2) dropped; duplicate t1's FIRST occurrence wins.
    assert out["deepseek-chat"] == {"t1": "first answer"}

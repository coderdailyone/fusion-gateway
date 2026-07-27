from evaluator.fusion.panel import PanelCase, assemble


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

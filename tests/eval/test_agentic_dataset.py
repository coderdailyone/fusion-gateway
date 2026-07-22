from evaluator.agentic.dataset import (
    Instance, filter_by_date, load_instances, stratified_sample)

ROWS = [
    {"instance_id": "a__a-1", "repo": "a/a", "base_commit": "c1",
     "problem_statement": "p1", "created_at": "2025-01-10T00:00:00Z",
     "image_ref": "img:a1", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": []},
    {"instance_id": "a__a-2", "repo": "a/a", "base_commit": "c2",
     "problem_statement": "p2", "created_at": "2025-09-01T00:00:00Z",
     "image_ref": "img:a2", "FAIL_TO_PASS": ["t2"], "PASS_TO_PASS": []},
    {"instance_id": "b__b-1", "repo": "b/b", "base_commit": "c3",
     "problem_statement": "p3", "created_at": "2025-10-01T00:00:00Z",
     "image_ref": "img:b1", "FAIL_TO_PASS": ["t3"], "PASS_TO_PASS": []},
]


def test_filter_by_date_keeps_only_post_cutoff():
    kept = filter_by_date(ROWS, "2025-06-01")
    assert [r["instance_id"] for r in kept] == ["a__a-2", "b__b-1"]


def test_load_instances_maps_fields_and_preserves_raw():
    insts = load_instances(ROWS)
    assert insts[0].instance_id == "a__a-1"
    assert insts[0].repo == "a/a"
    assert insts[0].image_ref == "img:a1"
    # hidden test fields survive in raw (for the grader) but are not surfaced
    assert insts[0].raw["FAIL_TO_PASS"] == ["t1"]
    assert not hasattr(insts[0], "FAIL_TO_PASS")


def test_stratified_sample_is_deterministic_and_spreads_repos():
    insts = load_instances(ROWS)
    s1 = stratified_sample(insts, k=2, seed="m4")
    s2 = stratified_sample(insts, k=2, seed="m4")
    assert [i.instance_id for i in s1] == [i.instance_id for i in s2]  # deterministic
    assert len({i.repo for i in s1}) == 2  # spread across both repos before repeating

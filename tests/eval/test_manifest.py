from evaluator.suite.types import Task
from evaluator.suite.manifest import SourceSpec, Manifest, content_sha, save, load

def test_task_is_frozen():
    t = Task(id="a", source="mmlu_pro", problem="2+2?", answer="4", tests=(), meta={})
    assert t.answer == "4"
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.answer = "5"

def test_content_sha_stable_and_order_independent():
    a = [{"id": "1", "q": "x"}, {"id": "2", "q": "y"}]
    assert content_sha(a) == content_sha(list(a))          # stable
    assert content_sha(a) != content_sha([{"id": "1", "q": "z"}])  # sensitive to content

def test_manifest_roundtrip(tmp_path):
    m = Manifest(version=1, sources=(
        SourceSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", "abc123", "test", ("q1", "q2"), "deadbeef"),
    ))
    p = tmp_path / "manifest.json"
    save(m, p)
    m2 = load(p)
    assert m2 == m
    assert m2.sources[0].task_ids == ("q1", "q2")

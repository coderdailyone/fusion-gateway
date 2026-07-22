import json
from evaluator.agentic.records import AgenticAttempt
from evaluator.agentic.verifier import VerifierResult
from evaluator.agentic.cascade import CascadeResult
from evaluator.agentic.grade import write_predictions

V = VerifierResult(True, True, True, True, True, False)


def test_write_predictions_swebench_shape(tmp_path):
    r = CascadeResult(
        "astropy__astropy-1", "diff --git a/f b/f\n+x\n", True, 1.1,
        AgenticAttempt("astropy__astropy-1", "deepseek-chat", "", "t", 3, 0.1, "ok", None),
        AgenticAttempt("astropy__astropy-1", "claude-opus-4-8", "diff --git a/f b/f\n+x\n", "t2", 7, 1.0, "ok", None),
        V)
    p = tmp_path / "predictions.jsonl"
    write_predictions([r], model_name="cascade-deepseek-opus", path=p)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert rows == [{
        "instance_id": "astropy__astropy-1",
        "model_name_or_path": "cascade-deepseek-opus",
        "model_patch": "diff --git a/f b/f\n+x\n",
    }]

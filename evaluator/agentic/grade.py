"""Write SWE-bench predictions and (Phase B) invoke the official harness."""
from __future__ import annotations

import json
from pathlib import Path


def write_predictions(results, model_name: str, path) -> None:
    """Write one SWE-bench prediction object per instance to predictions.jsonl."""
    path = Path(path)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "instance_id": r.instance_id,
                "model_name_or_path": model_name,
                "model_patch": r.accepted_patch,
            }) + "\n")


def grade(predictions_path, instances, box) -> dict:
    """Run the official SWE-bench-Live evaluation harness on the eval box.

    Implemented in Phase B (Task 9): invokes the harness subprocess over `box`
    (an ssh/exec handle), parses its report json, and returns
    {instance_id: resolved_bool}. Never re-implements grading logic — it shells
    out to the official harness so `resolved` is upstream-defined.
    """
    raise NotImplementedError("grade() is wired on the Docker box in Task 9")

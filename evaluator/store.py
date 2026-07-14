"""Frozen-output JSONL store: persist and retrieve execution records."""

import json
from dataclasses import asdict
from pathlib import Path

from evaluator.runner import FrozenOutput


def new_run_dir(base, suite_name: str, clock_now: str) -> Path:
    """Create and return the run directory path.

    Args:
        base: Base path for runs directory
        suite_name: Name of the test suite
        clock_now: Timestamp string

    Returns:
        Path to the newly created run directory
    """
    run_dir = Path(base) / "runs" / f"{suite_name}_{clock_now}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_frozen(run_dir, fo: FrozenOutput) -> None:
    """Append one frozen output as a JSON line to the frozen.jsonl file.

    Args:
        run_dir: Path to the run directory
        fo: FrozenOutput object to append
    """
    jsonl_path = run_dir / "frozen.jsonl"
    line = json.dumps(asdict(fo))
    with open(jsonl_path, "a") as f:
        f.write(line + "\n")


def read_frozen(run_dir) -> list[FrozenOutput]:
    """Read frozen outputs from the JSONL file and reconstruct them.

    Args:
        run_dir: Path to the run directory

    Returns:
        List of FrozenOutput objects in order, preserving round-trip equality
    """
    jsonl_path = run_dir / "frozen.jsonl"
    results = []
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    results.append(FrozenOutput(**obj))
    return results

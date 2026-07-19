from __future__ import annotations
from typing import Callable

from evaluator.suite.types import Task
from evaluator.suite.manifest import Manifest, SourceSpec, content_sha

Fetcher = Callable[[SourceSpec], list[dict]]


class SuiteHashMismatch(Exception):
    pass


def _letters():
    i = 0
    while True:
        yield chr(ord("A") + i)
        i += 1


def parse_task(source: str, record: dict) -> Task:
    rec = dict(record)
    task_id = str(rec["id"])

    if source == "mmlu_pro":
        question = rec.pop("question")
        options = rec.pop("options")
        answer = rec.pop("answer")
        lines = [question, ""]
        for letter, opt in zip(_letters(), options):
            lines.append(f"{letter}. {opt}")
        problem = "\n".join(lines)
        return Task(id=task_id, source=source, problem=problem, answer=answer,
                    tests=(), meta=rec)

    if source == "math":
        problem = rec.pop("problem")
        answer = rec.pop("answer")
        return Task(id=task_id, source=source, problem=problem, answer=answer,
                    tests=(), meta=rec)

    if source == "humaneval":
        problem = rec.pop("prompt")
        test = rec.pop("test")
        entry_point = rec.pop("entry_point")
        return Task(id=task_id, source=source, problem=problem, answer=None,
                    tests=({"kind": "pyfunc", "test": test, "entry_point": entry_point},),
                    meta=rec)

    if source == "livecodebench":
        problem = rec.pop("prompt", None)
        if problem is None:
            problem = rec.pop("question", None)
        if problem is None:
            problem = rec.pop("problem")
        tests = tuple(rec.pop("tests"))
        return Task(id=task_id, source=source, problem=problem, answer=None,
                    tests=tests, meta=rec)

    raise ValueError(f"unknown source: {source!r}")


def load_suite(manifest: Manifest, fetchers: dict[str, Fetcher]) -> list[Task]:
    tasks: list[Task] = []
    for source in manifest.sources:
        records = fetchers[source.name](source)
        if content_sha(records) != source.content_sha:
            raise SuiteHashMismatch(
                f"content hash mismatch for source {source.name!r}"
            )
        for record in records:
            tasks.append(parse_task(source.name, record))
    return tasks

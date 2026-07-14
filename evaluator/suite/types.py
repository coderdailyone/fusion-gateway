from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    id: str
    source: str
    problem: str
    answer: str | None
    tests: tuple[dict, ...]
    meta: dict

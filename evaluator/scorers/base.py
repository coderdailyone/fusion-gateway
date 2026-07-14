from dataclasses import dataclass

@dataclass(frozen=True)
class Score:
    correct: bool
    detail: dict

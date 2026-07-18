from evaluator.suite.types import Task
from evaluator.scorers.base import Score
from evaluator.official.mmlu_extract import extract_answer


def score(task: Task, output_text: str) -> Score:
    letter = extract_answer(output_text)
    if letter is None:
        return Score(False, {"reason": "unparseable", "method": "none"})
    expected = (task.answer or "").upper()
    return Score(letter.upper() == expected,
                 {"extracted": letter.upper(), "expected": expected, "method": "official"})

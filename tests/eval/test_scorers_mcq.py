from evaluator.suite.types import Task
from evaluator.scorers.mcq import score

T = Task(id="q", source="mmlu_pro", problem="...", answer="B", tests=(), meta={})

def test_correct_various_formats():
    for out in ["The answer is B.", "Final: (B)", "\\boxed{B}", "... so B"]:
        assert score(T, out).correct

def test_answer_cue_formats():
    # Regression: real reasoning-model outputs (observed from Kimi) that the
    # old scorer missed — "Answer: B. <option text>", markdown-wrapped, etc.
    for out in [
        "reasoning...\n\n**Answer: B. South pole, 2000 poles**",
        "Answer: B",
        "The correct answer is B.",
        "Final answer: (B)",
        "So the correct answer = B.",
        "Therefore option B is correct.",
    ]:
        assert score(T, out).correct, out
    # and it still rejects a different letter in the same shape
    assert not score(T, "**Answer: A. 30 m**").correct

def test_markdown_and_newline_between_cue_and_letter():
    # Regression from GLM-5.2: "**Final Answer:**\nB. South pole" — stars and a
    # newline sit between the cue and the letter.
    assert score(T, "reasoning...\n\n**Final Answer:**\nB. South pole, 2000 poles").correct
    assert score(T, "the correct option is B.").correct
    assert not score(T, "**Final Answer:**\nA. 30 m").correct

def test_incorrect():
    assert not score(T, "The answer is A.").correct

def test_unparseable():
    s = score(T, "I am not sure.")
    assert not s.correct and s.detail["reason"] == "unparseable"

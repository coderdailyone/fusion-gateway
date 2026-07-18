from evaluator.suite.types import Task
from evaluator.scorers.mcq import score


def T(ans: str) -> Task:
    return Task(id="q", source="mmlu_pro", problem="...", answer=ans, tests=(), meta={})


def test_correct_various_formats():
    for out in ["The answer is B.", "Final: (B)", "\\boxed{B}", "... so B"]:
        assert score(T("B"), out).correct

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
        assert score(T("B"), out).correct, out
    # and it still rejects a different letter in the same shape
    assert not score(T("B"), "**Answer: A. 30 m**").correct

def test_markdown_and_newline_between_cue_and_letter():
    # Regression from GLM-5.2: "**Final Answer:**\nB. South pole" — stars and a
    # newline sit between the cue and the letter.
    assert score(T("B"), "reasoning...\n\n**Final Answer:**\nB. South pole, 2000 poles").correct
    assert score(T("B"), "the correct option is B.").correct
    assert not score(T("B"), "**Final Answer:**\nA. 30 m").correct

def test_incorrect():
    assert not score(T("B"), "The answer is A.").correct

def test_unparseable():
    # NOTE: was "I am not sure." — but under the official letter-only chain,
    # a bare "I" IS a valid MCQ option letter (A-J includes I), so that input
    # is no longer a fair unparseable fixture. Use text with no A-J letter.
    s = score(T("B"), "Not sure at all.")
    assert not s.correct and s.detail["reason"] == "unparseable"

def test_official_format_extracts():
    assert score(T("A"), "After analysis, the answer is (A).").correct
    assert not score(T("A"), "The answer is (B).").correct

def test_value_only_is_unparseable_under_official():
    # historical GLM value-boxing: no letter present -> official cannot parse
    r = score(T("A"), "\\boxed{30\\text{ m}}")
    assert not r.correct

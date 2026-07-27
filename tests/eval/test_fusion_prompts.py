from evaluator.suite.types import Task
from evaluator.fusion.panel import PanelCase
from evaluator.fusion.prompts import (build_fusion_prompt, build_review_prompt,
                                      format_instruction)
from evaluator.fusion.review import Verdict

MCQ = Task(id="t1", source="mmlu_pro", problem="What is 2+2?\nA) 3\nB) 4",
           answer="B", tests=(), meta={})
MATH = Task(id="t2", source="math", problem="Compute 2+2.", answer="4",
            tests=(), meta={})
CASE = PanelCase(task_id="t1", source="mmlu_pro",
                 candidates={"deepseek-chat": "I say (A).", "glm-5.2": "I say (B)."})


def test_format_instruction_matches_official_wording():
    assert "The answer is (X)." in format_instruction(MCQ)
    assert "\\boxed{}" in format_instruction(MATH)


def test_review_prompt_excludes_the_reviewers_own_answer():
    p = build_review_prompt(MCQ, CASE, reviewer="deepseek-chat")
    assert "I say (B)." in p          # the other candidate is reviewed
    assert "I say (A)." not in p      # its own answer is NOT shown (no self-review)
    assert "correct" in p and "wrong" in p and "unsure" in p   # verdict vocabulary


def test_fusion_prompt_carries_candidates_reviews_and_format():
    reviews = {"deepseek-chat": {"glm-5.2": Verdict("wrong", "B is not 4")}}
    p = build_fusion_prompt(MCQ, CASE, reviews)
    assert "I say (A)." in p and "I say (B)." in p     # all candidates present
    assert "B is not 4" in p                           # review evidence present
    assert "The answer is (X)." in p                   # official output contract
    assert MCQ.problem in p

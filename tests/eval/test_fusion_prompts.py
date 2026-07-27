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


def _task(source: str) -> Task:
    return Task(id="x", source=source, problem="p", answer="a", tests=(), meta={})


def test_format_instruction_covers_every_manifest_source():
    # mmlu_pro / suite.manifest.json + suite.hard.manifest.json sources.
    mcq_sources = ("mmlu_pro", "gpqa_diamond")
    for source in mcq_sources:
        instr = format_instruction(_task(source))
        assert "The answer is (X)." in instr, source

    math_sources = ("math", "aime", "math_l5")
    for source in math_sources:
        instr = format_instruction(_task(source))
        assert "\\boxed{}" in instr, source

    humaneval_instr = format_instruction(_task("humaneval"))
    assert "full function definition" in humaneval_instr

    livecodebench_instr = format_instruction(_task("livecodebench"))
    assert "complete solution" in livecodebench_instr

    # humaneval and livecodebench must NOT share wording: HumanEval's
    # extractor needs the full function signature, LiveCodeBench's does not.
    assert humaneval_instr != livecodebench_instr

    # Unknown source falls back to the generic default.
    assert format_instruction(_task("some_unknown_source")) == (
        "Put your final answer clearly at the end."
    )


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


def test_fusion_prompt_forbids_early_answer_is_for_mcq_sources():
    # evaluator/official/mmlu_extract.py::extract_answer takes the FIRST
    # occurrence of "answer is". The fusion prompt tells the fuser to
    # "decide using the specific objections raised", which invites restating
    # a rejected candidate's answer (e.g. "Candidate X's answer is (A), but
    # that is wrong ... The answer is (B)."), extracting the wrong letter. For
    # MCQ-format tasks, the prompt must explicitly rule this out.
    p_mcq = build_fusion_prompt(MCQ, CASE, {})
    assert "final answer line" in p_mcq
    assert "answer is" in p_mcq.lower()


def test_fusion_prompt_omits_early_answer_is_rule_for_non_mcq_sources():
    math_case = PanelCase(task_id="t2", source="math",
                          candidates={"deepseek-chat": "2+2=4", "glm-5.2": "4"})
    p_math = build_fusion_prompt(MATH, math_case, {})
    assert "final answer line" not in p_math

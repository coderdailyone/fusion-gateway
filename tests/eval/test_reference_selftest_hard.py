from evaluator.suite.types import Task
from evaluator.audit.references import Reference
from evaluator.audit.reference_selftest import selftest_one

def _t(source, answer=None):
    return Task(id="q", source=source, problem="P", answer=answer, tests=(), meta={})

def test_gpqa_gold_recognized():
    assert selftest_one(_t("gpqa_diamond", "C"), Reference("q","gpqa_diamond", gold="C")) is None

def test_aime_gold_recognized():
    assert selftest_one(_t("aime", "204"), Reference("q","aime", gold="204", solution="\\boxed{204}")) is None

def test_math_l5_gold_recognized():
    assert selftest_one(_t("math_l5", "3"), Reference("q","math_l5", gold="3", solution="so \\boxed{3}")) is None

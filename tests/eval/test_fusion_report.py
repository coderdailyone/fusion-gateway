from evaluator.fusion.review import Verdict
from scripts.fusion_report import fix_break, gate_curve, oracle, reviewer_agreement


def test_oracle_is_fraction_where_any_model_is_correct():
    correct = {
        "m1": {"t1": True,  "t2": False, "t3": False},
        "m2": {"t1": False, "t2": True,  "t3": False},
    }
    assert oracle(correct) == 2 / 3        # t1 and t2 covered, t3 by nobody


def test_fix_break_counts_both_directions():
    base  = {"t1": False, "t2": True,  "t3": True,  "t4": False}
    fused = {"t1": True,  "t2": False, "t3": True,  "t4": False}
    r = fix_break(base, fused)
    assert r["fix"] == 1 and r["break_"] == 1 and r["net"] == 0 and r["n"] == 4


def test_reviewer_agreement_fraction_of_matching_verdicts_on_same_target():
    # task t1: two reviewers both call m3 "wrong" (agree);
    # task t2: reviewers disagree about m3
    reviews = {
        "t1": {"m1": {"m3": "wrong"}, "m2": {"m3": "wrong"}},
        "t2": {"m1": {"m3": "correct"}, "m2": {"m3": "wrong"}},
    }
    assert reviewer_agreement(reviews) == 0.5


def test_reviewer_agreement_normalizes_verdict_objects_ignoring_reason_text():
    # cross_review's REAL shape is {reviewer: {target: Verdict}}, not plain
    # strings. Verdict is a hashable frozen dataclass of (verdict, reason);
    # comparing raw Verdict objects would compare `reason` too, so two
    # reviewers who agree "wrong" for DIFFERENT free-text reasons must still
    # count as agreeing once normalized.
    reviews = {
        "t1": {
            "m1": {"m3": Verdict("wrong", "picked A but B is correct")},
            "m2": {"m3": Verdict("wrong", "the units don't match")},
        },
    }
    assert reviewer_agreement(reviews) == 1.0


def test_gate_curve_reports_agreement_gate_cheaper_than_always_fuse():
    cands = {"t1": {"a": "X", "b": "X"},     # agree -> gate can skip fusion
             "t2": {"a": "X", "b": "Y"}}     # disagree -> must fuse
    # t1 (the unanimous task) deliberately has fused_correct != baseline_correct
    # so the test can tell WHICH correctness the free-adopt path actually used
    # (T5): if on_disagreement wrongly used fused_correct instead of
    # baseline_correct for the free-adopt task, "correct" below would be 1, not 2.
    fused = {"t1": False, "t2": True}
    base  = {"t1": True,  "t2": False}
    cost  = {"t1": 0.01, "t2": 0.01}
    rows = {r["policy"]: r for r in gate_curve(cands, fused, base, cost)}
    assert rows["always"]["cost"] == 0.02 and rows["always"]["correct"] == 1
    # agreement gate: t1 free-adopts using BASELINE correctness (True), t2 pays
    # and uses fused correctness (True) -> 2 total.
    assert rows["on_disagreement"]["cost"] == 0.01
    assert rows["on_disagreement"]["correct"] == 2


def test_gate_curve_extract_param_decides_unanimity_on_parsed_answer():
    # Raw texts differ ("The answer is (B)." vs "I think B is correct."), so
    # without `extract` the raw-text comparison sees disagreement and must
    # fuse. With an `extract` that maps both to the same parsed letter "B",
    # the task becomes unanimous and is free-adopted instead.
    cands = {"t1": {"a": "The answer is (B).", "b": "I think B is correct."}}
    fused = {"t1": True}
    base  = {"t1": True}
    cost  = {"t1": 0.01}

    def extract(text):
        return "B" if "B" in text else None

    rows_raw = {r["policy"]: r for r in gate_curve(cands, fused, base, cost)}
    assert rows_raw["on_disagreement"]["cost"] == 0.01     # raw texts differ

    rows_extracted = {r["policy"]: r
                       for r in gate_curve(cands, fused, base, cost, extract=extract)}
    assert rows_extracted["on_disagreement"]["cost"] == 0.0     # extracted-unanimous
    assert rows_extracted["on_disagreement"]["correct"] == 1


def test_gate_curve_extract_none_never_counts_as_unanimous():
    # Both candidates fail to parse -> must NOT be treated as "agreeing".
    cands = {"t1": {"a": "garbled nonsense", "b": "also garbled"}}
    fused = {"t1": True}
    base  = {"t1": False}
    cost  = {"t1": 0.01}

    rows = {r["policy"]: r
            for r in gate_curve(cands, fused, base, cost, extract=lambda _t: None)}
    assert rows["on_disagreement"]["cost"] == 0.01      # forced to fuse, not free-adopt
    assert rows["on_disagreement"]["correct"] == 1      # uses fused_correct (True), not base


def test_gate_curve_review_cost_asymmetry_between_policies():
    # "always" pays review cost for every task; "on_disagreement" only pays
    # review cost for the tasks it actually fuses (t2) — candidate agreement
    # is knowable at $0 from the frozen candidates before any review call.
    cands = {"t1": {"a": "X", "b": "X"},      # unanimous -> free adopt, no review cost
             "t2": {"a": "X", "b": "Y"}}      # disagreement -> fused, review cost counted
    fused = {"t1": True, "t2": True}
    base  = {"t1": True, "t2": False}
    cost  = {"t1": 0.01, "t2": 0.01}
    review_cost = {"t1": 0.05, "t2": 0.05}

    rows = {r["policy"]: r
            for r in gate_curve(cands, fused, base, cost, review_cost=review_cost)}
    assert rows["always"]["cost"] == (0.01 + 0.01) + (0.05 + 0.05)
    assert rows["on_disagreement"]["cost"] == 0.01 + 0.05

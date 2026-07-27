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


def test_gate_curve_reports_agreement_gate_cheaper_than_always_fuse():
    cands = {"t1": {"a": "X", "b": "X"},     # agree -> gate can skip fusion
             "t2": {"a": "X", "b": "Y"}}     # disagree -> must fuse
    fused = {"t1": True, "t2": True}
    base  = {"t1": True, "t2": False}
    cost  = {"t1": 0.01, "t2": 0.01}
    rows = {r["policy"]: r for r in gate_curve(cands, fused, base, cost)}
    assert rows["always"]["cost"] == 0.02 and rows["always"]["correct"] == 2
    # agreement gate: t1 adopts the (correct) agreed answer free, t2 pays
    assert rows["on_disagreement"]["cost"] == 0.01
    assert rows["on_disagreement"]["correct"] == 2

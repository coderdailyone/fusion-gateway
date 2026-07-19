from scripts.hard_report import wilson_ci, mcnemar_p


def test_wilson_ci_basic():
    lo, hi = wilson_ci(90, 100)
    assert 0.0 < lo < 0.90 < hi < 1.0
    assert lo < hi


def test_wilson_ci_degenerate():
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_mcnemar_symmetric_is_insignificant():
    assert mcnemar_p(10, 10) > 0.5   # equal discordant -> not significant


def test_mcnemar_lopsided_is_significant():
    assert mcnemar_p(20, 2) < 0.05

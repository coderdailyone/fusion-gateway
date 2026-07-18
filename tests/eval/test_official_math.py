from evaluator.official.math_grade import is_equiv, math_equiv

def test_strip_equal_units():
    # official _remove_right_units strips a trailing \text{...} unit label
    assert is_equiv("3\\text{ treeks}", "3")

def test_strip_equal_fracs_and_sqrt():
    assert is_equiv("\\frac12", "\\frac{1}{2}")
    assert is_equiv("\\sqrt3", "\\sqrt{3}")
    assert is_equiv("\\tfrac{1}{2}", "\\frac{1}{2}")

def test_strip_equal_leading_var_assignment():
    # official strips a short "k =" / "x =" LHS (len(lhs) <= 2)
    assert is_equiv("x=5", "5")

def test_sympy_fallback_numeric():
    # string forms differ but are numerically equal -> sympy fallback
    assert math_equiv("0.5", "\\frac{1}{2}")
    assert math_equiv("1+1", "2")

def test_var_in_prefix_is_NOT_stripped():
    # deliberate: official does not strip "x \in"; our old leniency is dropped
    assert not is_equiv("[-2,7]", "x \\in [-2,7]")

def test_wrong():
    assert not is_equiv("41", "42")
    assert not math_equiv("[0,5]", "[-2,7]")

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

def test_var_in_prefix_is_NOT_stripped_by_byte_faithful_is_equiv():
    # is_equiv stays byte-faithful Hendrycks: it does NOT strip "x \in".
    assert not is_equiv("[-2,7]", "x \\in [-2,7]")

def test_modern_normalization_interval_prefix():
    # Regression from the paid smoke: every model boxed the bare interval
    # "[-2,7]" while the MATH gold is "x \in [-2,7]" — same answer. The modern
    # symmetric normalization in math_equiv strips the leading "x \in" (and the
    # \, thin-space gpt-5.6-sol used) on BOTH sides.
    assert math_equiv("[-2,7]", "x \\in [-2,7]")
    assert math_equiv("x \\in [-2,7]", "[-2,7]")        # symmetric
    assert math_equiv("[-2,\\,7]", "x \\in [-2,7]")     # LaTeX thin-space
    assert math_equiv("[-2, 7]", "x \\in [-2,7]")       # plain space
    # a genuinely different interval must still be wrong after normalization
    assert not math_equiv("[0,5]", "x \\in [-2,7]")

def test_wrong():
    assert not is_equiv("41", "42")
    assert not math_equiv("[0,5]", "[-2,7]")
